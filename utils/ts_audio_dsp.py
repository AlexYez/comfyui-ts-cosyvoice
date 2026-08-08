"""
Signal-level DSP for long-form audio: silence detection, seam joining, levelling.

These are pure functions over torch tensors with no ComfyUI or CosyVoice
dependency, which is what makes them worth having in one place: every one of them
is a numerical claim that can be unit-tested against an analytic expectation
(`tests/test_audio_dsp.py`), and getting them wrong is audible rather than
crashy.

Why each exists:

- ``find_silence_intervals`` replaces ``librosa.effects.split``. librosa pulls in
  numba, which refuses to load against the installed NumPy, so the silence-aware
  split silently degraded to fixed-size cuts landing mid-word. Framed RMS in
  torch has no such dependency.
- ``crossfade_join`` replaces a "crossfade" that faded one chunk out to zero,
  faded the next in from zero and concatenated them — measurably reaching 0.0
  amplitude at every seam, i.e. a 40 ms dropout every chunk. A real overlap-add
  keeps the signal continuous.
- ``sola_join`` exists because chunk outputs cannot simply be overlapped: the
  model's output length is not an exact multiple of its input length (speed,
  token timing), so the overlap has to be *found* rather than assumed. This is
  the same synchronous overlap-add approach long-form and streaming voice
  conversion implementations use.
- ``normalise_rms`` is deliberately RMS-based and named so. It is not an
  ITU-R BS.1770 / EBU R128 loudness normaliser and must not be described as one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported lazily by callers; annotations only
    import torch


def _as_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Reduce ``[C, T]`` (or ``[T]``) to a 1-D mono view on CPU float32."""
    if waveform.ndim == 1:
        mono = waveform
    elif waveform.shape[0] > 1:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform[0]
    return mono.detach().cpu().float()


def find_silence_intervals(
    waveform: torch.Tensor,
    sample_rate: int,
    min_silence_seconds: float = 0.5,
    top_db: float = 35.0,
    frame_seconds: float = 0.05,
    hop_seconds: float = 0.01,
) -> list[tuple[int, int]]:
    """
    Return ``[(start, end), ...]`` sample ranges that are silent.

    Silence is defined the way librosa's ``top_db`` defines it: a frame is silent
    when its RMS is more than ``top_db`` below the loudest frame. That makes the
    threshold relative to the material, so a quiet recording is not treated as
    entirely silent.

    Args:
        waveform: ``[C, T]`` or ``[T]``; channels are averaged.
        sample_rate: samples per second of ``waveform``.
        min_silence_seconds: intervals shorter than this are not reported.
        top_db: how far below the peak frame counts as silence.
        frame_seconds: analysis window length.
        hop_seconds: analysis window step.

    Returns:
        Non-overlapping, ascending sample ranges. Empty when nothing qualifies.
        Entirely-silent input returns one interval covering the whole signal.
    """
    import torch

    mono = _as_mono(waveform)
    total_samples = int(mono.shape[-1])
    if total_samples == 0:
        return []

    min_silence_samples = int(min_silence_seconds * sample_rate)

    peak = float(mono.abs().max())
    if peak <= 1e-6:
        # Digital silence: the relative threshold below is meaningless here.
        return [(0, total_samples)] if total_samples >= min_silence_samples else []

    frame_length = max(2, int(sample_rate * frame_seconds))
    hop_length = max(1, int(sample_rate * hop_seconds))

    if total_samples < frame_length:
        return []

    # Centre the frames so the reported edges line up with the signal the way
    # librosa's centred framing does.
    padding = frame_length // 2
    padded = torch.nn.functional.pad(
        mono.view(1, 1, -1), (padding, padding), mode="reflect"
    )

    # Frame energy via average pooling over the squared signal, NOT via
    # `unfold(...).pow(2)`. `unfold` is a strided view, but squaring it
    # materialises frames x frame_length floats — with a 50 ms frame and a 10 ms
    # hop that is 5x the signal: measured at 288 MB for a 10-minute source and
    # 864 MB for 30 minutes, on top of the 1.5 GB model. Pooling allocates one
    # copy of the signal instead, and produces frame-for-frame identical values
    # (verified to float32 noise, ~1e-6).
    mean_square = torch.nn.functional.avg_pool1d(
        padded.pow(2), kernel_size=frame_length, stride=hop_length
    )[0, 0]
    frame_rms = mean_square.clamp(min=1e-20).sqrt()

    reference = float(frame_rms.max())
    if reference <= 0:
        return [(0, total_samples)] if total_samples >= min_silence_samples else []

    frame_db = 20.0 * torch.log10(frame_rms / reference)
    silent = (frame_db < -abs(top_db)).tolist()

    intervals: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = index
        elif not is_silent and run_start is not None:
            intervals.append((run_start, index))
            run_start = None
    if run_start is not None:
        intervals.append((run_start, len(silent)))

    sample_intervals: list[tuple[int, int]] = []
    for first_frame, last_frame in intervals:
        start = min(first_frame * hop_length, total_samples)
        end = min(last_frame * hop_length, total_samples)
        if end - start >= min_silence_samples:
            sample_intervals.append((start, end))

    return sample_intervals


def crossfade_join(
    first: torch.Tensor,
    second: torch.Tensor,
    fade_samples: int,
) -> torch.Tensor:
    """
    Overlap-add two waveforms over ``fade_samples``.

    The overlap is a linear crossfade, which is the correct choice here: SOLA has
    already aligned the two chunks so the overlapping content is *correlated*, and
    for correlated signals a linear fade pair sums to unity gain. (An equal-power
    fade would bulge to +3 dB in the middle for identical content.)

    Result length is ``len(first) + len(second) - fade_samples``: the overlap is
    consumed, not appended. With ``fade_samples <= 0`` this is a plain
    concatenation.

    Args:
        first: ``[C, T1]`` — its tail is faded out.
        second: ``[C, T2]`` — its head is faded in.
        fade_samples: overlap length; clamped to what both sides can supply.
    """
    import torch

    if first.numel() == 0:
        return second.clone()
    if second.numel() == 0:
        return first.clone()

    fade = min(int(fade_samples), first.shape[-1], second.shape[-1])
    if fade <= 0:
        return torch.cat([first, second], dim=-1)

    fade_out = torch.linspace(1.0, 0.0, fade, dtype=first.dtype, device=first.device)
    fade_in = torch.linspace(0.0, 1.0, fade, dtype=second.dtype, device=second.device)

    head = first[..., : first.shape[-1] - fade]
    overlap = first[..., first.shape[-1] - fade :] * fade_out + second[..., :fade] * fade_in
    tail = second[..., fade:]
    return torch.cat([head, overlap, tail], dim=-1)


def find_sola_offset(
    accumulated: torch.Tensor,
    incoming: torch.Tensor,
    anchor: int,
    search_samples: int,
    correlation_samples: int,
) -> int:
    """
    Find where ``incoming`` best lines up inside ``accumulated``, near ``anchor``.

    Uses normalised cross-correlation, evaluated for every candidate offset at
    once via a 1-D convolution rather than a Python loop — the search window is
    tens of thousands of samples wide.

    Args:
        accumulated: ``[C, T]`` signal built so far.
        incoming: ``[C, T]`` next chunk, whose head overlaps ``accumulated``.
        anchor: expected splice position in ``accumulated``.
        search_samples: how far either side of ``anchor`` to look.
        correlation_samples: length of ``incoming``'s head used for matching.

    Returns:
        The absolute splice position in ``accumulated``. Falls back to a clamped
        ``anchor`` when there is not enough signal to correlate.
    """
    import torch

    total = accumulated.shape[-1]
    correlation_samples = int(min(correlation_samples, incoming.shape[-1]))
    if correlation_samples <= 0:
        return max(0, min(anchor, total))

    lowest = max(0, anchor - int(search_samples))
    highest = min(total - correlation_samples, anchor + int(search_samples))
    if highest <= lowest:
        return max(0, min(anchor, max(0, total - correlation_samples)))

    window = _as_mono(accumulated[..., lowest : highest + correlation_samples]).unsqueeze(0).unsqueeze(0)
    template = _as_mono(incoming[..., :correlation_samples]).unsqueeze(0).unsqueeze(0)

    numerator = torch.nn.functional.conv1d(window, template)
    energy = torch.nn.functional.conv1d(
        window.pow(2), torch.ones_like(template)
    ).clamp(min=1e-12).sqrt()
    scores = (numerator / energy)[0, 0]

    if scores.numel() == 0:
        return max(0, min(anchor, total - correlation_samples))
    return lowest + int(torch.argmax(scores))


def sola_join(
    accumulated: torch.Tensor,
    incoming: torch.Tensor,
    expected_overlap_samples: int,
    fade_samples: int,
    search_samples: int | None = None,
    correlation_samples: int | None = None,
) -> torch.Tensor:
    """
    Append ``incoming`` to ``accumulated``, aligning it on their shared overlap.

    Chunk N+1 was converted from source audio that overlapped chunk N, so its
    head repeats the end of ``accumulated`` — but not at a predictable sample
    offset, because the model's output length is not an exact function of its
    input length. The offset is therefore measured, then the two are overlap-added
    over ``fade_samples``.

    ``search_samples`` and ``correlation_samples`` should be passed explicitly and
    sized in *absolute* terms. Deriving the search from the overlap length (an
    earlier default here) gave a ±500 ms search for a 1 s overlap, and for
    quasi-periodic speech a wide search does not find a better alignment — every
    pitch period correlates almost as well, so it only admits more wrong answers.
    Deriving the correlation window from the crossfade made it ~20 ms, under two
    periods of a low voice, which is close to degenerate.

    Args:
        accumulated: ``[C, T]`` result so far.
        incoming: ``[C, T]`` next converted chunk.
        expected_overlap_samples: nominal overlap, used as the search anchor.
        fade_samples: crossfade length at the splice.
        search_samples: search radius. Defaults to the expected overlap, which is
            almost certainly too wide — pass a value bounded by the plausible
            length drift instead.
        correlation_samples: length of the matching window. Defaults to the search
            radius, so it never collapses to the crossfade length.

    Returns:
        The joined waveform. Never shorter than ``accumulated``.
    """
    if accumulated.numel() == 0:
        return incoming.clone()
    if incoming.numel() == 0:
        return accumulated.clone()

    expected_overlap = max(0, int(expected_overlap_samples))
    fade = max(0, int(fade_samples))
    if expected_overlap <= 0:
        return crossfade_join(accumulated, incoming, fade)

    if search_samples is None:
        search_samples = expected_overlap
    search = max(1, int(search_samples))

    if correlation_samples is None:
        correlation_samples = search
    correlation = max(1, min(int(correlation_samples), expected_overlap))

    anchor = max(0, accumulated.shape[-1] - expected_overlap)
    offset = find_sola_offset(accumulated, incoming, anchor, search, correlation)

    # Keep everything up to the splice plus one fade window to blend across.
    keep = min(accumulated.shape[-1], offset + fade)
    return crossfade_join(accumulated[..., :keep], incoming, fade)


def rms_dbfs(waveform: torch.Tensor) -> float:
    """Return the RMS level of ``waveform`` in dBFS (``-inf`` for silence)."""
    mono = _as_mono(waveform)
    if mono.numel() == 0:
        return -math.inf
    mean_square = float(mono.pow(2).mean())
    if mean_square <= 0:
        return -math.inf
    return 10.0 * math.log10(mean_square)


def normalise_rms(
    waveform: torch.Tensor,
    target_dbfs: float,
    peak_ceiling_dbfs: float = -1.0,
) -> torch.Tensor:
    """
    Scale ``waveform`` to a target RMS level, without letting peaks clip.

    This is plain RMS normalisation plus a peak ceiling — deliberately *not*
    ITU-R BS.1770 loudness, and it should not be labelled as loudness anywhere in
    the UI or docs. It exists so that two runs with different reference clips come
    out at comparable levels.

    When honouring the target would push the peak above ``peak_ceiling_dbfs``, the
    gain is reduced so the ceiling wins; the result is then quieter than
    requested, which is the correct trade.

    Args:
        waveform: ``[C, T]`` or ``[T]``.
        target_dbfs: desired RMS level, e.g. ``-20.0``.
        peak_ceiling_dbfs: highest permitted sample peak, e.g. ``-1.0``.

    Returns:
        A new tensor. Digital silence is returned unchanged.
    """
    current = rms_dbfs(waveform)
    if current == -math.inf:
        return waveform.clone()

    gain = 10.0 ** ((target_dbfs - current) / 20.0)

    peak = float(waveform.abs().max())
    if peak > 0:
        ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
        gain = min(gain, ceiling / peak)

    return waveform * gain
