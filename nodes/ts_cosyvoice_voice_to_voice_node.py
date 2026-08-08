"""
TS CosyVoice3 Voice Conversion Node (V3 schema).
Convert one voice to sound like another (voice-to-voice).

Migrated from V1 to V3 on 2026-05-07.
Public contract preserved:
- node_id: TS_CosyVoice3_VoiceConversion
- inputs (required): model, source_audio, target_audio, speed, pitch_shift_semitones
- inputs (optional): seed, diffusion_steps, guidance_strength, normalize_output,
  target_rms_dbfs — all appended after `seed`, so saved graphs keep mapping
  their shorter widgets_values arrays correctly.
- outputs: (AUDIO,) named "audio"

Notes:
- This node is the most complex of the synthesis nodes: overlapping long-source
  chunking, silence-aware split points, formant-preserving pitch shift, and a
  SOLA-aligned overlap-add join.
"""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from comfy_api.v0_0_2 import IO

if TYPE_CHECKING:  # annotations only — torch is imported lazily where it is used
    import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_audio_dsp import (
        crossfade_join,
        find_silence_intervals,
        normalise_rms,
        rms_dbfs,
        sola_join,
    )
    from ..utils.ts_audio_utils import (
        cleanup_temp_file,
        ensure_mono,
        prepare_audio_for_cosyvoice,
        resample_audio,
        tensor_to_comfyui_audio,
    )
    from ..utils.ts_cosyvoice_adapter import flow_inference_overrides
    from ..utils.ts_fingerprint import seed_fingerprint
    from ..utils.ts_interrupt import raise_if_interrupted
    from ..utils.ts_logging import get_logger, log_banner, log_exception
    from ..utils.ts_node_utils import set_seed
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from nodes._v3_types import CosyVoiceModel
    from utils.ts_audio_dsp import (
        crossfade_join,
        find_silence_intervals,
        normalise_rms,
        rms_dbfs,
        sola_join,
    )
    from utils.ts_audio_utils import (
        cleanup_temp_file,
        ensure_mono,
        prepare_audio_for_cosyvoice,
        resample_audio,
        tensor_to_comfyui_audio,
    )
    from utils.ts_cosyvoice_adapter import flow_inference_overrides
    from utils.ts_fingerprint import seed_fingerprint
    from utils.ts_interrupt import raise_if_interrupted
    from utils.ts_logging import get_logger, log_banner, log_exception
    from utils.ts_node_utils import set_seed

import comfy.utils

LOGGER = get_logger("TS CosyVoice Voice To Voice")


MAX_SOURCE_CHUNK_DURATION_SECONDS = 24.0
MAX_TARGET_REFERENCE_SECONDS = 30.0
MIN_SILENCE_DURATION_SECONDS = 0.5
MIN_CHUNK_DURATION_SECONDS = 10.0
CHUNK_CROSSFADE_SECONDS = 0.02
SOURCE_VC_SAMPLE_RATE = 16000

# How far below the loudest frame counts as silence when choosing a cut point.
SILENCE_TOP_DB = 35.0

# Each chunk after the first re-reads this much of its predecessor. The model
# converts every chunk in isolation, so without shared context timbre and prosody
# restart at each boundary; the overlap also gives the join something to align on.
# One second is enough for alignment while costing ~4% extra inference at 24 s
# chunks.
SOURCE_CHUNK_OVERLAP_SECONDS = 1.0

# The model's output length is not an exact function of its input length, so the
# overlap is searched for rather than assumed. Both of these are absolute
# durations, deliberately NOT a fraction of the overlap — the previous version
# scaled the search with the overlap, giving +-500 ms of search for a 1 s overlap.
#
# Why these numbers. The output length quantises to one speech token: the shipped
# cosyvoice3.yaml has hop_size 480 at 24 kHz (a 20 ms mel frame) and
# token_mel_ratio 2, so a token is 40 ms. A search of +-80 ms therefore covers a
# drift of +-2 tokens, which is all the grid can produce.
#
# The search must stay small, because for quasi-periodic speech every pitch period
# correlates almost as well as the right one — a wider search does not find a
# better answer, it admits more wrong ones. Measured worst-case misalignment on a
# sustained vowel with two independent generations: +-500 ms search with a 20 ms
# window gave 554 ms of error; the values below gave 154 ms, and 71 ms when the
# only error source was the ambiguity itself.
#
# The correlation window is longer than the crossfade on purpose: 80 ms spans
# ~7 periods at 85 Hz, where 20 ms spans fewer than 2 and is close to degenerate.
SOLA_SEARCH_SECONDS = 0.080
SOLA_CORRELATION_SECONDS = 0.080

# Highest permitted sample peak after optional level normalisation.
OUTPUT_PEAK_CEILING_DBFS = -1.0

# Pitch shifting happens before the source is reduced for the tokenizer, so it
# runs at a rate that leaves the phase vocoder something to work with.
PITCH_SHIFT_WORK_SAMPLE_RATE = 24000


def _extract_waveform(audio: Dict[str, Any]) -> torch.Tensor:
    """Return waveform as [channels, samples]."""
    waveform = audio["waveform"]

    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    return waveform


def _clone_audio_with_waveform(audio: Dict[str, Any], waveform: torch.Tensor) -> Dict[str, Any]:
    """Clone ComfyUI audio dict while replacing waveform."""
    return {
        "waveform": waveform,
        "sample_rate": audio["sample_rate"],
    }


def _trim_audio_to_duration(audio: Dict[str, Any], max_duration_seconds: float) -> Dict[str, Any]:
    """Trim audio to a maximum duration without changing sample rate."""
    waveform = _extract_waveform(audio)
    sample_rate = audio["sample_rate"]
    max_samples = max(1, int(max_duration_seconds * sample_rate))

    if waveform.shape[-1] <= max_samples:
        return _clone_audio_with_waveform(audio, waveform)

    return _clone_audio_with_waveform(audio, waveform[..., :max_samples])


def _get_pitch_shift_fft_size(sample_rate: int, num_samples: int) -> int:
    """Pick a stable FFT size for pitch shifting."""
    if num_samples <= 0:
        return 256

    preferred_fft = 2048 if sample_rate >= 24000 else 1024
    max_fft = max(256, min(preferred_fft, num_samples))
    return max(256, 2 ** math.floor(math.log2(max_fft)))


def _prepare_source_audio_for_pitch_shift(audio: Dict[str, Any], target_sample_rate: int) -> Dict[str, Any]:
    """Downmix and resample before pitch shift to keep the operation responsive."""
    waveform = _extract_waveform(audio).detach()
    sample_rate = audio["sample_rate"]

    if waveform.shape[0] > 1:
        waveform = ensure_mono(waveform)

    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def _prepare_reference_audio_for_model(audio: Dict[str, Any], target_sample_rate: int) -> Dict[str, Any]:
    """Normalize reference audio to mono and the model acoustic sample rate."""
    waveform = _extract_waveform(audio).detach()
    sample_rate = audio["sample_rate"]

    if waveform.shape[0] > 1:
        waveform = ensure_mono(waveform)

    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def _pitch_shift_with_world(
    working_waveform: torch.Tensor,
    sample_rate: int,
    semitones: float,
) -> torch.Tensor | None:
    """
    Shift pitch by rescaling F0 with the WORLD vocoder, keeping formants intact.

    Preferred over a phase vocoder for speech for two reasons. It moves only the
    fundamental, leaving the spectral envelope — so the result does not acquire the
    "chipmunk" formant shift a resampling-based shifter gives, which matters at the
    ±12 semitones this input allows. And it avoids the phasiness a short-window
    phase vocoder introduces.

    ``pyworld`` is already a declared dependency (the vendored training pipeline
    uses it) but was previously unused at inference time.

    Returns:
        ``[C, T]`` float32 with the same length as the input, or None when pyworld
        is unavailable or fails — the caller then falls back to the phase vocoder.
    """
    import numpy as np
    import torch

    try:
        import pyworld
    except ImportError:
        LOGGER.info(
            "[TS CosyVoice3 VC] pyworld not installed; using the phase-vocoder pitch shift"
        )
        return None

    ratio = float(2.0 ** (semitones / 12.0))
    frame_period_ms = 5.0
    num_samples = int(working_waveform.shape[-1])

    try:
        shifted_channels = []
        for channel in working_waveform:
            samples = np.ascontiguousarray(channel.numpy().astype(np.float64))
            f0, time_axis = pyworld.dio(samples, sample_rate, frame_period=frame_period_ms)
            f0 = pyworld.stonemask(samples, f0, time_axis, sample_rate)
            spectrogram = pyworld.cheaptrick(samples, f0, time_axis, sample_rate)
            aperiodicity = pyworld.d4c(samples, f0, time_axis, sample_rate)
            resynthesised = pyworld.synthesize(
                f0 * ratio, spectrogram, aperiodicity, sample_rate, frame_period=frame_period_ms
            )
            shifted_channels.append(torch.from_numpy(resynthesised.astype(np.float32)))
        shifted = torch.stack(shifted_channels, dim=0)
    except Exception as exc:
        LOGGER.warning(
            "[TS CosyVoice3 VC] WORLD pitch shift failed (%s); falling back to the "
            "phase vocoder",
            exc,
        )
        return None

    # WORLD's synthesis length follows the frame grid, not the input exactly.
    if shifted.shape[-1] > num_samples:
        shifted = shifted[..., :num_samples]
    elif shifted.shape[-1] < num_samples:
        import torch.nn.functional as functional

        shifted = functional.pad(shifted, (0, num_samples - shifted.shape[-1]))
    return shifted


def _apply_pitch_shift(audio: Dict[str, Any], semitones: float) -> Dict[str, Any]:
    """
    Pitch-shift source audio while preserving duration.

    Tries WORLD F0 rescaling first (formant-preserving, cleaner on speech) and
    falls back to torchaudio's phase vocoder, then to librosa.
    """
    import torch
    import torchaudio.functional as torchaudio_functional

    if abs(semitones) < 1e-6:
        return audio

    waveform = _extract_waveform(audio)
    sample_rate = audio["sample_rate"]
    num_samples = waveform.shape[-1]

    if num_samples < 256:
        return _clone_audio_with_waveform(audio, waveform)

    original_device = waveform.device
    original_dtype = waveform.dtype
    working_waveform = waveform.detach().cpu().to(torch.float32)

    world_result = _pitch_shift_with_world(working_waveform, sample_rate, semitones)
    if world_result is not None:
        shifted = world_result.to(dtype=original_dtype)
        if original_device != torch.device("cpu"):
            shifted = shifted.to(original_device)
        return _clone_audio_with_waveform(audio, shifted)

    n_fft = _get_pitch_shift_fft_size(sample_rate, num_samples)
    hop_length = max(64, n_fft // 4)

    try:
        shifted_waveform = torchaudio_functional.pitch_shift(
            working_waveform,
            sample_rate=sample_rate,
            n_steps=semitones,
            bins_per_octave=12,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
        )
    except Exception as torchaudio_exc:
        LOGGER.warning(
            "[TS CosyVoice3 VC] torchaudio pitch_shift failed (%s); falling back to librosa "
            "(slower, same result)",
            torchaudio_exc,
        )
        # Guarded: this is the last link in the chain, and it is the one most likely
        # to be unavailable. librosa's `effects` module pulls in numba, which refuses
        # to load against a newer NumPy — an unguarded call here surfaced as a raw
        # `ImportError: Numba needs NumPy 2.1 or less` from inside librosa, with
        # nothing pointing at this pack or at a fix.
        try:
            import librosa

            shifted_channels: List[torch.Tensor] = []
            for channel_waveform in working_waveform:
                shifted_channel = librosa.effects.pitch_shift(
                    y=channel_waveform.numpy(),
                    sr=sample_rate,
                    n_steps=semitones,
                    bins_per_octave=12,
                )
                shifted_channels.append(torch.from_numpy(shifted_channel))
            shifted_waveform = torch.stack(shifted_channels, dim=0)
        except Exception as librosa_exc:
            raise RuntimeError(
                "[TS CosyVoice3 VC] Could not pitch-shift the source audio: all three "
                "shifters failed.\n"
                "  1. WORLD (pyworld) — unavailable; install it with: pip install pyworld\n"
                f"  2. torchaudio phase vocoder — {torchaudio_exc}\n"
                f"  3. librosa — {librosa_exc}\n"
                "Set pitch_shift_semitones to 0 to convert without a shift, or install "
                "pyworld to get the better shifter."
            ) from librosa_exc

    if shifted_waveform.shape[-1] > num_samples:
        shifted_waveform = shifted_waveform[..., :num_samples]
    elif shifted_waveform.shape[-1] < num_samples:
        pad_samples = num_samples - shifted_waveform.shape[-1]
        shifted_waveform = torch.nn.functional.pad(shifted_waveform, (0, pad_samples))

    shifted_waveform = shifted_waveform.to(dtype=original_dtype)
    if original_device != torch.device("cpu"):
        shifted_waveform = shifted_waveform.to(original_device)

    return _clone_audio_with_waveform(audio, shifted_waveform)


def _get_silence_intervals(
    waveform: torch.Tensor,
    sample_rate: int,
    min_silence_duration_seconds: float,
) -> List[Tuple[int, int]]:
    """
    Detect silence intervals so a long source is cut on a pause, not mid-word.

    Implemented on framed RMS in torch (``utils.ts_audio_dsp``). This used to call
    ``librosa.effects.split``, whose numba backend refuses to load against the
    installed NumPy — so the whole silence-aware path was dead and every cut
    landed at a fixed offset regardless of content. There is no optional
    dependency left to fail here.
    """
    return find_silence_intervals(
        waveform,
        sample_rate,
        min_silence_seconds=min_silence_duration_seconds,
        top_db=SILENCE_TOP_DB,
    )


def _split_source_audio_into_chunks(audio: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split source audio into overlapping chunks, cutting on pauses where possible.

    Two things happen here:

    - The cut point prefers a silence, so a split does not land mid-word.
    - Each chunk after the first *starts earlier* than the previous one ended, by
      ``SOURCE_CHUNK_OVERLAP_SECONDS``. The model converts each chunk with no
      knowledge of its neighbours, so without shared context the timbre and
      prosody visibly restart at every boundary. The overlap gives the join
      something to align on and blend across; long-form and streaming voice
      conversion implementations do the same.

    Each returned dict carries ``overlap_samples`` — how much of its head repeats
    the previous chunk — which the join needs to know to remove the duplication.
    The first chunk always reports 0.
    """
    waveform = _extract_waveform(audio)
    sample_rate = audio["sample_rate"]
    total_samples = waveform.shape[-1]

    max_chunk_samples = int(MAX_SOURCE_CHUNK_DURATION_SECONDS * sample_rate)
    if total_samples <= max_chunk_samples:
        chunk = _clone_audio_with_waveform(audio, waveform)
        chunk["overlap_samples"] = 0
        return [chunk]

    min_chunk_samples = int(MIN_CHUNK_DURATION_SECONDS * sample_rate)
    overlap_samples = int(SOURCE_CHUNK_OVERLAP_SECONDS * sample_rate)
    silence_intervals = _get_silence_intervals(
        waveform,
        sample_rate,
        min_silence_duration_seconds=MIN_SILENCE_DURATION_SECONDS,
    )

    chunks: List[Dict[str, Any]] = []
    # `start` is where this chunk's *new* content begins; the chunk itself is
    # emitted from `start - overlap` so it re-reads the tail of its predecessor.
    start = 0

    while start < total_samples:
        chunk_start = max(0, start - overlap_samples) if chunks else 0
        actual_overlap = start - chunk_start

        # The limit applies to the emitted chunk, overlap included — the vendored
        # tokenizer asserts on the length of what it is actually handed.
        if total_samples - chunk_start <= max_chunk_samples:
            chunk = _clone_audio_with_waveform(audio, waveform[..., chunk_start:total_samples])
            chunk["overlap_samples"] = actual_overlap
            chunks.append(chunk)
            break

        latest_end = min(chunk_start + max_chunk_samples, total_samples)
        earliest_end = min(latest_end, start + min_chunk_samples)
        split_point: int | None = None
        best_distance: int | None = None
        for silence_start, silence_end in silence_intervals:
            silence_overlap_start = max(silence_start, earliest_end)
            silence_overlap_end = min(silence_end, latest_end)
            if silence_overlap_end - silence_overlap_start < int(
                MIN_SILENCE_DURATION_SECONDS * sample_rate
            ):
                continue

            candidate = (silence_overlap_start + silence_overlap_end) // 2
            if candidate >= earliest_end:
                # Prefer the silence closest to the chunk limit: it yields the
                # fewest chunks while still splitting on a pause.
                distance_to_limit = latest_end - candidate
                if best_distance is None or distance_to_limit < best_distance:
                    best_distance = distance_to_limit
                    split_point = candidate

        if split_point is None or split_point <= start:
            split_point = latest_end

        chunk = _clone_audio_with_waveform(audio, waveform[..., chunk_start:split_point])
        chunk["overlap_samples"] = actual_overlap
        chunks.append(chunk)
        start = split_point

    return chunks


def _collect_inference_output(
    cosyvoice_model: Any,
    source_temp: str,
    target_temp: str,
    speed: float,
) -> torch.Tensor:
    """Run inference_vc and merge streamed model chunks for a single source chunk."""
    import torch

    output = cosyvoice_model.inference_vc(
        source_wav=source_temp,
        prompt_wav=target_temp,
        stream=False,
        speed=speed,
    )

    speech_parts: List[torch.Tensor] = []
    for model_chunk in output:
        raise_if_interrupted()
        speech_parts.append(model_chunk["tts_speech"])

    if not speech_parts:
        raise RuntimeError("Model returned no audio for voice conversion chunk")

    if len(speech_parts) == 1:
        return speech_parts[0]

    return torch.cat(speech_parts, dim=-1)


def _join_converted_chunks(
    waveforms: List[torch.Tensor],
    expected_overlaps: List[int],
    sample_rate: int,
) -> torch.Tensor:
    """
    Join converted chunks, removing the duplicated overlap and blending the seam.

    Replaces a previous implementation that faded each chunk out to zero, faded
    the next in from zero and concatenated them: measured on a constant-amplitude
    signal, the result reached exactly 0.0 at every boundary, i.e. a 40 ms dropout
    once per chunk. An overlap-add crossfade holds the level instead.

    Args:
        waveforms: converted chunks, in order, ``[1, T]`` / ``[C, T]`` / ``[1, C, T]``.
        expected_overlaps: for each chunk, how many output samples of its head are
            expected to repeat the previous chunk. Index 0 is ignored. Zero means
            the chunks do not overlap and only a short blend is applied.
        sample_rate: output sample rate, used to size the blend window.

    Returns:
        ``[C, T]`` on CPU.
    """
    if not waveforms:
        raise ValueError("No waveforms to concatenate")

    prepared: List[torch.Tensor] = []
    for waveform in waveforms:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        prepared.append(waveform.detach().cpu())

    if len(expected_overlaps) != len(prepared):
        # The caller builds both lists in one loop, so a mismatch is a bug here,
        # not bad input. Tolerating it would silently leave the overlap duplicated
        # in the output — a defect that looks like "works, only longer".
        raise ValueError(
            f"expected one overlap per chunk, got {len(expected_overlaps)} "
            f"overlaps for {len(prepared)} chunks"
        )

    fade_samples = max(1, int(sample_rate * CHUNK_CROSSFADE_SECONDS))
    search_samples = max(1, int(sample_rate * SOLA_SEARCH_SECONDS))
    correlation_samples = max(1, int(sample_rate * SOLA_CORRELATION_SECONDS))
    combined = prepared[0]

    for index, next_waveform in enumerate(prepared[1:], start=1):
        expected_overlap = expected_overlaps[index]
        if expected_overlap > 0:
            combined = sola_join(
                combined,
                next_waveform,
                expected_overlap_samples=expected_overlap,
                fade_samples=fade_samples,
                search_samples=search_samples,
                correlation_samples=correlation_samples,
            )
        else:
            combined = crossfade_join(combined, next_waveform, fade_samples)

    return combined


class TS_CosyVoice3_VoiceConversion(IO.ComfyNode):
    """Voice conversion - convert source voice to target voice."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_VoiceConversion",
            display_name="TS CosyVoice Voice To Voice",
            category="TS CosyVoice3/Synthesis",
            description="Convert source audio to sound like the target voice (voice-to-voice).",
            inputs=[
                CosyVoiceModel.Input(
                    "model",
                    tooltip="Загруженная модель CosyVoice из ноды загрузчика.",
                ),
                IO.Audio.Input(
                    "source_audio",
                    tooltip="Исходное аудио, которое нужно преобразовать в другой тембр.",
                ),
                IO.Audio.Input(
                    "target_audio",
                    tooltip="Референс целевого голоса; будет обрезан до 30 секунд и приведен к оптимальному формату.",
                ),
                IO.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Множитель скорости итоговой речи.",
                ),
                IO.Int.Input(
                    "pitch_shift_semitones",
                    default=0,
                    min=-12,
                    max=12,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Сдвиг высоты тона исходного аудио в полутонах перед voice conversion.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=-1,
                    max=2147483647,
                    optional=True,
                    tooltip="Зерно случайности; значение -1 использует случайный seed.",
                ),
                # Appended after `seed`, which is where widgets_values already
                # ended: existing saved graphs carry shorter arrays and keep
                # mapping correctly, since values are applied from index 0.
                IO.Int.Input(
                    "diffusion_steps",
                    default=10,
                    min=4,
                    max=60,
                    step=1,
                    optional=True,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Число шагов флоу-декодера. 10 — значение модели по "
                            "умолчанию (быстро); 25–40 даёт больше деталей и "
                            "пропорционально дольше считает.",
                ),
                IO.Float.Input(
                    "guidance_strength",
                    default=0.7,
                    min=0.0,
                    max=1.5,
                    step=0.05,
                    optional=True,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Сила classifier-free guidance. 0.7 — значение из "
                            "конфигурации модели; выше — ближе к референсу, но "
                            "растёт риск артефактов.",
                ),
                IO.Boolean.Input(
                    "normalize_output",
                    default=False,
                    optional=True,
                    tooltip="Приводит громкость результата к целевому уровню RMS, "
                            "чтобы разные референсы давали сопоставимую громкость.",
                ),
                IO.Float.Input(
                    "target_rms_dbfs",
                    default=-20.0,
                    min=-40.0,
                    max=-6.0,
                    step=0.5,
                    optional=True,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Целевой уровень RMS в dBFS при включённой нормализации. "
                            "Это RMS, а не LUFS: пики ограничиваются -1 dBFS.",
                ),
            ],
            outputs=[
                IO.Audio.Output(display_name="audio"),
            ],
            search_aliases=[],
            is_output_node=False,
            essentials_category="Audio",
        )

    @classmethod
    def fingerprint_inputs(cls, seed: int = 42, **kwargs) -> Any:
        """Make seed=-1 mean what its tooltip says: a new take on every run."""
        return seed_fingerprint(seed)

    @classmethod
    def execute(
        cls,
        model: Dict[str, Any],
        source_audio: Dict[str, Any],
        target_audio: Dict[str, Any],
        speed: float = 1.0,
        pitch_shift_semitones: int = 0,
        seed: int = 42,
        diffusion_steps: int = 10,
        guidance_strength: float = 0.7,
        normalize_output: bool = False,
        target_rms_dbfs: float = -20.0,
    ) -> IO.NodeOutput:
        """Convert source voice to target voice."""
        log_banner(
            LOGGER,
            "[TS CosyVoice3 VC] Converting voice...",
            Speed=f"{speed}x",
        )

        source_waveform = _extract_waveform(source_audio)
        source_sample_rate = source_audio["sample_rate"]
        source_duration = source_waveform.shape[-1] / source_sample_rate

        target_waveform = _extract_waveform(target_audio)
        target_duration = target_waveform.shape[-1] / target_audio["sample_rate"]

        if source_duration <= 0:
            raise ValueError("Source audio is empty")

        if target_duration <= 0:
            raise ValueError("Target audio is empty")

        source_temp = None
        target_temp = None

        try:
            set_seed(seed)

            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate

            LOGGER.info("[TS CosyVoice3 VC] Model sample rate: %s Hz", sample_rate)

            if not hasattr(cosyvoice_model, 'inference_vc'):
                raise RuntimeError("Model does not support voice conversion")

            # Work at up to the model's own rate rather than the tokenizer's
            # 16 kHz: the pitch shift used to run after the downsample, which is
            # the worst place for it. Each chunk is reduced to 16 kHz later, when
            # it is written out for the tokenizer.
            work_sample_rate = min(int(source_sample_rate), PITCH_SHIFT_WORK_SAMPLE_RATE)
            processed_source_audio = _prepare_source_audio_for_pitch_shift(
                source_audio,
                work_sample_rate,
            )
            if pitch_shift_semitones != 0:
                processed_source_audio = _apply_pitch_shift(processed_source_audio, pitch_shift_semitones)

            if abs(pitch_shift_semitones) >= 1e-6:
                LOGGER.info(
                    "[TS CosyVoice3 VC] Applying pitch shift: %+.0f semitones",
                    pitch_shift_semitones,
                )

            trimmed_target_audio = _trim_audio_to_duration(target_audio, MAX_TARGET_REFERENCE_SECONDS)
            trimmed_target_audio = _prepare_reference_audio_for_model(trimmed_target_audio, sample_rate)
            trimmed_target_duration = _extract_waveform(trimmed_target_audio).shape[-1] / trimmed_target_audio["sample_rate"]
            if target_duration > MAX_TARGET_REFERENCE_SECONDS:
                LOGGER.info(
                    "[TS CosyVoice3 VC] Target audio is longer than %.0fs, trimming reference to %.1fs",
                    MAX_TARGET_REFERENCE_SECONDS,
                    trimmed_target_duration,
                )

            source_chunks = _split_source_audio_into_chunks(processed_source_audio)
            total_chunks = len(source_chunks)
            LOGGER.info(
                "[TS CosyVoice3 VC] Source audio: %.1fs -> %s chunk(s) up to %.1fs",
                source_duration,
                total_chunks,
                MAX_SOURCE_CHUNK_DURATION_SECONDS,
            )

            total_steps = total_chunks + 2
            pbar = comfy.utils.ProgressBar(total_steps)
            pbar.update_absolute(0, total_steps)

            LOGGER.info("[TS CosyVoice3 VC] Preparing target audio (%.1fs)...", trimmed_target_duration)
            _, _, target_temp = prepare_audio_for_cosyvoice(trimmed_target_audio, target_sample_rate=sample_rate)
            pbar.update_absolute(1, total_steps)

            LOGGER.info("[TS CosyVoice3 VC] Running voice conversion...")
            # Scoped: the model instance is shared through the cache, so these
            # must not leak into other nodes in the same graph.

            with flow_inference_overrides(
                cosyvoice_model,
                diffusion_steps=diffusion_steps,
                cfg_rate=guidance_strength,
            ):
                converted_chunks: List[Any] = []
                # How much of each converted chunk's head repeats its predecessor, in
                # *output* samples. Derived rather than assumed: the model resamples
                # 16 kHz in to `sample_rate` out and `speed` scales duration.
                expected_overlaps: List[int] = []
                chunk_sample_rate = int(processed_source_audio["sample_rate"])
                for chunk_index, source_chunk in enumerate(source_chunks, start=1):
                    # Cancel between chunks: a long source can be dozens of chunks and
                    # hours of work, so this is the difference between Cancel meaning
                    # something and meaning nothing.
                    raise_if_interrupted()
                    chunk_duration = _extract_waveform(source_chunk).shape[-1] / source_chunk["sample_rate"]
                    source_temp = None
                    LOGGER.info(
                        "[TS CosyVoice3 VC] Preparing source chunk %s/%s (%.1fs)...",
                        chunk_index,
                        total_chunks,
                        chunk_duration,
                    )
                    _, _, source_temp = prepare_audio_for_cosyvoice(source_chunk, target_sample_rate=SOURCE_VC_SAMPLE_RATE)

                    try:
                        converted_chunk = _collect_inference_output(
                            cosyvoice_model,
                            source_temp=source_temp,
                            target_temp=target_temp,
                            speed=speed,
                        )
                    finally:
                        cleanup_temp_file(source_temp)

                    converted_chunks.append(converted_chunk)
                    overlap_in = int(source_chunk.get("overlap_samples", 0))
                    expected_overlaps.append(
                        int(overlap_in * (sample_rate / chunk_sample_rate) / max(speed, 1e-6))
                    )
                    LOGGER.info(
                        "[TS CosyVoice3 VC] Processed source chunk %s/%s",
                        chunk_index,
                        total_chunks,
                    )
                    pbar.update_absolute(chunk_index + 1, total_steps)

            waveform = _join_converted_chunks(converted_chunks, expected_overlaps, sample_rate)

            if normalize_output:
                before_dbfs = rms_dbfs(waveform)
                waveform = normalise_rms(
                    waveform,
                    target_dbfs=float(target_rms_dbfs),
                    peak_ceiling_dbfs=OUTPUT_PEAK_CEILING_DBFS,
                )
                LOGGER.info(
                    "[TS CosyVoice3 VC] Output level: %.1f -> %.1f dBFS RMS (peak ceiling %.1f)",
                    before_dbfs,
                    rms_dbfs(waveform),
                    OUTPUT_PEAK_CEILING_DBFS,
                )

            audio = tensor_to_comfyui_audio(waveform, sample_rate)

            duration = waveform.shape[-1] / sample_rate

            pbar.update_absolute(total_steps, total_steps)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 VC] Voice conversion successful!",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )

            return IO.NodeOutput(audio)

        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 VC] ERROR", exc)
            raise

        finally:
            cleanup_temp_file(source_temp)
            cleanup_temp_file(target_temp)
