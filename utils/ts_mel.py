"""
Mel filterbanks and whisper's log-mel front-end, without numba.

Two places on the CosyVoice inference path reach for a numba-carrying package to
compute a mel spectrogram, and neither needs anything numba provides:

- ``cosyvoice/cli/frontend.py`` calls ``whisper.log_mel_spectrogram(speech,
  n_mels=128)`` to feed the speech tokenizer. Importing ``whisper`` pulls
  ``whisper.timing``, which imports numba.
- ``matcha/utils/audio.py`` does ``from librosa.filters import mel`` at module
  level, and the shipped ``cosyvoice3.yaml`` reaches that module through
  ``feat_extractor: !name:matcha.utils.audio.mel_spectrogram``. Importing
  ``librosa.filters`` imports numba too (``import librosa`` alone does not --
  librosa 0.10 loads submodules lazily).

numba refuses to load against NumPy newer than 2.1, and current ComfyUI ships
NumPy 2.5. That made the whole pack unloadable on an up-to-date ComfyUI, for the
sake of two mel matrices. Both are ordinary arithmetic:

``mel_filterbank`` is librosa's ``filters.mel`` under its default settings
(slaney mel scale, slaney normalisation, float32). It is not an approximation --
it agrees with librosa's own implementation to within one float32 ULP at every
parameter set the pack uses, which ``tests/test_mel.py`` pins by lifting
librosa's reference code out of its source. The residual is float32 rounding and
is the same magnitude by which whisper's own shipped filterbank asset differs
from a fresh librosa computation.

``log_mel_spectrogram`` is a faithful port of ``whisper.audio``'s function of the
same name, kept deliberately literal so it can be diffed against upstream.

Nothing here imports whisper or librosa. When whisper *is* installed, its
precomputed filterbank asset is read straight off disk with NumPy so the
tokenizer front-end stays bit-identical to upstream.
"""

from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # heavy deps are imported inside the functions that need them
    import numpy as np
    import torch

# whisper.audio's constants, restated rather than imported.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160

# The only two filterbanks whisper ships, and so the only two it supports.
WHISPER_MEL_COUNTS = (80, 128)

_MEL_BREAK_HZ = 1000.0
_MEL_LINEAR_STEP_HZ = 200.0 / 3.0
_MEL_LOG_RATIO = 6.4
_MEL_LOG_DIVISOR = 27.0


def _hz_to_mel_slaney(frequencies: Any) -> Any:
    """Slaney (librosa default, non-HTK) Hz to mel. Linear below 1 kHz, log above."""
    import numpy as np

    frequencies = np.asarray(frequencies, dtype=np.float64)
    linear = frequencies / _MEL_LINEAR_STEP_HZ
    break_mel = _MEL_BREAK_HZ / _MEL_LINEAR_STEP_HZ
    logstep = np.log(_MEL_LOG_RATIO) / _MEL_LOG_DIVISOR
    # 0 Hz would warn on the log branch even though np.where discards it.
    with np.errstate(divide="ignore"):
        logarithmic = break_mel + np.log(frequencies / _MEL_BREAK_HZ) / logstep
    return np.where(frequencies >= _MEL_BREAK_HZ, logarithmic, linear)


def _mel_to_hz_slaney(mels: Any) -> Any:
    """Inverse of :func:`_hz_to_mel_slaney`."""
    import numpy as np

    mels = np.asarray(mels, dtype=np.float64)
    linear = _MEL_LINEAR_STEP_HZ * mels
    break_mel = _MEL_BREAK_HZ / _MEL_LINEAR_STEP_HZ
    logstep = np.log(_MEL_LOG_RATIO) / _MEL_LOG_DIVISOR
    logarithmic = _MEL_BREAK_HZ * np.exp(logstep * (mels - break_mel))
    return np.where(mels >= break_mel, logarithmic, linear)


def mel_filterbank(
    sr: float,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Build a ``(n_mels, 1 + n_fft // 2)`` mel projection matrix.

    Equivalent to ``librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
    fmin=fmin, fmax=fmax)`` with librosa's defaults: the slaney mel scale, slaney
    normalisation (each filter integrates to a constant, so the result is
    independent of filter width) and a float32 result.

    Args:
        sr: Sample rate of the signal the STFT was taken from.
        n_fft: FFT size used for the STFT.
        n_mels: Number of mel bands.
        fmin: Lowest frequency to cover, in Hz.
        fmax: Highest frequency to cover, in Hz. ``None`` means ``sr / 2``.

    Returns:
        The projection matrix as float32.
    """
    import numpy as np

    if fmax is None:
        fmax = sr / 2.0

    n_freqs = int(1 + n_fft // 2)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)

    # n_mels + 2 band edges: each filter spans edges [i, i+2] and peaks at i+1.
    edges_mel = np.linspace(
        _hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2
    )
    edges_hz = _mel_to_hz_slaney(edges_mel)

    widths = np.diff(edges_hz)
    ramps = np.subtract.outer(edges_hz, fft_freqs)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for band in range(n_mels):
        rising = -ramps[band] / widths[band]
        falling = ramps[band + 2] / widths[band + 1]
        weights[band] = np.maximum(0.0, np.minimum(rising, falling))

    # Slaney normalisation, applied to the triangles rather than to their peaks.
    enorm = 2.0 / (edges_hz[2 : n_mels + 2] - edges_hz[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights.astype(np.float32)


def librosa_compatible_mel(
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: str | None = "slaney",
    dtype: Any = None,
) -> np.ndarray:
    """
    Drop-in stand-in for ``librosa.filters.mel``, matching its keyword-only signature.

    Only librosa's default scale and normalisation are implemented, because they
    are the only ones the vendored code asks for. Anything else raises rather
    than quietly returning a different filterbank than the caller expects.

    Raises:
        NotImplementedError: If ``htk`` or a non-slaney ``norm`` is requested.
    """
    import numpy as np

    if htk:
        raise NotImplementedError(
            "[TS CosyVoice3] the bundled mel filterbank implements the slaney "
            "mel scale only (htk=False); install librosa for htk=True."
        )
    if norm != "slaney":
        raise NotImplementedError(
            "[TS CosyVoice3] the bundled mel filterbank implements slaney "
            f"normalisation only; norm={norm!r} needs librosa."
        )

    weights = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    if dtype is not None and np.dtype(dtype) != weights.dtype:
        weights = weights.astype(dtype)
    return weights


def _whisper_asset_path() -> str | None:
    """
    Locate whisper's ``assets/mel_filters.npz`` on disk without importing whisper.

    Deliberately uses ``PathFinder`` rather than ``importlib.util.find_spec``.
    ``find_spec`` consults ``sys.modules`` first, and by the time anything calls
    this the pack has usually registered its own stand-in for ``whisper`` there
    (see :mod:`utils.ts_runtime_shims`) -- so ``find_spec`` would report the stub,
    which has no files, and the real asset sitting right there on disk would be
    missed. ``PathFinder`` searches the filesystem, so the answer does not depend
    on whether a shim happens to be installed.

    Neither approach executes the package, which matters: ``import whisper`` is
    exactly what fails in the environments this module exists for.
    """
    import os
    from importlib.machinery import PathFinder

    try:
        spec = PathFinder.find_spec("whisper", sys.path)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    for location in spec.submodule_search_locations:
        candidate = os.path.join(location, "assets", "mel_filters.npz")
        if os.path.isfile(candidate):
            return candidate
    return None


@functools.lru_cache(maxsize=len(WHISPER_MEL_COUNTS))
def _whisper_filterbank(n_mels: int) -> np.ndarray:
    """
    whisper's own filterbank if its asset is on disk, otherwise ours.

    Preferring the shipped asset keeps the speech tokenizer's input bit-identical
    to upstream whenever whisper is installed at all, importable or not.
    """
    import numpy as np

    path = _whisper_asset_path()
    if path is not None:
        try:
            with np.load(path, allow_pickle=False) as archive:
                return archive[f"mel_{n_mels}"]
        except (OSError, KeyError, ValueError):
            # A truncated or unexpected asset is not worth failing over; the
            # computed bank is equivalent to within float32 rounding.
            pass

    return mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)


def whisper_mel_filters(device: Any, n_mels: int) -> torch.Tensor:
    """
    Mel projection matrix for whisper's front-end, on ``device``.

    Mirrors ``whisper.audio.mel_filters``.

    Raises:
        ValueError: If ``n_mels`` is not one whisper supports.
    """
    import torch

    if n_mels not in WHISPER_MEL_COUNTS:
        raise ValueError(
            f"[TS CosyVoice3] unsupported n_mels={n_mels}; whisper's front-end "
            f"defines {WHISPER_MEL_COUNTS}"
        )
    return torch.from_numpy(_whisper_filterbank(n_mels)).to(device)


def log_mel_spectrogram(
    audio: Any,
    n_mels: int = 80,
    padding: int = 0,
    device: Any = None,
) -> torch.Tensor:
    """
    Log-mel spectrogram matching ``whisper.audio.log_mel_spectrogram``.

    Kept structurally identical to upstream -- same window, same dropped final
    STFT frame, same 8 dB floor and same ``(x + 4) / 4`` scaling -- so that a
    future upstream change is easy to spot in a diff.

    One deliberate deviation: the filterbank is cast to the magnitudes' dtype
    before the matmul. Upstream assumes float32 and raises on a half-precision
    waveform; this costs nothing at float32 and makes fp16 work.

    Args:
        audio: Waveform at 16 kHz as a torch tensor or NumPy array. Unlike
            upstream, a file path is not accepted: decoding one needs ffmpeg,
            and nothing on this pack's inference path passes one.
        n_mels: 80 or 128.
        padding: Zero samples appended on the right.
        device: Optional device to move the waveform to before the STFT.

    Returns:
        ``(n_mels, n_frames)``, or a leading batch dimension if ``audio`` had one.

    Raises:
        TypeError: If ``audio`` is a path rather than an array or tensor.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if isinstance(audio, str):
        raise TypeError(
            "[TS CosyVoice3] log_mel_spectrogram needs decoded samples, not a "
            "file path; load the audio first (upstream whisper shells out to "
            "ffmpeg for this)."
        )
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(np.asarray(audio))

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = whisper_mel_filters(audio.device, n_mels).to(magnitudes.dtype)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
