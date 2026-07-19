"""
Audio Utilities for TS CosyVoice3
Handles audio format conversions and processing

Heavy runtime dependencies (torch, torchaudio, soundfile) are imported inside the
functions that need them: this module is pulled in by every synthesis node at pack
import time, and a module-level import would make one missing optional dependency
prevent ComfyUI from registering any node of the pack.
"""

import os
import tempfile
from typing import Any, Dict, Optional, Tuple

try:
    from .ts_logging import get_logger
except (ImportError, ValueError):
    from ts_logging import get_logger


LOGGER = get_logger("TS CosyVoice3 Audio Utils")

REFERENCE_AUDIO_SAMPLE_RATE = 24000
REFERENCE_AUDIO_MAX_SECONDS = 30.0


def _require_soundfile():
    """Import soundfile lazily with an actionable error message."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "[TS CosyVoice3 Audio Utils] soundfile is not installed, so audio cannot be "
            "written for CosyVoice. Install requirements.txt and restart ComfyUI."
        ) from exc
    return sf


def _drop_batch_dim(waveform: Any, context: str = "") -> Any:
    """
    Reduce a ``[B, C, T]`` AUDIO waveform to ``[C, T]``.

    CosyVoice operates on a single clip. When ``B == 1`` this is a plain squeeze;
    when ``B > 1`` we take the first item and log it, rather than leaving a 3-D
    array that soundfile would later reject with a cryptic error. Non-3-D input is
    returned unchanged.
    """
    if waveform.ndim != 3:
        return waveform
    if waveform.shape[0] > 1:
        LOGGER.debug(
            "[TS CosyVoice3 Audio Utils] %s received a batch of %d clips; using the first one.",
            context or "audio",
            int(waveform.shape[0]),
        )
        return waveform[0]
    return waveform.squeeze(0)


def _write_temp_wav(audio_np: Any, sample_rate: int, suffix: str = ".wav") -> str:
    """
    Write a numpy audio array to a fresh temp WAV, cleaning up on write failure.

    ``NamedTemporaryFile(delete=False)`` leaves an empty file on disk the moment
    it is created; if ``sf.write`` then fails (full disk, bad array) the caller
    never receives a path to clean up, so the empty file would leak. We remove it
    here and re-raise.
    """
    sf = _require_soundfile()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()

    try:
        sf.write(temp_path, audio_np, sample_rate)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

    return temp_path


def comfyui_audio_to_tensor(audio: Dict[str, Any]) -> Tuple[Any, int]:
    """
    Convert ComfyUI AUDIO format to tensor and sample rate

    Args:
        audio: ComfyUI audio dict {"waveform": tensor, "sample_rate": int}

    Returns:
        Tuple of (waveform_tensor, sample_rate)
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    return waveform, sample_rate


def tensor_to_comfyui_audio(waveform: Any, sample_rate: int) -> Dict[str, Any]:
    """
    Convert tensor to ComfyUI AUDIO format

    Args:
        waveform: Audio tensor
        sample_rate: Sample rate in Hz

    Returns:
        ComfyUI audio dict
    """
    import torch

    # Ensure waveform is on CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Ensure proper shape [batch, channels, samples]
    if waveform.ndim == 1:
        # Mono, no batch -> [1, 1, samples]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        # Either [channels, samples] or [batch, samples]
        # Assume [channels, samples] and add batch dim
        waveform = waveform.unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }


def save_audio_to_tempfile(waveform: Any, sample_rate: int, suffix: str = ".wav") -> str:
    """
    Save audio tensor to a temporary file

    Args:
        waveform: Audio tensor [channels, samples] or [batch, channels, samples]
        sample_rate: Sample rate in Hz
        suffix: File suffix

    Returns:
        Path to temporary file
    """
    import torch

    # Ensure waveform is on CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Remove batch dimension if present
    waveform = _drop_batch_dim(waveform, "save_audio_to_tempfile")

    # Save audio using soundfile directly (avoids torchaudio's torchcodec requirement)
    # soundfile expects shape (samples, channels), so transpose from (channels, samples)
    audio_np = waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
    return _write_temp_wav(audio_np, sample_rate, suffix)


def resample_audio(waveform: Any, orig_sample_rate: int, target_sample_rate: int) -> Any:
    """
    Resample audio tensor to target sample rate

    Args:
        waveform: Audio tensor
        orig_sample_rate: Original sample rate
        target_sample_rate: Target sample rate

    Returns:
        Resampled audio tensor
    """
    if orig_sample_rate == target_sample_rate:
        return waveform

    import torchaudio

    resampler = torchaudio.transforms.Resample(orig_sample_rate, target_sample_rate)
    return resampler(waveform)


def ensure_mono(waveform: Any) -> Any:
    """
    Convert audio to mono by averaging channels

    Args:
        waveform: Audio tensor [..., channels, samples]

    Returns:
        Mono audio tensor [..., 1, samples]
    """
    if waveform.shape[-2] == 1:
        return waveform

    # Average across channels
    return waveform.mean(dim=-2, keepdim=True)


def prepare_reference_audio_for_cosyvoice(
    audio: Dict[str, Any],
    target_sample_rate: int = REFERENCE_AUDIO_SAMPLE_RATE,
    max_duration_seconds: float = REFERENCE_AUDIO_MAX_SECONDS,
    mono: bool = True,
) -> Dict[str, Any]:
    """
    Prepare reference audio for CosyVoice reference inputs.

    The reference clip is trimmed to the supported maximum duration and
    normalized into mono / target sample rate so every node feeds CosyVoice
    the same stable format.

    Args:
        audio: ComfyUI audio dict
        target_sample_rate: Target sample rate for the processed reference audio
        max_duration_seconds: Maximum allowed duration before trimming
        mono: Convert to mono

    Returns:
        New ComfyUI audio dict with trimmed/resampled waveform
    """
    waveform, sample_rate = comfyui_audio_to_tensor(audio)

    if waveform.ndim == 3:
        waveform = _drop_batch_dim(waveform, "prepare_reference_audio_for_cosyvoice")
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    max_samples = max(1, int(max_duration_seconds * sample_rate))
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    if mono and waveform.shape[0] > 1:
        waveform = ensure_mono(waveform)

    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return {
        "waveform": waveform.contiguous(),
        "sample_rate": sample_rate,
    }


def prepare_audio_for_cosyvoice(
    audio: Dict[str, Any],
    target_sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[Any, int, Optional[str]]:
    """
    Prepare ComfyUI audio for CosyVoice inference

    Args:
        audio: ComfyUI audio dict
        target_sample_rate: Target sample rate for CosyVoice
        mono: Convert to mono

    Returns:
        Tuple of (waveform, sample_rate, temp_file_path)
    """
    waveform, sample_rate = comfyui_audio_to_tensor(audio)

    # Remove batch dimension if present
    waveform = _drop_batch_dim(waveform, "prepare_audio_for_cosyvoice")

    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = ensure_mono(waveform)

    # Resample if needed
    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    # Save to temp file (CosyVoice may expect file paths)
    temp_path = save_audio_to_tempfile(waveform, sample_rate)

    return waveform, sample_rate, temp_path


def save_raw_audio_to_tempfile(audio: Dict[str, Any]) -> str:
    """
    Save ComfyUI audio to temp file WITHOUT any processing.

    CosyVoice's load_wav() handles mono conversion and resampling internally,
    so we should NOT preprocess the audio.

    Args:
        audio: ComfyUI audio dict {"waveform": tensor, "sample_rate": int}

    Returns:
        Path to temporary file
    """
    import torch

    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Remove batch dim if present
    waveform = _drop_batch_dim(waveform, "save_raw_audio_to_tempfile")

    # Ensure CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # soundfile expects (samples, channels) format
    audio_np = waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
    return _write_temp_wav(audio_np, sample_rate)


def cleanup_temp_file(temp_path: Optional[str]) -> None:
    """
    Clean up temporary audio file

    Args:
        temp_path: Path to temporary file
    """
    if not temp_path or not os.path.exists(temp_path):
        return

    try:
        os.unlink(temp_path)
    except OSError as e:
        # A leaked temp file is not worth failing a finished generation over;
        # the OS cleans the temp dir eventually.
        LOGGER.debug("[TS CosyVoice3 Audio Utils] Could not remove temp file %s: %s", temp_path, e)
