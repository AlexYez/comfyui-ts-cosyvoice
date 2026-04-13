"""
TS CosyVoice3 Voice Conversion Node
Convert one voice to sound like another (voice-to-voice)
"""

import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio.functional as torchaudio_functional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_audio_utils import (
        cleanup_temp_file,
        ensure_mono,
        prepare_audio_for_cosyvoice,
        resample_audio,
        tensor_to_comfyui_audio,
    )
    from ..utils.ts_node_utils import set_seed
except (ImportError, ValueError):
    from utils.ts_audio_utils import (
        cleanup_temp_file,
        ensure_mono,
        prepare_audio_for_cosyvoice,
        resample_audio,
        tensor_to_comfyui_audio,
    )
    from utils.ts_node_utils import set_seed

import comfy.utils


MAX_SOURCE_CHUNK_DURATION_SECONDS = 24.0
MAX_TARGET_REFERENCE_SECONDS = 30.0
MIN_SILENCE_DURATION_SECONDS = 0.5
MIN_CHUNK_DURATION_SECONDS = 10.0
CHUNK_CROSSFADE_SECONDS = 0.02
LIBROSA_TOP_DB = 35
PITCH_SHIFT_WORK_SAMPLE_RATE = 16000
SOURCE_VC_SAMPLE_RATE = 16000


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
    return max(256, 2 ** int(math.floor(math.log2(max_fft))))


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


def _apply_pitch_shift(audio: Dict[str, Any], semitones: float) -> Dict[str, Any]:
    """Pitch-shift source audio while preserving duration."""
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
    except Exception:
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
    """Detect silence intervals using librosa's non-silent split."""
    mono_waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    mono_waveform = mono_waveform.detach().cpu().float()

    if mono_waveform.numel() == 0:
        return []

    peak = float(mono_waveform.abs().max().item())
    if peak <= 1e-6:
        return [(0, mono_waveform.shape[-1])]

    import librosa

    frame_length = max(2048, int(sample_rate * 0.05))
    hop_length = max(512, int(sample_rate * 0.01))
    nonsilent_intervals = librosa.effects.split(
        mono_waveform.numpy(),
        top_db=LIBROSA_TOP_DB,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    min_silence_samples = int(min_silence_duration_seconds * sample_rate)
    silence_intervals: List[Tuple[int, int]] = []
    previous_end = 0
    total_samples = mono_waveform.shape[-1]

    for start, end in nonsilent_intervals.tolist():
        if start - previous_end >= min_silence_samples:
            silence_intervals.append((previous_end, start))
        previous_end = end

    if total_samples - previous_end >= min_silence_samples:
        silence_intervals.append((previous_end, total_samples))

    return silence_intervals


def _split_source_audio_into_chunks(audio: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split source audio into <= 29.5 second chunks, preferring silence boundaries."""
    waveform = _extract_waveform(audio)
    sample_rate = audio["sample_rate"]
    total_samples = waveform.shape[-1]

    max_chunk_samples = int(MAX_SOURCE_CHUNK_DURATION_SECONDS * sample_rate)
    if total_samples <= max_chunk_samples:
        return [_clone_audio_with_waveform(audio, waveform)]

    min_chunk_samples = int(MIN_CHUNK_DURATION_SECONDS * sample_rate)
    silence_intervals = _get_silence_intervals(
        waveform,
        sample_rate,
        min_silence_duration_seconds=MIN_SILENCE_DURATION_SECONDS,
    )

    chunks: List[Dict[str, Any]] = []
    start = 0

    while start < total_samples:
        remaining_samples = total_samples - start
        if remaining_samples <= max_chunk_samples:
            chunks.append(_clone_audio_with_waveform(audio, waveform[..., start:total_samples]))
            break

        latest_end = min(start + max_chunk_samples, total_samples)
        earliest_end = min(latest_end, start + min_chunk_samples)
        split_point: Optional[int] = None
        best_distance: Optional[int] = None
        for silence_start, silence_end in silence_intervals:
            overlap_start = max(silence_start, earliest_end)
            overlap_end = min(silence_end, latest_end)
            if overlap_end - overlap_start < int(MIN_SILENCE_DURATION_SECONDS * sample_rate):
                continue

            candidate = (overlap_start + overlap_end) // 2
            if candidate >= earliest_end:
                distance_to_limit = latest_end - candidate
                if best_distance is None or distance_to_limit < best_distance:
                    best_distance = distance_to_limit
                split_point = candidate

        if split_point is None or split_point <= start:
            split_point = latest_end

        chunks.append(_clone_audio_with_waveform(audio, waveform[..., start:split_point]))
        start = split_point

    return chunks


def _collect_inference_output(
    cosyvoice_model: Any,
    source_temp: str,
    target_temp: str,
    speed: float,
) -> torch.Tensor:
    """Run inference_vc and merge streamed model chunks for a single source chunk."""
    output = cosyvoice_model.inference_vc(
        source_wav=source_temp,
        prompt_wav=target_temp,
        stream=False,
        speed=speed,
    )

    speech_parts: List[torch.Tensor] = []
    for model_chunk in output:
        speech_parts.append(model_chunk["tts_speech"])

    if not speech_parts:
        raise RuntimeError("Model returned no audio for voice conversion chunk")

    if len(speech_parts) == 1:
        return speech_parts[0]

    return torch.cat(speech_parts, dim=-1)


def _concatenate_with_crossfade(waveforms: List[torch.Tensor], sample_rate: int) -> torch.Tensor:
    """Concatenate converted chunks with boundary fades while preserving full duration."""
    if not waveforms:
        raise ValueError("No waveforms to concatenate")

    prepared_waveforms: List[torch.Tensor] = []
    for waveform in waveforms:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        prepared_waveforms.append(waveform.detach().cpu())

    combined = prepared_waveforms[0]
    crossfade_samples = max(1, int(sample_rate * CHUNK_CROSSFADE_SECONDS))

    for next_waveform in prepared_waveforms[1:]:
        fade_samples = min(
            crossfade_samples,
            combined.shape[-1] // 2,
            next_waveform.shape[-1] // 2,
        )

        if fade_samples < 1:
            combined = torch.cat([combined, next_waveform], dim=-1)
            continue

        fade_out = torch.linspace(1.0, 0.0, fade_samples, dtype=combined.dtype)
        fade_in = torch.linspace(0.0, 1.0, fade_samples, dtype=next_waveform.dtype)
        boundary_index = combined.shape[-1]
        combined = torch.cat([combined, next_waveform], dim=-1)
        combined[..., boundary_index - fade_samples:boundary_index] *= fade_out
        combined[..., boundary_index:boundary_index + fade_samples] *= fade_in

    return combined


class TS_CosyVoice3_VoiceConversion:
    """
    Voice conversion - convert source voice to target voice
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert_voice"
    CATEGORY = "TS CosyVoice3/Synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL", {
                    "description": "CosyVoice model from ModelLoader",
                    "tooltip": "Загруженная модель CosyVoice из ноды загрузчика."
                }),
                "source_audio": ("AUDIO", {
                    "description": "Source audio to convert",
                    "tooltip": "Исходное аудио, которое нужно преобразовать в другой тембр."
                }),
                "target_audio": ("AUDIO", {
                    "description": "Target voice reference",
                    "tooltip": "Референс целевого голоса; будет обрезан до 30 секунд и приведен к оптимальному формату."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "description": "Speech speed multiplier",
                    "tooltip": "Множитель скорости итоговой речи."
                }),
                "pitch_shift_semitones": ("INT", {
                    "default": 0,
                    "min": -12,
                    "max": 12,
                    "step": 1,
                    "display": "slider",
                    "description": "Pitch shift for source audio in semitones",
                    "tooltip": "Сдвиг высоты тона исходного аудио в полутонах перед voice conversion."
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 42,
                    "min": -1,
                    "max": 2147483647,
                    "description": "Random seed (-1 for random)",
                    "tooltip": "Зерно случайности; значение -1 использует случайный seed."
                }),
            }
        }

    def convert_voice(
        self,
        model: Dict[str, Any],
        source_audio: Dict[str, Any],
        target_audio: Dict[str, Any],
        speed: float = 1.0,
        pitch_shift_semitones: int = 0,
        seed: int = 42
    ) -> Tuple[Dict[str, Any]]:
        """
        Convert source voice to target voice

        Args:
            model: CosyVoice model info dict
            source_audio: Source audio to convert
            target_audio: Target voice reference
            speed: Speech speed
            pitch_shift_semitones: Pitch shift for source audio
            seed: Random seed

        Returns:
            Tuple containing audio dict
        """
        print(f"\n{'='*60}")
        print(f"[TS CosyVoice3 VC] Converting voice...")
        print(f"[TS CosyVoice3 VC] Speed: {speed}x")
        print(f"{'='*60}\n")

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

            # Get model instance
            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate  # Use actual model sample rate (24000 for v2/v3)

            print(f"[TS CosyVoice3 VC] Model sample rate: {sample_rate} Hz")

            # Check if model supports voice conversion
            if not hasattr(cosyvoice_model, 'inference_vc'):
                raise RuntimeError("Model does not support voice conversion")

            processed_source_audio = _prepare_source_audio_for_pitch_shift(
                source_audio,
                min(sample_rate, SOURCE_VC_SAMPLE_RATE, PITCH_SHIFT_WORK_SAMPLE_RATE),
            )
            if pitch_shift_semitones != 0:
                processed_source_audio = _apply_pitch_shift(processed_source_audio, pitch_shift_semitones)

            if abs(pitch_shift_semitones) >= 1e-6:
                print(f"[TS CosyVoice3 VC] Applying pitch shift: {pitch_shift_semitones:+.0f} semitones")

            trimmed_target_audio = _trim_audio_to_duration(target_audio, MAX_TARGET_REFERENCE_SECONDS)
            trimmed_target_audio = _prepare_reference_audio_for_model(trimmed_target_audio, sample_rate)
            trimmed_target_duration = _extract_waveform(trimmed_target_audio).shape[-1] / trimmed_target_audio["sample_rate"]
            if target_duration > MAX_TARGET_REFERENCE_SECONDS:
                print(
                    "[TS CosyVoice3 VC] Target audio is longer than 30s, "
                    f"trimming reference to {trimmed_target_duration:.1f}s"
                )

            source_chunks = _split_source_audio_into_chunks(processed_source_audio)
            total_chunks = len(source_chunks)
            print(
                f"[TS CosyVoice3 VC] Source audio: {source_duration:.1f}s -> "
                f"{total_chunks} chunk(s) up to {MAX_SOURCE_CHUNK_DURATION_SECONDS:.1f}s"
            )

            total_steps = total_chunks + 2
            pbar = comfy.utils.ProgressBar(total_steps)
            pbar.update_absolute(0, total_steps)

            print(f"[TS CosyVoice3 VC] Preparing target audio ({trimmed_target_duration:.1f}s)...")
            _, _, target_temp = prepare_audio_for_cosyvoice(trimmed_target_audio, target_sample_rate=sample_rate)
            pbar.update_absolute(1, total_steps)

            print(f"[TS CosyVoice3 VC] Running voice conversion...")

            converted_chunks: List[torch.Tensor] = []
            for chunk_index, source_chunk in enumerate(source_chunks, start=1):
                chunk_duration = _extract_waveform(source_chunk).shape[-1] / source_chunk["sample_rate"]
                source_temp = None
                print(
                    f"[TS CosyVoice3 VC] Preparing source chunk {chunk_index}/{total_chunks} "
                    f"({chunk_duration:.1f}s)..."
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
                print(f"[TS CosyVoice3 VC] Processed source chunk {chunk_index}/{total_chunks}")
                pbar.update_absolute(chunk_index + 1, total_steps)

            waveform = _concatenate_with_crossfade(converted_chunks, sample_rate)

            # Ensure waveform is on CPU
            if waveform.device != torch.device('cpu'):
                waveform = waveform.cpu()

            # Convert to ComfyUI AUDIO format
            audio = tensor_to_comfyui_audio(waveform, sample_rate)

            duration = waveform.shape[-1] / sample_rate

            pbar.update_absolute(total_steps, total_steps)

            print(f"\n{'='*60}")
            print(f"[TS CosyVoice3 VC] Voice conversion successful!")
            print(f"[TS CosyVoice3 VC] Duration: {duration:.2f} seconds")
            print(f"[TS CosyVoice3 VC] Sample rate: {sample_rate} Hz")
            print(f"{'='*60}\n")

            return (audio,)

        except Exception as e:
            error_msg = f"Error in voice conversion: {str(e)}"
            print(f"\n{'='*60}")
            print(f"[TS CosyVoice3 VC] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            # Return empty audio on error
            empty_audio = {
                "waveform": torch.zeros(1, 1, 22050),
                "sample_rate": 22050
            }
            return (empty_audio,)

        finally:
            # Clean up temp files
            cleanup_temp_file(source_temp)
            cleanup_temp_file(target_temp)
