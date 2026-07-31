"""
TS CosyVoice3 Dialog TTS Node (V3 schema).
Multi-speaker dialog synthesis with voice cloning.

Migrated from V1 to V3 on 2026-05-07.
Public contract preserved:
- node_id: TS_CosyVoice3_Dialog
- inputs (required): model, dialog_text, speaker_A_Audio, speaker_B_Audio, speed
- inputs (optional): speaker_C_Audio, speaker_D_Audio, seed
- outputs (in order): dialog_audio, speaker_a_audio, speaker_b_audio,
                      speaker_c_audio, speaker_d_audio, message
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

from comfy_api.v0_0_2 import IO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_audio_utils import cleanup_temp_file, save_raw_audio_to_tempfile, tensor_to_comfyui_audio
    from ..utils.ts_cosyvoice_adapter import END_OF_PROMPT, SYSTEM_PROMPT, is_cosyvoice3_model_info
    from ..utils.ts_fingerprint import seed_fingerprint
    from ..utils.ts_interrupt import raise_if_interrupted
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import build_empty_audio, collect_speech_chunks, merge_speech_chunks, set_seed
    from ..utils.ts_whisper_utils import transcribe_audio
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from nodes._v3_types import CosyVoiceModel
    from utils.ts_audio_utils import cleanup_temp_file, save_raw_audio_to_tempfile, tensor_to_comfyui_audio
    from utils.ts_cosyvoice_adapter import END_OF_PROMPT, SYSTEM_PROMPT, is_cosyvoice3_model_info
    from utils.ts_fingerprint import seed_fingerprint
    from utils.ts_interrupt import raise_if_interrupted
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import build_empty_audio, collect_speech_chunks, merge_speech_chunks, set_seed
    from utils.ts_whisper_utils import transcribe_audio

import comfy.utils

LOGGER = get_logger("TS CosyVoice Dialog")

MAX_SPEAKER_REFERENCE_SECONDS = 30.0


def _format_prompt_for_model(text: str, is_v3: bool) -> str:
    """Prefix the CosyVoice3 system prompt when the loaded model expects it."""
    if is_v3:
        return f"{SYSTEM_PROMPT}{END_OF_PROMPT}{text}"
    return text


def _validate_audio_duration(audio: Dict[str, Any], speaker_name: str) -> float:
    """Validate audio duration and return it."""
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']
    duration = waveform.shape[-1] / sample_rate

    if duration > MAX_SPEAKER_REFERENCE_SECONDS:
        raise ValueError(
            f"Speaker {speaker_name} reference audio is too long ({duration:.1f} seconds). "
            f"CosyVoice only supports reference audio up to {MAX_SPEAKER_REFERENCE_SECONDS:.0f} seconds. "
            f"Please trim your audio to {MAX_SPEAKER_REFERENCE_SECONDS:.0f} seconds or less. "
            f"Recommended: 3-10 seconds for best quality."
        )
    return duration


def _prepare_speaker_data(
    speakers: Dict[str, Dict[str, Any] | None],
    is_v3: bool,
    temp_file_list: List[str],
) -> Tuple[Dict[str, str], Dict[str, str | None]]:
    """
    Prepare speaker temp files and transcripts.

    Every temp file is appended to the caller-owned ``temp_file_list`` the moment
    it is created, so a failure part-way through (e.g. transcription raising after
    two of four speakers were written) still leaves the caller's ``finally`` able
    to clean up the files already on disk.

    Returns:
        temp_files: Dict mapping speaker ID to temp file path
        transcripts: Dict mapping speaker ID to formatted transcript (or None for cross-lingual fallback)
    """
    temp_files: Dict[str, str] = {}
    transcripts: Dict[str, str | None] = {}

    for speaker_id, audio in speakers.items():
        if audio is None:
            continue

        # Transcribing up to four references is itself slow enough to be worth
        # cancelling, and it happens before the first line is ever generated.
        raise_if_interrupted()
        temp_file = save_raw_audio_to_tempfile(audio)
        temp_file_list.append(temp_file)
        temp_files[speaker_id] = temp_file

        LOGGER.info("[TS CosyVoice3 Dialog] Transcribing Speaker %s reference audio...", speaker_id)
        transcript = transcribe_audio(temp_file, "TS CosyVoice3 Dialog")

        if transcript:
            transcripts[speaker_id] = _format_prompt_for_model(transcript, is_v3)
            LOGGER.info(
                "[TS CosyVoice3 Dialog] Speaker %s transcript: '%s'",
                speaker_id,
                preview_text(transcript),
            )
        else:
            transcripts[speaker_id] = None
            LOGGER.info(
                "[TS CosyVoice3 Dialog] Speaker %s: No transcript (will use cross-lingual mode)",
                speaker_id,
            )

    return temp_files, transcripts


def _parse_dialog_line(line: str) -> Tuple[str | None, str]:
    """Parse a dialog line to extract speaker and content."""
    line = line.strip()
    if not line:
        return None, ""

    prefixes = {
        "SPEAKER A:": "A",
        "SPEAKER B:": "B",
        "SPEAKER C:": "C",
        "SPEAKER D:": "D",
    }

    for prefix, speaker_id in prefixes.items():
        if line.upper().startswith(prefix):
            content = line[len(prefix):].strip()
            return speaker_id, content

    return None, ""


class TS_CosyVoice3_Dialog(IO.ComfyNode):
    """
    Multi-speaker dialog TTS using CosyVoice voice cloning.
    Accepts dialog text with speaker labels and generates audio using separate voice prompts.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_Dialog",
            display_name="TS CosyVoice Dialog",
            category="TS CosyVoice3/Synthesis",
            description="Multi-speaker dialog synthesis with voice cloning (SPEAKER A: ... SPEAKER B: ...).",
            inputs=[
                CosyVoiceModel.Input(
                    "model",
                    tooltip="Загруженная модель CosyVoice из ноды загрузчика.",
                ),
                IO.String.Input(
                    "dialog_text",
                    default="SPEAKER A: Hello, how are you?\nSPEAKER B: I'm doing great, thanks for asking!",
                    multiline=True,
                    tooltip="Текст диалога с репликами вида SPEAKER A:, SPEAKER B: и так далее.",
                ),
                IO.Audio.Input(
                    "speaker_A_Audio",
                    tooltip="Референсный голос для SPEAKER A; желательно 3-10 секунд, максимум 30 секунд.",
                ),
                IO.Audio.Input(
                    "speaker_B_Audio",
                    tooltip="Референсный голос для SPEAKER B; желательно 3-10 секунд, максимум 30 секунд.",
                ),
                IO.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Множитель скорости речи для всех реплик в диалоге.",
                ),
                IO.Audio.Input(
                    "speaker_C_Audio",
                    optional=True,
                    tooltip="Необязательный референсный голос для SPEAKER C.",
                ),
                IO.Audio.Input(
                    "speaker_D_Audio",
                    optional=True,
                    tooltip="Необязательный референсный голос для SPEAKER D.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=-1,
                    max=2147483647,
                    optional=True,
                    tooltip="Зерно случайности; значение -1 использует случайный seed.",
                ),
            ],
            outputs=[
                IO.Audio.Output(display_name="dialog_audio"),
                IO.Audio.Output(display_name="speaker_a_audio"),
                IO.Audio.Output(display_name="speaker_b_audio"),
                IO.Audio.Output(display_name="speaker_c_audio"),
                IO.Audio.Output(display_name="speaker_d_audio"),
                IO.String.Output(display_name="message"),
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
        dialog_text: str,
        speaker_A_Audio: Dict[str, Any],
        speaker_B_Audio: Dict[str, Any],
        speed: float = 1.0,
        speaker_C_Audio: Dict[str, Any] | None = None,
        speaker_D_Audio: Dict[str, Any] | None = None,
        seed: int = 42,
    ) -> IO.NodeOutput:
        """Generate multi-speaker dialog audio."""
        import torch

        log_banner(
            LOGGER,
            "[TS CosyVoice3 Dialog] Starting dialog generation...",
            Speed=f"{speed}x",
        )

        temp_file_list: List[str] = []

        try:
            set_seed(seed)

            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate

            is_v3 = is_cosyvoice3_model_info(model)

            LOGGER.info("[TS CosyVoice3 Dialog] Validating speaker audio durations...")
            _validate_audio_duration(speaker_A_Audio, "A")
            _validate_audio_duration(speaker_B_Audio, "B")
            if speaker_C_Audio is not None:
                _validate_audio_duration(speaker_C_Audio, "C")
            if speaker_D_Audio is not None:
                _validate_audio_duration(speaker_D_Audio, "D")

            speakers: Dict[str, Dict[str, Any] | None] = {
                "A": speaker_A_Audio,
                "B": speaker_B_Audio,
                "C": speaker_C_Audio,
                "D": speaker_D_Audio,
            }

            temp_files, transcripts = _prepare_speaker_data(speakers, is_v3, temp_file_list)

            lines = dialog_text.strip().splitlines()
            valid_lines: List[Tuple[str, str]] = []
            # Skipped lines used to be a log warning only, so a script with a
            # mislabelled prefix rendered short and the user had no idea why.
            speakers_without_audio: List[str] = []
            unparsed_line_count = 0
            for line in lines:
                speaker_id, content = _parse_dialog_line(line)
                if speaker_id and content:
                    if speaker_id in temp_files:
                        valid_lines.append((speaker_id, content))
                    else:
                        if speaker_id not in speakers_without_audio:
                            speakers_without_audio.append(speaker_id)
                        LOGGER.warning(
                            "[TS CosyVoice3 Dialog] Skipping line for Speaker %s: no audio reference provided",
                            speaker_id,
                        )
                elif line.strip():
                    unparsed_line_count += 1
                    LOGGER.warning(
                        "[TS CosyVoice3 Dialog] Skipping unrecognised line: %s",
                        preview_text(line, 60),
                    )

            if not valid_lines:
                LOGGER.warning("[TS CosyVoice3 Dialog] No valid dialog lines found!")
                empty_audio = build_empty_audio(sample_rate)
                return IO.NodeOutput(
                    empty_audio, empty_audio, empty_audio, empty_audio, empty_audio,
                    "No valid dialog lines found. Use format: SPEAKER A: text",
                )

            total_steps = len(valid_lines) + 2
            pbar = comfy.utils.ProgressBar(total_steps)
            pbar.update_absolute(1, total_steps)

            speaker_waveforms: Dict[str, List[torch.Tensor]] = {"A": [], "B": [], "C": [], "D": []}
            combined_waveforms: List[torch.Tensor] = []

            for i, (speaker_id, content) in enumerate(valid_lines):
                # Cancel between lines: a long script is many separate inferences.
                raise_if_interrupted()
                LOGGER.info(
                    "[TS CosyVoice3 Dialog] Generating line %s/%s: Speaker %s",
                    i + 1,
                    len(valid_lines),
                    speaker_id,
                )

                temp_file = temp_files[speaker_id]
                transcript = transcripts.get(speaker_id)

                if transcript:
                    output = cosyvoice_model.inference_zero_shot(
                        tts_text=content,
                        prompt_text=transcript,
                        prompt_wav=temp_file,
                        stream=False,
                        speed=speed,
                    )
                else:
                    formatted_text = _format_prompt_for_model(content, is_v3)

                    output = cosyvoice_model.inference_cross_lingual(
                        tts_text=formatted_text,
                        prompt_wav=temp_file,
                        stream=False,
                        speed=speed,
                    )

                all_speech = collect_speech_chunks(output)
                current_wav = merge_speech_chunks(all_speech)

                if current_wav.device != torch.device('cpu'):
                    current_wav = current_wav.cpu()

                combined_waveforms.append(current_wav)

                silence = torch.zeros_like(current_wav)
                for sid in speaker_waveforms.keys():
                    if sid == speaker_id:
                        speaker_waveforms[sid].append(current_wav)
                    else:
                        speaker_waveforms[sid].append(silence)

                pbar.update_absolute(i + 2, total_steps)

            combined_waveform = torch.cat(combined_waveforms, dim=-1)

            speaker_tracks: Dict[str, torch.Tensor] = {}
            for sid, waveforms in speaker_waveforms.items():
                if waveforms:
                    speaker_tracks[sid] = torch.cat(waveforms, dim=-1)
                else:
                    speaker_tracks[sid] = torch.zeros_like(combined_waveform)

            dialog_audio = tensor_to_comfyui_audio(combined_waveform, sample_rate)
            speaker_a_audio = tensor_to_comfyui_audio(speaker_tracks["A"], sample_rate)
            speaker_b_audio = tensor_to_comfyui_audio(speaker_tracks["B"], sample_rate)
            speaker_c_audio = tensor_to_comfyui_audio(speaker_tracks["C"], sample_rate)
            speaker_d_audio = tensor_to_comfyui_audio(speaker_tracks["D"], sample_rate)

            pbar.update_absolute(total_steps, total_steps)

            duration = combined_waveform.shape[-1] / sample_rate
            message = f"Dialog synthesized successfully! Duration: {duration:.2f}s, Lines: {len(valid_lines)}"
            if speakers_without_audio:
                message += (
                    f". Skipped lines for Speaker "
                    f"{', '.join(sorted(speakers_without_audio))}: no reference audio connected"
                )
            if unparsed_line_count:
                message += (
                    f". Skipped {unparsed_line_count} line(s) with no recognised "
                    f"SPEAKER A:/B:/C:/D: prefix"
                )

            log_banner(
                LOGGER,
                f"[TS CosyVoice3 Dialog] {message}",
                SampleRate=f"{sample_rate} Hz",
            )

            return IO.NodeOutput(
                dialog_audio, speaker_a_audio, speaker_b_audio,
                speaker_c_audio, speaker_d_audio, message,
            )

        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 Dialog] ERROR", exc)
            raise

        finally:
            for temp_file in temp_file_list:
                cleanup_temp_file(temp_file)
