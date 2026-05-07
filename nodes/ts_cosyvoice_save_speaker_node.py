"""
TS CosyVoice save-speaker node (V3 schema).

Migrated from V1 to V3 on 2026-05-07.
Public contract preserved:
- node_id: TS_CosyVoice3_SaveSpeaker
- inputs (required): model, reference_audio, reference_text, speaker_name
- outputs: (STRING,) named "saved_path"
- is_output_node: True (writes preset .pt to disk)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch

from comfy_api.v0_0_2 import IO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_reference_audio_for_cosyvoice,
        save_raw_audio_to_tempfile,
    )
    from ..utils.ts_cosyvoice_adapter import SYSTEM_PROMPT, END_OF_PROMPT, is_cosyvoice3_model_info
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_whisper_utils import transcribe_audio
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_reference_audio_for_cosyvoice,
        save_raw_audio_to_tempfile,
    )
    from utils.ts_cosyvoice_adapter import SYSTEM_PROMPT, END_OF_PROMPT, is_cosyvoice3_model_info
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_whisper_utils import transcribe_audio
    from nodes._v3_types import CosyVoiceModel

import comfy.utils
import folder_paths


LOGGER = get_logger("TS CosyVoice Save Speaker")


def get_speaker_save_dir() -> str:
    """Return the writable speaker preset directory."""
    speaker_dir = os.path.join(folder_paths.models_dir, "cosyvoice", "speaker")
    os.makedirs(speaker_dir, exist_ok=True)
    return speaker_dir


class TS_CosyVoice3_SaveSpeaker(IO.ComfyNode):
    """Extract zero-shot speaker features and save them for later reuse."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_SaveSpeaker",
            display_name="TS CosyVoice Save Speaker",
            category="TS CosyVoice3/Utilities",
            description="Extract zero-shot speaker features from reference audio and save as a .pt preset.",
            inputs=[
                CosyVoiceModel.Input(
                    "model",
                    tooltip="Загруженная модель CosyVoice из ноды загрузчика.",
                ),
                IO.Audio.Input(
                    "reference_audio",
                    tooltip="Референсное аудио для сохранения тембра голоса; "
                            "будет обрезано до 30 секунд и приведено к mono 24 kHz.",
                ),
                IO.String.Input(
                    "reference_text",
                    default="",
                    multiline=True,
                    tooltip="Текст, который произносится в референсном аудио; "
                            "если оставить пустым, будет использована "
                            "авторасшифровка Whisper.",
                ),
                IO.String.Input(
                    "speaker_name",
                    default="my_speaker",
                    multiline=False,
                    tooltip="Имя сохраняемого пресета голоса без расширения "
                            "файла; будет использовано как имя .pt файла.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="saved_path"),
            ],
            search_aliases=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model: Dict[str, Any],
        reference_audio: Dict[str, Any],
        reference_text: str,
        speaker_name: str,
    ) -> IO.NodeOutput:
        """Extract and persist a reusable speaker preset."""
        log_banner(
            LOGGER,
            "[TS CosyVoice3 SaveSpeaker] Extracting speaker features",
            Speaker=speaker_name,
            ReferenceText=preview_text(reference_text, 60),
        )

        speaker_name = speaker_name.strip()
        if not speaker_name:
            raise ValueError("speaker_name cannot be empty.")

        temp_file = None
        try:
            pbar = comfy.utils.ProgressBar(3)
            ref_waveform = reference_audio["waveform"]
            ref_sample_rate = reference_audio["sample_rate"]
            ref_duration = ref_waveform.shape[-1] / ref_sample_rate
            processed_reference_audio = prepare_reference_audio_for_cosyvoice(
                reference_audio,
                target_sample_rate=REFERENCE_AUDIO_SAMPLE_RATE,
                max_duration_seconds=REFERENCE_AUDIO_MAX_SECONDS,
            )
            processed_ref_duration = (
                processed_reference_audio["waveform"].shape[-1] / processed_reference_audio["sample_rate"]
            )

            LOGGER.info("[TS CosyVoice3 SaveSpeaker] Saving reference audio to temp file...")
            LOGGER.info(
                "[TS CosyVoice3 SaveSpeaker] Reference audio: %.1fs -> %.1fs, mono/%s Hz",
                ref_duration,
                processed_ref_duration,
                REFERENCE_AUDIO_SAMPLE_RATE,
            )
            pbar.update_absolute(0, 3)
            temp_file = save_raw_audio_to_tempfile(processed_reference_audio)

            LOGGER.info("[TS CosyVoice3 SaveSpeaker] Extracting features via frontend_zero_shot...")
            pbar.update_absolute(1, 3)

            cosyvoice_model = model["model"]
            if not reference_text.strip():
                LOGGER.info("[TS CosyVoice3 SaveSpeaker] No reference_text provided, auto-transcribing with Whisper...")
                reference_text = transcribe_audio(temp_file, "TS CosyVoice3 SaveSpeaker")
                if reference_text:
                    LOGGER.info("[TS CosyVoice3 SaveSpeaker] Transcribed: '%s'", preview_text(reference_text, 80))
                else:
                    LOGGER.warning(
                        "[TS CosyVoice3 SaveSpeaker] Transcription failed, using empty reference_text"
                    )

            prompt_text = reference_text
            if is_cosyvoice3_model_info(model):
                prompt_text = f"{SYSTEM_PROMPT}{END_OF_PROMPT}{reference_text}"
                LOGGER.info("[TS CosyVoice3 SaveSpeaker] CosyVoice3 detected: prepended system prompt")
            else:
                LOGGER.info("[TS CosyVoice3 SaveSpeaker] Legacy CosyVoice mode: using raw reference_text")

            model_input = cosyvoice_model.frontend.frontend_zero_shot(
                "", prompt_text, temp_file, cosyvoice_model.sample_rate, ""
            )
            del model_input["text"]
            del model_input["text_len"]

            spk2info = {speaker_name: model_input}
            pbar.update_absolute(2, 3)

            save_dir = get_speaker_save_dir()
            save_path = os.path.join(save_dir, f"{speaker_name}.pt")
            torch.save(spk2info, save_path)
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 SaveSpeaker] Saved successfully",
                Path=save_path,
                SpeakerKey=speaker_name,
                Keys=", ".join(spk2info[speaker_name].keys()),
            )
            return IO.NodeOutput(save_path)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 SaveSpeaker] ERROR", exc)
            raise RuntimeError(f"Error saving speaker preset: {exc}") from exc
        finally:
            cleanup_temp_file(temp_file)
