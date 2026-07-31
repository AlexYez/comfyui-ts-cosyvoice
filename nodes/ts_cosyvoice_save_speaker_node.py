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

from comfy_api.v0_0_2 import IO, UI

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
    from ..utils.ts_cosyvoice_adapter import END_OF_PROMPT, SYSTEM_PROMPT, is_cosyvoice3_model_info
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_speaker_io import sanitize_speaker_name, save_speaker_preset
    from ..utils.ts_whisper_utils import transcribe_audio_with_status
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from nodes._v3_types import CosyVoiceModel
    from utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_reference_audio_for_cosyvoice,
        save_raw_audio_to_tempfile,
    )
    from utils.ts_cosyvoice_adapter import END_OF_PROMPT, SYSTEM_PROMPT, is_cosyvoice3_model_info
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_speaker_io import sanitize_speaker_name, save_speaker_preset
    from utils.ts_whisper_utils import transcribe_audio_with_status

import comfy.utils

LOGGER = get_logger("TS CosyVoice Save Speaker")


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
            essentials_category="Audio",
        )

    @classmethod
    def validate_inputs(cls, speaker_name):
        """
        Reject an unusable preset name before the graph runs.

        Without this the name is only checked inside execute(), i.e. after the
        model loader has already spent time loading 1.5 GB of weights, just to
        fail on a typo the frontend could have flagged immediately.

        Deliberately declared without **kwargs: ComfyUI skips its own min/max and
        combo-membership checks for every input once a validator accepts kwargs
        (execution.py), and naming only this parameter keeps the built-in
        validation intact for the rest.
        """
        try:
            sanitize_speaker_name(speaker_name)
        except ValueError as exc:
            return str(exc)
        return True

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

        # Reject path separators / traversal / reserved names before any disk work.
        speaker_name = sanitize_speaker_name(speaker_name)

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
            transcription_warning: str | None = None
            if not reference_text.strip():
                LOGGER.info("[TS CosyVoice3 SaveSpeaker] No reference_text provided, auto-transcribing with Whisper...")
                reference_text, transcription_warning = transcribe_audio_with_status(
                    temp_file, "TS CosyVoice3 SaveSpeaker"
                )
                if reference_text:
                    LOGGER.info("[TS CosyVoice3 SaveSpeaker] Transcribed: '%s'", preview_text(reference_text, 80))
                else:
                    LOGGER.warning(
                        "[TS CosyVoice3 SaveSpeaker] %s Saving preset with an empty reference text.",
                        transcription_warning,
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

            pbar.update_absolute(2, 3)

            save_path = save_speaker_preset(speaker_name, model_input)
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 SaveSpeaker] Saved successfully",
                Path=save_path,
                SpeakerKey=speaker_name,
                Keys=", ".join(model_input.keys()),
            )

            # saved_path stays a plain path — it is a public output others may
            # consume. The degraded-save warning goes to the node body instead,
            # where it is actually visible; previously it only reached the log.
            status = f"Saved speaker preset '{speaker_name}'."
            if transcription_warning:
                status += (
                    f"\n\nWarning: {transcription_warning}"
                    f"\nThe preset was saved with an empty reference text, which clones"
                    f" noticeably worse. Fill in reference_text by hand and save again"
                    f" for a better result."
                )
            return IO.NodeOutput(save_path, ui=UI.PreviewText(status))
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 SaveSpeaker] ERROR", exc)
            raise RuntimeError(f"Error saving speaker preset: {exc}") from exc
        finally:
            cleanup_temp_file(temp_file)
