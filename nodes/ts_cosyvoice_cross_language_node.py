"""
TS CosyVoice cross-language node (V3 schema).

Migrated from V1 to V3 on 2026-05-07.
Public contract preserved:
- node_id: TS_CosyVoice3_CrossLingual
- inputs (required): model, text, reference_audio, speed
- inputs (optional): target_language, seed, text_normalize
- outputs: (AUDIO,) named "audio"
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

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
        tensor_to_comfyui_audio,
    )
    from ..utils.ts_cosyvoice_adapter import format_cross_lingual_text, is_cosyvoice3_model_info
    from ..utils.ts_fingerprint import seed_fingerprint
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import collect_speech_chunks, merge_speech_chunks, set_seed
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from nodes._v3_types import CosyVoiceModel
    from utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_reference_audio_for_cosyvoice,
        save_raw_audio_to_tempfile,
        tensor_to_comfyui_audio,
    )
    from utils.ts_cosyvoice_adapter import format_cross_lingual_text, is_cosyvoice3_model_info
    from utils.ts_fingerprint import seed_fingerprint
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import collect_speech_chunks, merge_speech_chunks, set_seed

import comfy.utils

LOGGER = get_logger("TS CosyVoice Cross Language")

_TARGET_LANGUAGES = ["auto", "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"]


class TS_CosyVoice3_CrossLingual(IO.ComfyNode):
    """Generate speech in another language while preserving the reference timbre."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_CrossLingual",
            display_name="TS CosyVoice Cross-Language",
            category="TS CosyVoice3/Synthesis",
            description="Cross-language speech synthesis: keep the reference timbre, switch the language.",
            inputs=[
                CosyVoiceModel.Input(
                    "model",
                    tooltip="Загруженная модель CosyVoice из ноды загрузчика.",
                ),
                IO.String.Input(
                    "text",
                    default="Hello, this is cross-lingual speech synthesis.",
                    multiline=True,
                    tooltip="Текст, который нужно озвучить на целевом языке.",
                ),
                IO.Audio.Input(
                    "reference_audio",
                    tooltip="Референсный голос; аудио будет обрезано "
                            "до 30 секунд и приведено к mono 24 kHz.",
                ),
                IO.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Множитель скорости речи на выходе.",
                ),
                IO.Combo.Input(
                    "target_language",
                    options=_TARGET_LANGUAGES,
                    default="auto",
                    optional=True,
                    tooltip="Целевой язык текста; auto пытается определить язык автоматически.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=-1,
                    max=2147483647,
                    optional=True,
                    tooltip="Зерно случайности; значение -1 использует случайный seed.",
                ),
                IO.Boolean.Input(
                    "text_normalize",
                    default=True,
                    optional=True,
                    tooltip="Включает нормализацию текста; отключайте для фонем CMU и специальных тегов.",
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
        text: str,
        reference_audio: Dict[str, Any],
        speed: float = 1.0,
        target_language: str = "auto",
        seed: int = 42,
        text_normalize: bool = True,
    ) -> IO.NodeOutput:
        """Generate cross-language speech from text and a voice reference."""
        log_banner(
            LOGGER,
            "[TS CosyVoice3 CrossLingual] Generating cross-language speech",
            Text=preview_text(text),
            TargetLanguage=target_language,
            Speed=f"{speed}x",
        )

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

        temp_file = None
        try:
            set_seed(seed)
            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate
            pbar = comfy.utils.ProgressBar(3)

            pbar.update_absolute(0, 3)
            LOGGER.info("[TS CosyVoice3 CrossLingual] Preparing reference audio...")
            LOGGER.info("[TS CosyVoice3 CrossLingual] Model sample rate: %s Hz", sample_rate)
            LOGGER.info(
                "[TS CosyVoice3 CrossLingual] Reference audio: %.1fs -> %.1fs, mono/%s Hz",
                ref_duration,
                processed_ref_duration,
                REFERENCE_AUDIO_SAMPLE_RATE,
            )
            # The reference is already mono / 24 kHz / trimmed from the call above,
            # so write it straight to a temp file — the same path Instruct2 and
            # SaveSpeaker take. Re-running mono/resample here would be a no-op.
            temp_file = save_raw_audio_to_tempfile(processed_reference_audio)

            is_v3 = is_cosyvoice3_model_info(model)
            formatted_text = format_cross_lingual_text(text, is_v3, target_language)
            LOGGER.info(
                "[TS CosyVoice3 CrossLingual] Using %s formatting path",
                "CosyVoice3" if is_v3 else "legacy CosyVoice",
            )
            LOGGER.info("[TS CosyVoice3 CrossLingual] Formatted text: %s", preview_text(formatted_text, 100))

            pbar.update_absolute(1, 3)
            LOGGER.info("[TS CosyVoice3 CrossLingual] Running cross-language inference...")
            output = cosyvoice_model.inference_cross_lingual(
                tts_text=formatted_text,
                prompt_wav=temp_file,
                stream=False,
                speed=speed,
                text_frontend=text_normalize,
            )

            all_speech = collect_speech_chunks(output)
            for chunk_count in range(1, len(all_speech) + 1):
                LOGGER.info("[TS CosyVoice3 CrossLingual] Processed chunk %s", chunk_count)

            waveform = merge_speech_chunks(all_speech)
            if len(all_speech) > 1:
                LOGGER.info("[TS CosyVoice3 CrossLingual] Combined %s chunks", len(all_speech))

            pbar.update_absolute(2, 3)

            audio = tensor_to_comfyui_audio(waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 CrossLingual] Speech generated successfully",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )
            return IO.NodeOutput(audio)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 CrossLingual] ERROR", exc)
            raise
        finally:
            cleanup_temp_file(temp_file)
