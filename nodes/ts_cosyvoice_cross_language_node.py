"""TS CosyVoice cross-language node."""

import os
import sys
from typing import Any, Dict, Tuple

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_audio_for_cosyvoice,
        prepare_reference_audio_for_cosyvoice,
        tensor_to_comfyui_audio,
    )
    from ..utils.ts_cosyvoice_adapter import format_cross_lingual_text, is_cosyvoice3_model_info
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import build_empty_audio, collect_speech_chunks, merge_speech_chunks, set_seed
except (ImportError, ValueError):
    from utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_audio_for_cosyvoice,
        prepare_reference_audio_for_cosyvoice,
        tensor_to_comfyui_audio,
    )
    from utils.ts_cosyvoice_adapter import format_cross_lingual_text, is_cosyvoice3_model_info
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import build_empty_audio, collect_speech_chunks, merge_speech_chunks, set_seed

import comfy.utils


LOGGER = get_logger("TS CosyVoice Cross Language")


class TS_CosyVoice3_CrossLingual:
    """Generate speech in another language while preserving the reference timbre."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "cross_lingual_synthesis"
    CATEGORY = "TS CosyVoice3/Synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL", {
                    "description": "CosyVoice model from ModelLoader",
                    "tooltip": "Загруженная модель CosyVoice из ноды загрузчика.",
                }),
                "text": ("STRING", {
                    "default": "Hello, this is cross-lingual speech synthesis.",
                    "multiline": True,
                    "description": "Text to synthesize in target language",
                    "tooltip": "Текст, который нужно озвучить на целевом языке.",
                }),
                "reference_audio": ("AUDIO", {
                    "description": "Reference voice (can be in any language)",
                    "tooltip": "Референсный голос; аудио будет обрезано "
                               "до 30 секунд и приведено к mono 24 kHz.",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "description": "Speech speed multiplier",
                    "tooltip": "Множитель скорости речи на выходе.",
                }),
            },
            "optional": {
                "target_language": (["auto", "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"], {
                    "default": "auto",
                    "description": "Target language (auto-detect from text)",
                    "tooltip": "Целевой язык текста; auto пытается "
                               "определить язык автоматически.",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": -1,
                    "max": 2147483647,
                    "description": "Random seed (-1 for random)",
                    "tooltip": "Зерно случайности; значение -1 "
                               "использует случайный seed.",
                }),
                "text_normalize": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable text normalization. Disable for CMU phonemes or special tags like <slow>",
                    "tooltip": "Включает нормализацию текста; "
                               "отключайте для фонем CMU и специальных тегов.",
                }),
            },
        }

    def cross_lingual_synthesis(
        self,
        model: Dict[str, Any],
        text: str,
        reference_audio: Dict[str, Any],
        speed: float = 1.0,
        target_language: str = "auto",
        seed: int = 42,
        text_normalize: bool = True,
    ) -> Tuple[Dict[str, Any]]:
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
            _, _, temp_file = prepare_audio_for_cosyvoice(
                processed_reference_audio,
                target_sample_rate=REFERENCE_AUDIO_SAMPLE_RATE,
            )

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
            if waveform.device != torch.device("cpu"):
                waveform = waveform.cpu()

            audio = tensor_to_comfyui_audio(waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 CrossLingual] Speech generated successfully",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )
            return (audio,)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 CrossLingual] ERROR", exc)
            return (build_empty_audio(),)
        finally:
            cleanup_temp_file(temp_file)
