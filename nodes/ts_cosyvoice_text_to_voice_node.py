"""TS CosyVoice text-to-voice node."""

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
        prepare_reference_audio_for_cosyvoice,
        save_raw_audio_to_tempfile,
        tensor_to_comfyui_audio,
    )
    from ..utils.ts_cosyvoice_adapter import format_instruct_text, is_cosyvoice3_model_info
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import (
        CUSTOM_INSTRUCTION_LABEL,
        build_empty_audio,
        collect_speech_chunks,
        load_emotion_presets,
        merge_speech_chunks,
        set_seed,
    )
except (ImportError, ValueError):
    from utils.ts_audio_utils import (
        REFERENCE_AUDIO_MAX_SECONDS,
        REFERENCE_AUDIO_SAMPLE_RATE,
        cleanup_temp_file,
        prepare_reference_audio_for_cosyvoice,
        save_raw_audio_to_tempfile,
        tensor_to_comfyui_audio,
    )
    from utils.ts_cosyvoice_adapter import format_instruct_text, is_cosyvoice3_model_info
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import (
        CUSTOM_INSTRUCTION_LABEL,
        build_empty_audio,
        collect_speech_chunks,
        load_emotion_presets,
        merge_speech_chunks,
        set_seed,
    )

import comfy.utils


LOGGER = get_logger("TS CosyVoice Text to Voice")
INSTRUCT_PRESETS = load_emotion_presets()
INSTRUCT_PRESET_OPTIONS = list(INSTRUCT_PRESETS)


class TS_CosyVoice3_Instruct2:
    """Generate speech from text with reference-guided timbre and instruction control."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_with_instruct"
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
                    "default": "Hello, this is my cloned voice speaking.",
                    "multiline": True,
                    "description": "Text to synthesize in cloned voice",
                    "tooltip": "Текст, который нужно озвучить голосом из референса.",
                }),
                "instruct_text": ("STRING", {
                    "default": "Speak in a warm and friendly tone.",
                    "multiline": True,
                    "description": "Instructions to control speaking style, emotion, and tone. "
                                   f"Works only when preset is set to '{CUSTOM_INSTRUCTION_LABEL}'. "
                                   "Examples: 'Speak slowly and gently', "
                                   "'Use an excited and energetic tone', "
                                   "'Sound calm and professional'.",
                    "tooltip": "Текстовая инструкция для эмоции и манеры речи; "
                               "используется только при выборе пункта "
                               f"'{CUSTOM_INSTRUCTION_LABEL}'.",
                }),
                "reference_audio": ("AUDIO", {
                    "description": "Reference voice to clone (max 30 seconds, recommended 3-10s)",
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
                    "tooltip": "Множитель скорости итоговой речи.",
                }),
            },
            "optional": {
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
                "emotion_preset": (INSTRUCT_PRESET_OPTIONS, {
                    "default": CUSTOM_INSTRUCTION_LABEL,
                    "description": "Preset for emotional delivery and speaking style",
                    "tooltip": "Готовый пресет эмоции и манеры подачи; "
                               f"при выборе '{CUSTOM_INSTRUCTION_LABEL}' "
                               "используется поле instruct_text.",
                }),
            },
        }

    def generate_with_instruct(
        self,
        model: Dict[str, Any],
        text: str,
        instruct_text: str,
        reference_audio: Dict[str, Any],
        speed: float = 1.0,
        seed: int = 42,
        text_normalize: bool = True,
        emotion_preset: str = CUSTOM_INSTRUCTION_LABEL,
    ) -> Tuple[Dict[str, Any]]:
        """Generate instructed speech from text and reference audio."""
        log_banner(
            LOGGER,
            "[TS CosyVoice3 Instruct2] Generating instructed speech",
            Text=preview_text(text),
            Preset=emotion_preset,
            Speed=f"{speed}x",
        )

        resolved_instruct_text = (
            instruct_text.strip()
            if emotion_preset == CUSTOM_INSTRUCTION_LABEL
            else INSTRUCT_PRESETS.get(emotion_preset, "")
        )
        LOGGER.info("[TS CosyVoice3 Instruct2] Instruct: %s", preview_text(resolved_instruct_text, limit=80))

        if not resolved_instruct_text:
            raise ValueError(
                "Instruction text cannot be empty. Please select a preset or provide your own instruction, "
                "e.g. 'Speak in a warm, friendly tone'."
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
            is_v3 = is_cosyvoice3_model_info(model)

            LOGGER.info("[TS CosyVoice3 Instruct2] Model sample rate: %s Hz", sample_rate)
            LOGGER.info("[TS CosyVoice3 Instruct2] Is CosyVoice3: %s", is_v3)
            LOGGER.info(
                "[TS CosyVoice3 Instruct2] Reference audio: %.1fs -> %.1fs, mono/%s Hz",
                ref_duration,
                processed_ref_duration,
                REFERENCE_AUDIO_SAMPLE_RATE,
            )

            if not hasattr(cosyvoice_model, "inference_instruct2"):
                raise RuntimeError(
                    "inference_instruct2 is not available on this model. "
                    "The Instruct node requires a CosyVoice2 or CosyVoice3 model. "
                    "Please load a compatible model in the Model Loader."
                )

            pbar = comfy.utils.ProgressBar(4)

            LOGGER.info("[TS CosyVoice3 Instruct2] Saving reference audio...")
            pbar.update_absolute(0, 4)
            temp_file = save_raw_audio_to_tempfile(processed_reference_audio)

            pbar.update_absolute(1, 4)
            formatted_instruct = format_instruct_text(resolved_instruct_text, is_v3)
            LOGGER.info(
                "[TS CosyVoice3 Instruct2] Using %s instruct mode",
                "CosyVoice3" if is_v3 else "CosyVoice2",
            )
            LOGGER.info("[TS CosyVoice3 Instruct2] Formatted instruct: %s", preview_text(formatted_instruct, 100))

            pbar.update_absolute(2, 4)

            LOGGER.info("[TS CosyVoice3 Instruct2] Running instruct inference...")
            output = cosyvoice_model.inference_instruct2(
                tts_text=text,
                instruct_text=formatted_instruct,
                prompt_wav=temp_file,
                zero_shot_spk_id="",
                stream=False,
                speed=speed,
                text_frontend=text_normalize,
            )

            all_speech = collect_speech_chunks(output)
            for chunk_count in range(1, len(all_speech) + 1):
                LOGGER.info("[TS CosyVoice3 Instruct2] Processed chunk %s", chunk_count)

            waveform = merge_speech_chunks(all_speech)
            if len(all_speech) > 1:
                LOGGER.info("[TS CosyVoice3 Instruct2] Combined %s chunks", len(all_speech))

            pbar.update_absolute(3, 4)
            if waveform.device != torch.device("cpu"):
                waveform = waveform.cpu()

            audio = tensor_to_comfyui_audio(waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate

            pbar.update_absolute(4, 4)
            log_banner(
                LOGGER,
                "[TS CosyVoice3 Instruct2] Speech generated successfully",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )
            return (audio,)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 Instruct2] ERROR", exc)
            return (build_empty_audio(),)
        finally:
            cleanup_temp_file(temp_file)
