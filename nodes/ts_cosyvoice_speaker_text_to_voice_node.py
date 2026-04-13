"""TS CosyVoice speaker-text-to-voice node."""

import os
import sys
from typing import Any, Dict, Tuple

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_audio_utils import tensor_to_comfyui_audio
    from ..utils.ts_cosyvoice_adapter import (
        apply_speaker_prompt_tokens,
        format_instruct_text,
        is_cosyvoice3_model_info,
    )
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import (
        CUSTOM_INSTRUCTION_LABEL,
        build_empty_audio,
        collect_speech_chunks,
        get_speaker_dir,
        list_speaker_presets,
        load_emotion_presets,
        merge_speech_chunks,
        set_seed,
    )
except (ImportError, ValueError):
    from utils.ts_audio_utils import tensor_to_comfyui_audio
    from utils.ts_cosyvoice_adapter import apply_speaker_prompt_tokens, format_instruct_text, is_cosyvoice3_model_info
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import (
        CUSTOM_INSTRUCTION_LABEL,
        build_empty_audio,
        collect_speech_chunks,
        get_speaker_dir,
        list_speaker_presets,
        load_emotion_presets,
        merge_speech_chunks,
        set_seed,
    )

import comfy.utils


LOGGER = get_logger("TS CosyVoice Speaker Text To Voice")
INSTRUCT_PRESETS = load_emotion_presets()
INSTRUCT_PRESET_OPTIONS = list(INSTRUCT_PRESETS)


class TS_CosyVoice3_SpeakerInstruct2:
    """Synthesize speech from a saved speaker preset and an emotion instruction."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speaker_instruct2"
    CATEGORY = "TS CosyVoice3/Synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        presets = list_speaker_presets()
        return {
            "required": {
                "model": ("COSYVOICE_MODEL", {
                    "description": "CosyVoice model from Model Loader",
                    "tooltip": "Загруженная модель CosyVoice из ноды загрузчика.",
                }),
                "text": ("STRING", {
                    "default": "Hello, this is my cloned voice speaking.",
                    "multiline": True,
                    "description": "Text to synthesize",
                    "tooltip": "Текст, который нужно озвучить выбранным "
                               "сохраненным голосом.",
                }),
                "instruct_text": ("STRING", {
                    "default": "Please say this in a warm and soft voice.",
                    "multiline": True,
                    "description": "Instructions to control speaking style, emotion, and tone. "
                                   f"Works only when preset is set to '{CUSTOM_INSTRUCTION_LABEL}'.",
                    "tooltip": "Текстовая инструкция для эмоции и манеры речи; "
                               f"используется только при выборе '{CUSTOM_INSTRUCTION_LABEL}'.",
                }),
                "speaker_preset": (presets, {
                    "description": "Speaker preset saved by TS CosyVoice3 Save Speaker",
                    "tooltip": "Выберите ранее сохраненный пресет голоса "
                               "из папки models/cosyvoice/speaker.",
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
                    "description": "Enable text normalization. Disable for CMU phonemes or special tags",
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

    def speaker_instruct2(
        self,
        model: Dict[str, Any],
        text: str,
        instruct_text: str,
        speaker_preset: str,
        speed: float = 1.0,
        seed: int = 42,
        text_normalize: bool = True,
        emotion_preset: str = CUSTOM_INSTRUCTION_LABEL,
    ) -> Tuple[Dict[str, Any]]:
        """Generate instructed speech using a saved speaker preset."""
        resolved_instruct_text = (
            instruct_text.strip()
            if emotion_preset == CUSTOM_INSTRUCTION_LABEL
            else INSTRUCT_PRESETS.get(emotion_preset, "")
        )

        log_banner(
            LOGGER,
            "[TS CosyVoice3 SpeakerInstruct2] Synthesizing",
            Preset=speaker_preset,
            Text=preview_text(text),
            EmotionPreset=emotion_preset,
            Speed=f"{speed}x",
        )
        LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Instruct: %s", preview_text(resolved_instruct_text, 80))

        if speaker_preset == "[none]":
            raise ValueError(
                "No speaker presets found. "
                "Please use the TS CosyVoice3 Save Speaker node to create one first."
            )
        if not resolved_instruct_text:
            raise ValueError(
                "Instruction text cannot be empty. Please select a preset or provide your own instruction."
            )

        try:
            set_seed(seed)
            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate

            if not hasattr(cosyvoice_model, "inference_instruct2"):
                raise RuntimeError(
                    "inference_instruct2 is not available on this model. "
                    "The Speaker Instruct2 node requires a CosyVoice2 or CosyVoice3 model."
                )

            pbar = comfy.utils.ProgressBar(3)
            pbar.update_absolute(0, 3)

            speaker_dir = get_speaker_dir()
            pt_path = os.path.join(speaker_dir, f"{speaker_preset}.pt")
            if not os.path.isfile(pt_path):
                raise FileNotFoundError(
                    f"Speaker preset file not found: {pt_path}\n"
                    f"Please run TS CosyVoice3 Save Speaker to create it."
                )

            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Loading preset from: %s", pt_path)
            load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            spk2info = torch.load(pt_path, map_location=load_device)
            spk_id = next(iter(spk2info))
            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Speaker ID: '%s'", spk_id)

            cosyvoice_model.frontend.spk2info = spk2info
            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Injected spk2info into model frontend")
            pbar.update_absolute(1, 3)

            is_v3 = is_cosyvoice3_model_info(model)
            formatted_instruct = format_instruct_text(resolved_instruct_text, is_v3)
            LOGGER.info(
                "[TS CosyVoice3 SpeakerInstruct2] Using %s instruct mode",
                "CosyVoice3" if is_v3 else "CosyVoice2",
            )
            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Formatted instruct: %s", preview_text(formatted_instruct, 100))

            apply_speaker_prompt_tokens(cosyvoice_model, spk_id, formatted_instruct)
            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Overwrote prompt_text with formatted instruct tokens")

            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Running inference_instruct2...")
            output = cosyvoice_model.inference_instruct2(
                tts_text=text,
                instruct_text=formatted_instruct,
                prompt_wav=None,
                zero_shot_spk_id=spk_id,
                stream=False,
                speed=speed,
                text_frontend=text_normalize,
            )

            all_speech = collect_speech_chunks(output)
            for chunk_count in range(1, len(all_speech) + 1):
                LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Processed chunk %s", chunk_count)

            waveform = merge_speech_chunks(all_speech)
            if len(all_speech) > 1:
                LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Combined %s chunks", len(all_speech))

            pbar.update_absolute(2, 3)
            if waveform.device != torch.device("cpu"):
                waveform = waveform.cpu()

            audio = tensor_to_comfyui_audio(waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 SpeakerInstruct2] Speech generated successfully",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )
            return (audio,)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 SpeakerInstruct2] ERROR", exc)
            return (build_empty_audio(),)
