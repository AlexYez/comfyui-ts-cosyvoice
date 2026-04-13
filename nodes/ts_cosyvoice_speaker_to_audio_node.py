"""TS CosyVoice speaker-to-audio node."""

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
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import (
        build_empty_audio,
        collect_speech_chunks,
        get_speaker_dir,
        list_speaker_presets,
        merge_speech_chunks,
        set_seed,
    )
except (ImportError, ValueError):
    from utils.ts_audio_utils import tensor_to_comfyui_audio
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import (
        build_empty_audio,
        collect_speech_chunks,
        get_speaker_dir,
        list_speaker_presets,
        merge_speech_chunks,
        set_seed,
    )

import comfy.utils


LOGGER = get_logger("TS CosyVoice Speaker To Audio")


class TS_CosyVoice3_SpeakerClone:
    """Synthesize speech using a saved speaker preset."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "sft_clone"
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
                    "description": "Text to synthesize using the selected speaker preset",
                    "tooltip": "Текст, который нужно озвучить выбранным "
                               "сохраненным пресетом голоса.",
                }),
                "speaker_preset": (presets, {
                    "description": "Speaker preset saved by TS CosyVoice3 Save Speaker",
                    "tooltip": "Выберите ранее сохраненный пресет "
                               "голоса из папки models/cosyvoice/speaker.",
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
            },
        }

    def sft_clone(
        self,
        model: Dict[str, Any],
        text: str,
        speaker_preset: str,
        speed: float = 1.0,
        seed: int = 42,
        text_normalize: bool = True,
    ) -> Tuple[Dict[str, Any]]:
        """Load a speaker preset and synthesize speech via inference_zero_shot."""
        log_banner(
            LOGGER,
            "[TS CosyVoice3 SpeakerClone] Synthesizing with speaker preset",
            Preset=speaker_preset,
            Text=preview_text(text),
            Speed=f"{speed}x",
        )

        if speaker_preset == "[none]":
            raise ValueError(
                "No speaker presets found. "
                "Please use the TS CosyVoice3 Save Speaker node to create one first."
            )

        try:
            set_seed(seed)
            cosyvoice_model = model["model"]
            sample_rate = cosyvoice_model.sample_rate
            pbar = comfy.utils.ProgressBar(3)

            pbar.update_absolute(0, 3)
            speaker_dir = get_speaker_dir()
            pt_path = os.path.join(speaker_dir, f"{speaker_preset}.pt")
            if not os.path.isfile(pt_path):
                raise FileNotFoundError(
                    f"Speaker preset file not found: {pt_path}\n"
                    f"Please run TS CosyVoice3 Save Speaker to create it."
                )

            LOGGER.info("[TS CosyVoice3 SpeakerClone] Loading preset from: %s", pt_path)
            load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            spk2info = torch.load(pt_path, map_location=load_device)
            LOGGER.info("[TS CosyVoice3 SpeakerClone] Preset loaded to device: %s", load_device)

            spk_id = next(iter(spk2info))
            LOGGER.info("[TS CosyVoice3 SpeakerClone] Speaker ID: '%s'", spk_id)

            pbar.update_absolute(1, 3)
            cosyvoice_model.frontend.spk2info = spk2info
            LOGGER.info("[TS CosyVoice3 SpeakerClone] Injected spk2info into model frontend")

            LOGGER.info("[TS CosyVoice3 SpeakerClone] Running inference_zero_shot...")
            output = cosyvoice_model.inference_zero_shot(
                tts_text=text,
                prompt_text="",
                prompt_wav=None,
                zero_shot_spk_id=spk_id,
                stream=False,
                speed=speed,
                text_frontend=text_normalize,
            )

            all_speech = collect_speech_chunks(output)
            for chunk_count in range(1, len(all_speech) + 1):
                LOGGER.info("[TS CosyVoice3 SpeakerClone] Processed chunk %s", chunk_count)

            waveform = merge_speech_chunks(all_speech)
            if len(all_speech) > 1:
                LOGGER.info("[TS CosyVoice3 SpeakerClone] Combined %s chunks", len(all_speech))

            pbar.update_absolute(2, 3)
            if waveform.device != torch.device("cpu"):
                waveform = waveform.cpu()

            audio = tensor_to_comfyui_audio(waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 SpeakerClone] Speech generated successfully",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )
            return (audio,)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 SpeakerClone] ERROR", exc)
            return (build_empty_audio(),)
