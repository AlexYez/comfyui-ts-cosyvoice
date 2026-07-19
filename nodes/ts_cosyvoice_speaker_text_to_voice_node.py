"""
TS CosyVoice speaker-text-to-voice node (V3 schema).

Migrated from V1 to V3 on 2026-05-07.
Public contract preserved:
- node_id: TS_CosyVoice3_SpeakerInstruct2
- inputs (required): model, text, instruct_text, speaker_preset, speed
- inputs (optional): seed, text_normalize, emotion_preset
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
    from ..utils.ts_audio_utils import tensor_to_comfyui_audio
    from ..utils.ts_cosyvoice_adapter import (
        apply_speaker_prompt_tokens,
        format_instruct_text,
        get_runtime_device,
        is_cosyvoice3_model_info,
    )
    from ..utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from ..utils.ts_node_utils import (
        CUSTOM_INSTRUCTION_LABEL,
        collect_speech_chunks,
        load_emotion_presets,
        merge_speech_chunks,
        resolve_instruct_text,
        set_seed,
    )
    from ..utils.ts_speaker_io import (
        NONE_PRESET,
        inject_speaker_preset,
        list_speaker_presets,
        load_speaker_preset,
    )
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from utils.ts_audio_utils import tensor_to_comfyui_audio
    from utils.ts_cosyvoice_adapter import (
        apply_speaker_prompt_tokens,
        format_instruct_text,
        get_runtime_device,
        is_cosyvoice3_model_info,
    )
    from utils.ts_logging import get_logger, log_banner, log_exception, preview_text
    from utils.ts_node_utils import (
        CUSTOM_INSTRUCTION_LABEL,
        collect_speech_chunks,
        load_emotion_presets,
        merge_speech_chunks,
        resolve_instruct_text,
        set_seed,
    )
    from utils.ts_speaker_io import (
        NONE_PRESET,
        inject_speaker_preset,
        list_speaker_presets,
        load_speaker_preset,
    )
    from nodes._v3_types import CosyVoiceModel

import comfy.utils


LOGGER = get_logger("TS CosyVoice Speaker Text To Voice")
# Widget options are read once at schema build time (ComfyUI caches the schema);
# the effective instruction is resolved per call via resolve_instruct_text().
INSTRUCT_PRESET_OPTIONS = list(load_emotion_presets())


class TS_CosyVoice3_SpeakerInstruct2(IO.ComfyNode):
    """Synthesize speech from a saved speaker preset and an emotion instruction."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_SpeakerInstruct2",
            display_name="TS CosyVoice Speaker Text To Voice",
            category="TS CosyVoice3/Synthesis",
            description="Synthesize speech using a saved speaker preset (.pt) and an emotion instruction.",
            inputs=[
                CosyVoiceModel.Input(
                    "model",
                    tooltip="Загруженная модель CosyVoice из ноды загрузчика.",
                ),
                IO.String.Input(
                    "text",
                    default="Hello, this is my cloned voice speaking.",
                    multiline=True,
                    tooltip="Текст, который нужно озвучить выбранным сохраненным голосом.",
                ),
                IO.String.Input(
                    "instruct_text",
                    default="Please say this in a warm and soft voice.",
                    multiline=True,
                    tooltip=f"Текстовая инструкция для эмоции и манеры речи; "
                            f"используется только при выборе '{CUSTOM_INSTRUCTION_LABEL}'.",
                ),
                IO.Combo.Input(
                    "speaker_preset",
                    options=list_speaker_presets(),
                    tooltip="Выберите ранее сохраненный пресет голоса из папки models/cosyvoice/speaker.",
                ),
                IO.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Множитель скорости итоговой речи.",
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
                IO.Combo.Input(
                    "emotion_preset",
                    options=INSTRUCT_PRESET_OPTIONS,
                    default=CUSTOM_INSTRUCTION_LABEL,
                    optional=True,
                    tooltip="Готовый пресет эмоции и манеры подачи; "
                            f"при выборе '{CUSTOM_INSTRUCTION_LABEL}' "
                            "используется поле instruct_text.",
                ),
            ],
            outputs=[
                IO.Audio.Output(display_name="audio"),
            ],
            search_aliases=[],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        model: Dict[str, Any],
        text: str,
        instruct_text: str,
        speaker_preset: str,
        speed: float = 1.0,
        seed: int = 42,
        text_normalize: bool = True,
        emotion_preset: str = CUSTOM_INSTRUCTION_LABEL,
    ) -> IO.NodeOutput:
        """Generate instructed speech using a saved speaker preset."""
        resolved_instruct_text = resolve_instruct_text(instruct_text, emotion_preset)

        log_banner(
            LOGGER,
            "[TS CosyVoice3 SpeakerInstruct2] Synthesizing",
            Preset=speaker_preset,
            Text=preview_text(text),
            EmotionPreset=emotion_preset,
            Speed=f"{speed}x",
        )
        LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Instruct: %s", preview_text(resolved_instruct_text, 80))

        if speaker_preset == NONE_PRESET:
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

            # Load onto the device the model actually runs on. The loader's
            # requested device can disagree with reality (the vendored runtime
            # hardcodes cuda-if-available and ignores the request), so we read the
            # frontend's own device rather than model["device"].
            load_device = get_runtime_device(cosyvoice_model)
            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Loading preset '%s' onto %s", speaker_preset, load_device)
            spk2info = load_speaker_preset(speaker_preset, device=load_device)

            # Merges into the frontend's existing speaker table instead of replacing
            # it: the model instance is shared through the module-level cache.
            spk_id = inject_speaker_preset(cosyvoice_model, spk2info)
            LOGGER.info("[TS CosyVoice3 SpeakerInstruct2] Speaker ID: '%s'", spk_id)
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

            audio = tensor_to_comfyui_audio(waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate
            pbar.update_absolute(3, 3)

            log_banner(
                LOGGER,
                "[TS CosyVoice3 SpeakerInstruct2] Speech generated successfully",
                Duration=f"{duration:.2f} seconds",
                SampleRate=f"{sample_rate} Hz",
            )
            return IO.NodeOutput(audio)
        except Exception as exc:
            log_exception(LOGGER, "[TS CosyVoice3 SpeakerInstruct2] ERROR", exc)
            raise
