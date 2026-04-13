"""
TS CosyVoice3 - Advanced Text-to-Speech for ComfyUI
Zero-shot voice cloning, cross-lingual synthesis, and instruction-based control
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from .utils.ts_logging import get_logger
except ImportError:
    from utils.ts_logging import get_logger

try:
    from .nodes.ts_cosyvoice_model_loader_node import TS_CosyVoice3_ModelLoader
    from .nodes.ts_cosyvoice_cross_language_node import TS_CosyVoice3_CrossLingual
    from .nodes.ts_cosyvoice_voice_to_voice_node import TS_CosyVoice3_VoiceConversion
    from .nodes.ts_cosyvoice_dialog_node import TS_CosyVoice3_Dialog
    from .nodes.ts_cosyvoice_text_to_voice_node import TS_CosyVoice3_Instruct2
    from .nodes.ts_cosyvoice_save_speaker_node import TS_CosyVoice3_SaveSpeaker
    from .nodes.ts_cosyvoice_speaker_to_audio_node import TS_CosyVoice3_SpeakerClone
    from .nodes.ts_cosyvoice_speaker_text_to_voice_node import TS_CosyVoice3_SpeakerInstruct2
except ImportError:
    from nodes.ts_cosyvoice_model_loader_node import TS_CosyVoice3_ModelLoader
    from nodes.ts_cosyvoice_cross_language_node import TS_CosyVoice3_CrossLingual
    from nodes.ts_cosyvoice_voice_to_voice_node import TS_CosyVoice3_VoiceConversion
    from nodes.ts_cosyvoice_dialog_node import TS_CosyVoice3_Dialog
    from nodes.ts_cosyvoice_text_to_voice_node import TS_CosyVoice3_Instruct2
    from nodes.ts_cosyvoice_save_speaker_node import TS_CosyVoice3_SaveSpeaker
    from nodes.ts_cosyvoice_speaker_to_audio_node import TS_CosyVoice3_SpeakerClone
    from nodes.ts_cosyvoice_speaker_text_to_voice_node import TS_CosyVoice3_SpeakerInstruct2

NODE_CLASS_MAPPINGS = {
    "TS_CosyVoice3_ModelLoader": TS_CosyVoice3_ModelLoader,
    "TS_CosyVoice3_CrossLingual": TS_CosyVoice3_CrossLingual,
    "TS_CosyVoice3_VoiceConversion": TS_CosyVoice3_VoiceConversion,
    "TS_CosyVoice3_Dialog": TS_CosyVoice3_Dialog,
    "TS_CosyVoice3_Instruct2": TS_CosyVoice3_Instruct2,
    "TS_CosyVoice3_SaveSpeaker": TS_CosyVoice3_SaveSpeaker,
    "TS_CosyVoice3_SpeakerClone": TS_CosyVoice3_SpeakerClone,
    "TS_CosyVoice3_SpeakerInstruct2": TS_CosyVoice3_SpeakerInstruct2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_CosyVoice3_ModelLoader": "TS CosyVoice Model Loader",
    "TS_CosyVoice3_CrossLingual": "TS CosyVoice Cross-Language",
    "TS_CosyVoice3_VoiceConversion": "TS CosyVoice Voice To Voice",
    "TS_CosyVoice3_Dialog": "TS CosyVoice Dialog",
    "TS_CosyVoice3_Instruct2": "TS CosyVoice Text to Voice",
    "TS_CosyVoice3_SaveSpeaker": "TS CosyVoice Save Speaker",
    "TS_CosyVoice3_SpeakerClone": "TS CosyVoice Speaker To Audio",
    "TS_CosyVoice3_SpeakerInstruct2": "TS CosyVoice Speaker Text To Voice",
}

get_logger("TS CosyVoice").info("TS CosyVoice3 Custom Nodes Loaded - Version 1.2.1")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
