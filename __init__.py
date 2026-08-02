"""
TS CosyVoice3 - Advanced Text-to-Speech for ComfyUI (V3 schema only).
Zero-shot voice cloning, cross-lingual synthesis, and instruction-based control.

Registration is V3-only: every node inherits from IO.ComfyNode and is exposed
through the ComfyExtension returned by comfy_entrypoint(). The legacy
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS dicts were dropped in the
0.7 release after the V1 -> V3 migration completed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from comfy_api.v0_0_2 import ComfyExtension

try:
    from .utils.ts_logging import get_logger
except ImportError:
    from utils.ts_logging import get_logger

try:
    from .nodes.ts_cosyvoice_cross_language_node import TS_CosyVoice3_CrossLingual
    from .nodes.ts_cosyvoice_dialog_node import TS_CosyVoice3_Dialog
    from .nodes.ts_cosyvoice_model_loader_node import TS_CosyVoice3_ModelLoader
    from .nodes.ts_cosyvoice_save_speaker_node import TS_CosyVoice3_SaveSpeaker
    from .nodes.ts_cosyvoice_speaker_text_to_voice_node import TS_CosyVoice3_SpeakerInstruct2
    from .nodes.ts_cosyvoice_text_to_voice_node import TS_CosyVoice3_Instruct2
    from .nodes.ts_cosyvoice_voice_to_voice_node import TS_CosyVoice3_VoiceConversion
except ImportError:
    from nodes.ts_cosyvoice_cross_language_node import TS_CosyVoice3_CrossLingual
    from nodes.ts_cosyvoice_dialog_node import TS_CosyVoice3_Dialog
    from nodes.ts_cosyvoice_model_loader_node import TS_CosyVoice3_ModelLoader
    from nodes.ts_cosyvoice_save_speaker_node import TS_CosyVoice3_SaveSpeaker
    from nodes.ts_cosyvoice_speaker_text_to_voice_node import TS_CosyVoice3_SpeakerInstruct2
    from nodes.ts_cosyvoice_text_to_voice_node import TS_CosyVoice3_Instruct2
    from nodes.ts_cosyvoice_voice_to_voice_node import TS_CosyVoice3_VoiceConversion


# Single source of truth for the pack version at runtime. Kept in lockstep with
# pyproject.toml (and the README badge) by tests/test_version_consistency.py.
__version__ = "1.0.0"

# Serves the generated embedded help pages at
# /extensions/comfyui-ts-cosyvoice/docs/<node_id>/<locale>.md, which is where the
# frontend's per-node help button looks. There is no JavaScript in this folder —
# the pack ships no frontend extension. Regenerate with
# `python tools/generate_node_docs.py`.
WEB_DIRECTORY = "./web"

_V3_NODES: list[type] = [
    TS_CosyVoice3_ModelLoader,
    TS_CosyVoice3_Instruct2,
    TS_CosyVoice3_SpeakerInstruct2,
    TS_CosyVoice3_CrossLingual,
    TS_CosyVoice3_VoiceConversion,
    TS_CosyVoice3_SaveSpeaker,
    TS_CosyVoice3_Dialog,
]


class _PackExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return list(_V3_NODES)


async def comfy_entrypoint() -> ComfyExtension:
    return _PackExtension()


get_logger("TS CosyVoice").info("TS CosyVoice3 Custom Nodes Loaded - Version %s", __version__)

__all__ = ["WEB_DIRECTORY", "__version__", "comfy_entrypoint"]
