"""
Project-specific V3 custom IO types for comfyui-ts-cosyvoice.

Declared once, imported everywhere. Adding a duplicate ``IO.Custom("...")``
declaration elsewhere in the codebase is a desync bug — every node that
consumes ``COSYVOICE_MODEL`` must import the symbol from here.

Notes:
- Pinned to ``comfy_api.v0_0_2``. Never use ``comfy_api.latest`` —
  it is an unstable alias and may change between ComfyUI releases.
- The underscore-prefixed filename (``_v3_types.py``) intentionally signals
  that this module is internal scaffolding, not a public node.
"""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

# The single source of truth for the COSYVOICE_MODEL custom IO type.
# All synthesis nodes (Instruct2, SpeakerInstruct2, CrossLingual,
# VoiceConversion, SaveSpeaker, Dialog) import CosyVoiceModel from here
# and use ``CosyVoiceModel.Input("model", ...)`` in their schemas.
# The ModelLoader exposes ``CosyVoiceModel.Output(display_name="model")``.
CosyVoiceModel = IO.Custom("COSYVOICE_MODEL")


__all__ = ["CosyVoiceModel"]
