"""Pytest config: ensure pack root is importable as `comfyui_ts_cosyvoice`-style top-level imports work."""

from __future__ import annotations

import os
import sys

PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PACK_ROOT not in sys.path:
    sys.path.insert(0, PACK_ROOT)
