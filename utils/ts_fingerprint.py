"""
Cache-invalidation helpers for TS CosyVoice3 nodes (V3 ``fingerprint_inputs``).

ComfyUI builds a node's cache key as ``[class_type, fingerprint] + inputs``, so a
fingerprint only ever *adds* invalidation: returning a constant is safe, and
returning something that never compares equal forces a re-run.

Two things this module exists to get right:

- **A fresh NaN per call.** Since Python 3.10 ``hash(float("nan"))`` derives from
  the object's identity, so two separately created NaNs land in different cache
  slots — which is what forces the re-run. A module-level NaN constant would be
  the *same* object every time and would therefore cache normally, silently
  undoing the effect. ``always_rerun()`` must build a new one on every call.
- **Fingerprints never raise.** They run during prompt validation, before
  ``execute``. A bad preset name has to surface as the node's own actionable
  error, not as a validation crash, so lookups here degrade to a stable marker.
"""

from __future__ import annotations

import os
from typing import Any


def always_rerun() -> float:
    """
    Return a value that never matches a previous run, forcing re-execution.

    Always constructs a new float: see the module docstring for why reusing one
    would quietly disable the invalidation.
    """
    return float("nan")


def is_always_rerun(value: Any) -> bool:
    """True when ``value`` is the always-rerun marker produced above."""
    return isinstance(value, float) and value != value


def seed_fingerprint(seed: Any) -> Any:
    """
    Fingerprint the seed widget, honouring the documented meaning of -1.

    The tooltip promises that -1 means "random". Without this, a node whose
    inputs did not change is served from cache, so the promised new take never
    happens — the user sees the identical audio no matter how often they run it.
    """
    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        # Validation will reject it anyway; keep the fingerprint total.
        return "seed:invalid"

    if seed_value < 0:
        return always_rerun()
    return f"seed:{seed_value}"


def speaker_preset_fingerprint(speaker_preset: Any) -> str:
    """
    Fingerprint a speaker preset by the file it names, not just its name.

    The preset lives on disk and can be re-saved under the same name. Keying on
    the name alone means the node keeps returning the voice from before the
    re-save, because nothing in its inputs changed.

    Returns a stable marker when the preset cannot be resolved or read: the node
    itself raises a clear error at execute time, and a fingerprint must not.
    """
    name = str(speaker_preset)
    try:
        try:
            from .ts_speaker_io import get_speaker_dir, resolve_preset_path
        except (ImportError, ValueError):
            from ts_speaker_io import get_speaker_dir, resolve_preset_path  # type: ignore[no-redef]

        preset_path = resolve_preset_path(name, get_speaker_dir())
        stat_result = os.stat(preset_path)
    except (OSError, ValueError, ImportError):
        # Missing file, rejected name, or no ComfyUI runtime to resolve against.
        return f"preset:{name}:absent"

    return f"preset:{name}:{stat_result.st_mtime_ns}:{stat_result.st_size}"


def combine_fingerprints(*parts: Any) -> Any:
    """
    Merge fingerprint parts, preserving an always-rerun request.

    If any part asks for a re-run the whole fingerprint must, otherwise joining
    it into a string would turn NaN into the literal ``"nan"`` — a perfectly
    stable value, and the invalidation would be lost.
    """
    for part in parts:
        if is_always_rerun(part):
            return always_rerun()
    return "|".join(str(part) for part in parts)
