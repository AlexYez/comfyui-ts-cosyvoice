"""
Speaker preset I/O for TS CosyVoice3.

Single entry point for everything that reads or writes the `.pt` speaker presets
stored in `models/cosyvoice/speaker/`:

- name sanitisation (no path separators, no traversal, no Windows-reserved names)
- containment check (a resolved preset path may never escape the speaker dir)
- save / load with `weights_only=True`
- safe injection of a preset into a shared (cached) model frontend

The on-disk format is unchanged: a dict `{speaker_name: model_input}` produced by
`frontend.frontend_zero_shot(...)` with the `text` / `text_len` keys removed.
Presets written by earlier versions load unchanged.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

try:
    from .ts_logging import get_logger
except (ImportError, ValueError):
    from ts_logging import get_logger


LOGGER = get_logger("TS CosyVoice3 Speaker IO")

# Characters that are illegal in Windows filenames, plus the POSIX separator.
# Unicode letters (e.g. Cyrillic preset names) stay allowed on purpose — existing
# user presets rely on them.
_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Reserved device names on Windows; a file called "CON.pt" is not creatable there.
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

NONE_PRESET = "[none]"

_SPEAKER_PRESET_CACHE: tuple[int, list[str]] | None = None


def get_speaker_dir() -> str:
    """Return the speaker preset directory inside the ComfyUI models tree."""
    import folder_paths

    return os.path.join(folder_paths.models_dir, "cosyvoice", "speaker")


def get_speaker_save_dir() -> str:
    """Return the speaker preset directory, creating it when missing."""
    speaker_dir = get_speaker_dir()
    os.makedirs(speaker_dir, exist_ok=True)
    return speaker_dir


def list_speaker_presets() -> list[str]:
    """List saved speaker presets with a lightweight directory mtime cache."""
    global _SPEAKER_PRESET_CACHE

    speaker_dir = get_speaker_dir()
    if not os.path.isdir(speaker_dir):
        return [NONE_PRESET]

    cache_key = os.stat(speaker_dir).st_mtime_ns
    if _SPEAKER_PRESET_CACHE and _SPEAKER_PRESET_CACHE[0] == cache_key:
        return list(_SPEAKER_PRESET_CACHE[1])

    # Case-insensitive extension match. macOS and Windows filesystems are
    # case-insensitive but case-preserving, so a preset copied in from another
    # machine — or renamed in Finder — can legitimately end in `.PT` and would
    # otherwise be invisible in the widget while still existing on disk.
    names = [
        os.path.splitext(filename)[0]
        for filename in sorted(os.listdir(speaker_dir))
        if filename.lower().endswith(".pt")
    ]
    cached_names = names if names else [NONE_PRESET]
    _SPEAKER_PRESET_CACHE = (cache_key, cached_names)
    return list(cached_names)


def sanitize_speaker_name(speaker_name: str) -> str:
    """
    Validate a UI-supplied preset name and return it in canonical form.

    The name must be usable as a bare filename: no directory components, no
    traversal, no characters that are illegal on Windows. Rejecting (rather than
    silently rewriting) keeps the saved path predictable for the user.

    Raises:
        ValueError: if the name cannot be used as a preset filename.
    """
    name = speaker_name.strip()

    if not name:
        raise ValueError("[TS CosyVoice3 Speaker IO] speaker_name cannot be empty.")

    if _ILLEGAL_NAME_CHARS.search(name):
        raise ValueError(
            f"[TS CosyVoice3 Speaker IO] speaker_name contains characters that are not "
            f"allowed in a file name (< > : \" / \\ | ? *): {speaker_name!r}"
        )

    # Path separators are already rejected above, so a name that still changes
    # under basename() is something like "C:name" or a trailing-dot form.
    if os.path.basename(name) != name or name in (".", ".."):
        raise ValueError(
            f"[TS CosyVoice3 Speaker IO] speaker_name must be a plain file name "
            f"without directories: {speaker_name!r}"
        )

    if name.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(
            f"[TS CosyVoice3 Speaker IO] speaker_name is a reserved device name on "
            f"Windows: {speaker_name!r}"
        )

    # A trailing space is already impossible after strip() above; a trailing dot is
    # not (strip removes whitespace, not dots) and is invalid on Windows.
    if name.endswith("."):
        raise ValueError(
            f"[TS CosyVoice3 Speaker IO] speaker_name must not end with a dot: "
            f"{speaker_name!r}"
        )

    return name


def resolve_preset_path(speaker_name: str, speaker_dir: str) -> str:
    """
    Resolve `<speaker_dir>/<speaker_name>.pt` and verify it stays inside speaker_dir.

    The containment check is the backstop: even if a name slipped through
    sanitisation (symlinked dir, exotic separator), the resolved path must remain
    under the preset directory.

    Raises:
        ValueError: if the resolved path escapes speaker_dir.
    """
    name = sanitize_speaker_name(speaker_name)
    candidate = os.path.realpath(os.path.join(speaker_dir, f"{name}.pt"))
    root = os.path.realpath(speaker_dir)

    if os.path.commonpath([candidate, root]) != root:
        raise ValueError(
            f"[TS CosyVoice3 Speaker IO] Resolved preset path escapes the speaker "
            f"directory: {speaker_name!r}"
        )

    return candidate


def save_speaker_preset(speaker_name: str, model_input: Dict[str, Any]) -> str:
    """
    Persist a speaker preset and return the path it was written to.

    Args:
        speaker_name: UI-supplied preset name (validated here).
        model_input: Extracted zero-shot features, without `text` / `text_len`.

    Returns:
        Absolute path of the written `.pt` file.
    """
    import torch

    save_dir = get_speaker_save_dir()
    save_path = resolve_preset_path(speaker_name, save_dir)
    name = sanitize_speaker_name(speaker_name)

    torch.save({name: model_input}, save_path)
    LOGGER.info("[TS CosyVoice3 Speaker IO] Saved preset '%s' to %s", name, os.path.basename(save_path))
    return save_path


def load_speaker_preset(speaker_name: str, device: Any = None) -> Dict[str, Any]:
    """
    Load a speaker preset onto the requested device.

    `weights_only=True` keeps preset loading from executing arbitrary pickle code.
    Presets only ever contain tensors, so this does not change what loads.

    Args:
        speaker_name: Preset name as shown in the combo widget.
        device: torch device to map tensors onto; defaults to CPU.

    Returns:
        The `{speaker_name: model_input}` dict as stored on disk.

    Raises:
        ValueError: if the name is unusable or `[none]`.
        FileNotFoundError: if no preset file exists for that name.
        RuntimeError: if the file is not a plain-tensor preset.
    """
    import torch

    if speaker_name == NONE_PRESET:
        raise ValueError(
            "[TS CosyVoice3 Speaker IO] No speaker presets found. "
            "Use the TS CosyVoice Save Speaker node to create one first."
        )

    speaker_dir = get_speaker_dir()
    pt_path = resolve_preset_path(speaker_name, speaker_dir)

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"[TS CosyVoice3 Speaker IO] Speaker preset file not found: {pt_path}\n"
            f"Please run TS CosyVoice Save Speaker to create it."
        )

    map_location = device if device is not None else "cpu"
    try:
        spk2info = torch.load(pt_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"[TS CosyVoice3 Speaker IO] Could not load speaker preset "
            f"{os.path.basename(pt_path)}: {exc}. Presets must contain only tensors — "
            f"re-create this preset with the TS CosyVoice Save Speaker node."
        ) from exc

    if not isinstance(spk2info, dict) or not spk2info:
        raise RuntimeError(
            f"[TS CosyVoice3 Speaker IO] Speaker preset is empty or malformed: "
            f"{os.path.basename(pt_path)}"
        )

    return spk2info


def inject_speaker_preset(cosyvoice_model: Any, spk2info: Dict[str, Any]) -> str:
    """
    Register a loaded preset on a (shared, cached) model frontend and return its id.

    The model instance lives in the module-level model cache and is shared by every
    node in the graph, so the frontend's `spk2info` dict is updated key-by-key
    instead of being replaced wholesale — replacing it would drop speakers that
    other nodes or earlier runs registered.

    Each entry is copied before insertion so later prompt-token edits never mutate
    the caller's loaded dict.
    """
    frontend = cosyvoice_model.frontend
    if not isinstance(getattr(frontend, "spk2info", None), dict):
        frontend.spk2info = {}

    spk_id = next(iter(spk2info))
    for name, entry in spk2info.items():
        frontend.spk2info[name] = dict(entry) if isinstance(entry, dict) else entry

    return spk_id
