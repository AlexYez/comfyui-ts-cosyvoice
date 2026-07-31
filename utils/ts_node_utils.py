"""
Shared node helpers for TS CosyVoice3.

Speaker preset paths / listing / persistence live in `ts_speaker_io`; this module
keeps the generic helpers (seeding, chunk merging, emotion presets, fallback audio).
Import preset helpers from `ts_speaker_io` directly — this module does not re-export
them.

`torch` is imported inside the functions that need it: every synthesis node imports
this module at pack import time, and §14 keeps heavy imports out of that path.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from typing import Any, Iterable

try:
    from .ts_interrupt import raise_if_interrupted
    from .ts_logging import get_logger
except (ImportError, ValueError):
    from ts_interrupt import raise_if_interrupted
    from ts_logging import get_logger


CUSTOM_INSTRUCTION_LABEL = "Ваша инструкция"

LOGGER = get_logger("TS CosyVoice3 Node Utils")


def set_seed(seed: int) -> None:
    """Seed torch and Python RNGs when deterministic generation is requested."""
    import torch

    if seed < 0:
        return

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_speech_chunks(output: Iterable[dict[str, Any]]) -> list[Any]:
    """
    Collect generated speech tensors from a CosyVoice iterator.

    The iterator drives the actual generation, so draining it is where a
    synthesis node spends nearly all of its time. Checking for cancellation
    between chunks is what makes Cancel work for every text-to-speech node in the
    pack; without it the flag is only read after the node has already finished.
    """
    chunks: list[Any] = []
    for chunk in output:
        raise_if_interrupted()
        speech = chunk.get("tts_speech")
        if speech is not None:
            chunks.append(speech)
    return chunks


def merge_speech_chunks(chunks: list[Any]) -> Any:
    """Concatenate all generated speech chunks without losing samples."""
    import torch

    if not chunks:
        raise RuntimeError("No audio was generated. Check model and inputs.")
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=-1)


def get_emotion_presets_path() -> str:
    """Return the shared emotion preset JSON path."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, "configs", "ts_emotion_presets.json")


@lru_cache(maxsize=4)
def _load_emotion_presets_cached(path: str, mtime_ns: int) -> dict[str, str]:
    del mtime_ns

    presets: dict[str, str] = {CUSTOM_INSTRUCTION_LABEL: ""}

    # This file is user-editable, and the emotion preset options are read at pack
    # import time. A malformed JSON must not raise here: a bare parse error would
    # propagate through the node module import and stop ComfyUI from registering
    # any node of the pack. Degrade to the built-in custom-instruction option.
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:  # ValueError covers json.JSONDecodeError
        LOGGER.warning(
            "[TS CosyVoice3 Node Utils] Could not read emotion presets from %s: %s. "
            "Falling back to the custom-instruction option only.",
            path,
            exc,
        )
        return presets

    if not isinstance(payload, dict):
        LOGGER.warning(
            "[TS CosyVoice3 Node Utils] Emotion preset file %s is not a JSON object; "
            "falling back to the custom-instruction option only.",
            path,
        )
        return presets

    preset_items = payload.get("presets", [])
    if not isinstance(preset_items, list):
        return presets

    for preset in preset_items:
        preset_name = str(preset.get("name", "")).strip()
        preset_instruction = str(preset.get("instruction", "")).strip()
        if preset_name and preset_instruction:
            presets[preset_name] = preset_instruction
    return presets


def load_emotion_presets() -> dict[str, str]:
    """Load shared emotion presets from disk with cache invalidation on file change."""
    path = get_emotion_presets_path()
    if not os.path.isfile(path):
        return {CUSTOM_INSTRUCTION_LABEL: ""}
    return _load_emotion_presets_cached(path, os.stat(path).st_mtime_ns)


def resolve_instruct_text(instruct_text: str, emotion_preset: str) -> str:
    """
    Resolve the effective instruction for a synthesis call.

    Presets are re-read through the mtime-invalidated cache on every call, so an
    edit to configs/ts_emotion_presets.json takes effect without a ComfyUI restart.
    """
    if emotion_preset == CUSTOM_INSTRUCTION_LABEL:
        return instruct_text.strip()
    return load_emotion_presets().get(emotion_preset, "")


def build_empty_audio(sample_rate: int) -> dict[str, Any]:
    """
    Return a silent AUDIO payload of one second at the given sample rate.

    Used for the graceful "nothing to synthesize" paths (e.g. a dialog with no
    parseable lines). Callers pass the model's own sample rate so the silence
    matches the rate other outputs use; genuine failures raise instead of
    returning silence.
    """
    import torch

    return {
        "waveform": torch.zeros(1, 1, sample_rate),
        "sample_rate": sample_rate,
    }
