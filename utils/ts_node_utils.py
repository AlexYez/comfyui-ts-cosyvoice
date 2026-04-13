import json
import os
import random
from functools import lru_cache
from typing import Any, Iterable

import folder_paths
import torch


CUSTOM_INSTRUCTION_LABEL = "Ваша инструкция"
_SPEAKER_PRESET_CACHE: tuple[int, list[str]] | None = None


def set_seed(seed: int) -> None:
    """Seed torch and Python RNGs when deterministic generation is requested."""
    if seed < 0:
        return

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_speech_chunks(output: Iterable[dict[str, Any]]) -> list[torch.Tensor]:
    """Collect generated speech tensors from a CosyVoice iterator."""
    chunks: list[torch.Tensor] = []
    for chunk in output:
        speech = chunk.get("tts_speech")
        if speech is not None:
            chunks.append(speech)
    return chunks


def merge_speech_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate all generated speech chunks without losing samples."""
    if not chunks:
        raise RuntimeError("No audio was generated. Check model and inputs.")
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=-1)


def get_speaker_dir() -> str:
    """Return the preset directory used by Timesaver speaker nodes."""
    return os.path.join(folder_paths.models_dir, "cosyvoice", "speaker")


def list_speaker_presets() -> list[str]:
    """List saved speaker presets with a lightweight directory mtime cache."""
    global _SPEAKER_PRESET_CACHE

    speaker_dir = get_speaker_dir()
    if not os.path.isdir(speaker_dir):
        return ["[none]"]

    cache_key = os.stat(speaker_dir).st_mtime_ns
    if _SPEAKER_PRESET_CACHE and _SPEAKER_PRESET_CACHE[0] == cache_key:
        return list(_SPEAKER_PRESET_CACHE[1])

    names = [
        os.path.splitext(filename)[0]
        for filename in sorted(os.listdir(speaker_dir))
        if filename.endswith(".pt")
    ]
    cached_names = names if names else ["[none]"]
    _SPEAKER_PRESET_CACHE = (cache_key, cached_names)
    return list(cached_names)


def get_emotion_presets_path() -> str:
    """Return the shared emotion preset JSON path."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, "configs", "ts_emotion_presets.json")


@lru_cache(maxsize=4)
def _load_emotion_presets_cached(path: str, mtime_ns: int) -> dict[str, str]:
    del mtime_ns

    presets: dict[str, str] = {CUSTOM_INSTRUCTION_LABEL: ""}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

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


def build_empty_audio(sample_rate: int = 22050) -> dict[str, Any]:
    """Return a consistent silent AUDIO payload for graceful node fallback."""
    return {
        "waveform": torch.zeros(1, 1, sample_rate),
        "sample_rate": sample_rate,
    }
