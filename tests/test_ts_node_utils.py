import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ts_node_utils import list_speaker_presets, merge_speech_chunks, set_seed
from utils.ts_whisper_utils import is_cosyvoice3_model


def test_set_seed_is_stable() -> None:
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_merge_speech_chunks_preserves_full_length() -> None:
    merged = merge_speech_chunks([torch.ones(1, 5), torch.zeros(1, 7)])
    assert merged.shape == (1, 12)


def test_list_speaker_presets_returns_placeholder_when_empty() -> None:
    presets = list_speaker_presets()
    assert presets


def test_is_cosyvoice3_model_detects_version() -> None:
    assert is_cosyvoice3_model({"model_version": "Fun-CosyVoice3-0.5B"})
    assert not is_cosyvoice3_model({"model_version": "something-else"})
