import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_emotion_presets_file_is_valid() -> None:
    presets_path = ROOT / "configs" / "ts_emotion_presets.json"
    data = json.loads(presets_path.read_text(encoding="utf-8"))
    assert "presets" in data
    assert isinstance(data["presets"], list)
    assert data["presets"]


def test_instruct_nodes_load_shared_presets() -> None:
    text_to_voice = importlib.import_module("nodes.ts_cosyvoice_text_to_voice_node")
    speaker_text_to_voice = importlib.import_module("nodes.ts_cosyvoice_speaker_text_to_voice_node")
    assert text_to_voice.CUSTOM_INSTRUCTION_LABEL in text_to_voice.INSTRUCT_PRESETS
    assert speaker_text_to_voice.CUSTOM_INSTRUCTION_LABEL in speaker_text_to_voice.INSTRUCT_PRESETS
    assert text_to_voice.INSTRUCT_PRESET_OPTIONS == speaker_text_to_voice.INSTRUCT_PRESET_OPTIONS
