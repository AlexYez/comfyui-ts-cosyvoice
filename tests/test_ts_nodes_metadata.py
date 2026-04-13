import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


NODE_MODULES = [
    "nodes.ts_cosyvoice_model_loader_node",
    "nodes.ts_cosyvoice_cross_language_node",
    "nodes.ts_cosyvoice_voice_to_voice_node",
    "nodes.ts_cosyvoice_dialog_node",
    "nodes.ts_cosyvoice_text_to_voice_node",
    "nodes.ts_cosyvoice_save_speaker_node",
    "nodes.ts_cosyvoice_speaker_to_audio_node",
    "nodes.ts_cosyvoice_speaker_text_to_voice_node",
]


def _iter_input_configs(input_types: dict) -> list[dict]:
    configs = []
    for section in ("required", "optional"):
        for _, value in input_types.get(section, {}).items():
            if isinstance(value, tuple) and len(value) > 1 and isinstance(value[1], dict):
                configs.append(value[1])
    return configs


def test_all_nodes_import_and_define_input_types() -> None:
    for module_name in NODE_MODULES:
        module = importlib.import_module(module_name)
        node_classes = [value for value in module.__dict__.values() if isinstance(value, type) and value.__name__.startswith("TS_")]
        assert node_classes, module_name
        for node_class in node_classes:
            input_types = node_class.INPUT_TYPES()
            assert "required" in input_types


def test_all_node_inputs_have_russian_tooltips() -> None:
    for module_name in NODE_MODULES:
        module = importlib.import_module(module_name)
        node_classes = [value for value in module.__dict__.values() if isinstance(value, type) and value.__name__.startswith("TS_")]
        for node_class in node_classes:
            for config in _iter_input_configs(node_class.INPUT_TYPES()):
                assert "tooltip" in config
                assert isinstance(config["tooltip"], str)
                assert config["tooltip"].strip()
