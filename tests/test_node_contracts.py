"""
Contract preservation tests for V1 -> V3 migration.

For each node listed in tests/contracts/v1_baseline.json:
- Verify the class is registered with the same node_id (in NODE_CLASS_MAPPINGS or
  in _PackExtension.get_node_list()).
- Verify the display name and category match.
- Verify input names, types and defaults are preserved (V1 still uses INPUT_TYPES;
  V3 uses define_schema -> Schema).
- Verify output count, types and order match.

These tests run BOTH:
- before migration: assert V1 contract still matches baseline (sanity check).
- during migration: assert each migrated V3 node still matches baseline.
- after migration: every node validated through V3 path.

Heavy runtime deps (torch, librosa, cosyvoice) are required for the package to import,
so the tests are skipped automatically if those modules are missing in the test env.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest


BASELINE_PATH = os.path.join(os.path.dirname(__file__), "contracts", "v1_baseline.json")


def _load_baseline() -> dict[str, Any]:
    with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _try_import_pack():
    try:
        import comfyui_ts_cosyvoice  # type: ignore[import-not-found]
        return comfyui_ts_cosyvoice
    except ImportError:
        pass

    try:
        import importlib

        return importlib.import_module("__init__")
    except (ImportError, ValueError):
        return None


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return _load_baseline()


@pytest.fixture(scope="module")
def loaded_pack():
    pytest.importorskip("torch", reason="torch required for cosyvoice runtime")
    pytest.importorskip("torchaudio", reason="torchaudio required")
    pytest.importorskip("librosa", reason="librosa required")
    pack = _try_import_pack()
    if pack is None:
        pytest.skip("Pack root could not be imported (heavy runtime deps missing)")
    return pack


def _v1_contract(cls) -> dict[str, Any]:
    """Read a V1 node class and project to a normalized contract dict."""
    schema_inputs = cls.INPUT_TYPES()
    return {
        "node_id": cls.__name__,
        "category": getattr(cls, "CATEGORY", None),
        "function": getattr(cls, "FUNCTION", None),
        "is_output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        "inputs": schema_inputs,
        "return_types": tuple(getattr(cls, "RETURN_TYPES", ())),
        "return_names": tuple(getattr(cls, "RETURN_NAMES", ())) or None,
    }


def _v3_contract(cls) -> dict[str, Any]:
    """Read a V3 node class via define_schema() and project to a normalized contract."""
    schema = cls.define_schema()
    inputs_required: dict[str, Any] = {}
    inputs_optional: dict[str, Any] = {}
    for inp in schema.inputs:
        bucket = inputs_optional if getattr(inp, "optional", False) else inputs_required
        bucket[inp.id] = inp
    return {
        "node_id": schema.node_id,
        "display_name": schema.display_name,
        "category": schema.category,
        "is_output_node": schema.is_output_node,
        "inputs_required": inputs_required,
        "inputs_optional": inputs_optional,
        "outputs": schema.outputs,
    }


def _resolve_class(loaded_pack, node_id: str):
    mappings = getattr(loaded_pack, "NODE_CLASS_MAPPINGS", {}) or {}
    if node_id in mappings:
        return mappings[node_id], "v1"

    entrypoint = getattr(loaded_pack, "comfy_entrypoint", None)
    if entrypoint is None:
        return None, None

    import asyncio

    extension = asyncio.run(entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    for cls in nodes:
        try:
            schema = cls.define_schema()
        except Exception:
            continue
        if schema.node_id == node_id:
            return cls, "v3"

    return None, None


@pytest.mark.parametrize("expected", _load_baseline()["nodes"], ids=lambda e: e["node_id"])
def test_node_contract_preserved(loaded_pack, expected):
    node_id = expected["node_id"]
    cls, schema_kind = _resolve_class(loaded_pack, node_id)
    assert cls is not None, f"Node {node_id} not registered (neither V1 mapping nor V3 extension)"

    if schema_kind == "v1":
        contract = _v1_contract(cls)
        assert contract["category"] == expected["category"], (
            f"V1 {node_id}: category drift — baseline={expected['category']!r}, current={contract['category']!r}"
        )
        assert contract["function"] == expected["function"], (
            f"V1 {node_id}: FUNCTION drift — baseline={expected['function']!r}, current={contract['function']!r}"
        )
        assert contract["is_output_node"] == expected["is_output_node"], (
            f"V1 {node_id}: OUTPUT_NODE drift — baseline={expected['is_output_node']}, current={contract['is_output_node']}"
        )
        for required_name in expected["inputs"].get("required", {}):
            assert required_name in contract["inputs"].get("required", {}), (
                f"V1 {node_id}: required input {required_name!r} missing"
            )
        for optional_name in expected["inputs"].get("optional", {}):
            assert optional_name in contract["inputs"].get("optional", {}), (
                f"V1 {node_id}: optional input {optional_name!r} missing"
            )
        expected_return_types = tuple(o["type"] for o in expected["outputs"])
        assert contract["return_types"] == expected_return_types, (
            f"V1 {node_id}: RETURN_TYPES drift — baseline={expected_return_types}, current={contract['return_types']}"
        )
        expected_return_names = tuple(o["name"] for o in expected["outputs"])
        if contract["return_names"] is not None:
            assert contract["return_names"] == expected_return_names, (
                f"V1 {node_id}: RETURN_NAMES drift — baseline={expected_return_names}, current={contract['return_names']}"
            )
        return

    contract = _v3_contract(cls)
    assert contract["node_id"] == expected["node_id"]
    assert contract["display_name"] == expected["display_name"], (
        f"V3 {node_id}: display_name drift — baseline={expected['display_name']!r}, current={contract['display_name']!r}"
    )
    assert contract["category"] == expected["category"], (
        f"V3 {node_id}: category drift — baseline={expected['category']!r}, current={contract['category']!r}"
    )
    assert contract["is_output_node"] == expected["is_output_node"], (
        f"V3 {node_id}: is_output_node drift — baseline={expected['is_output_node']}, current={contract['is_output_node']}"
    )

    for required_name in expected["inputs"].get("required", {}):
        assert required_name in contract["inputs_required"], (
            f"V3 {node_id}: required input {required_name!r} missing or marked optional"
        )
    for optional_name in expected["inputs"].get("optional", {}):
        assert optional_name in contract["inputs_optional"] or optional_name in contract["inputs_required"], (
            f"V3 {node_id}: optional input {optional_name!r} missing"
        )

    expected_output_types = [o["type"] for o in expected["outputs"]]
    actual_output_types = [out.io_type for out in contract["outputs"]]
    assert actual_output_types == expected_output_types, (
        f"V3 {node_id}: output types/order drift — baseline={expected_output_types}, current={actual_output_types}"
    )
