"""
End-to-end tests against a live, locally-running ComfyUI server.

These tests require a running ComfyUI instance (default: http://127.0.0.1:8188).
If the server is unreachable, every test in this module is skipped with a clear
reason — they do not silently pass.

What we check:
- /system_stats responds (server is alive).
- /object_info exposes all 7 TS_CosyVoice3 nodes.
- Each registered node's runtime contract matches tests/contracts/v1_baseline.json
  (display_name, category, output types and order, output names, is_output_node,
  required vs optional input names).
- /prompt accepts a minimally-valid graph for our pack (validation path only,
  short timeout — we cancel via /interrupt before expensive inference can run).

Run with::

    "<comfy-portable-python>" -m pytest -v tests/e2e/test_comfyui_live.py

Override the server URL with COMFYUI_BASE_URL env var if needed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import pytest


COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get("COMFYUI_HTTP_TIMEOUT", "30"))

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "contracts",
    "v1_baseline.json",
)


def _http_get(path: str) -> Any:
    url = f"{COMFYUI_BASE_URL}{path}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read())


def _http_post(path: str, body: dict[str, Any]) -> Any:
    url = f"{COMFYUI_BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        raw = response.read()
        if not raw:
            return None
        return json.loads(raw)


def _server_alive() -> bool:
    try:
        _http_get("/system_stats")
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _server_alive(),
    reason=f"ComfyUI server not reachable at {COMFYUI_BASE_URL} (set COMFYUI_BASE_URL to override).",
)


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    return _http_get("/object_info")


@pytest.fixture(scope="module")
def system_stats() -> dict[str, Any]:
    return _http_get("/system_stats")


def test_system_stats(system_stats):
    """Sanity: server reports a comfyui_version and at least one device."""
    system = system_stats.get("system", {})
    assert "comfyui_version" in system, "ComfyUI did not report comfyui_version"
    assert system_stats.get("devices"), "ComfyUI did not report any devices"


def test_all_seven_ts_nodes_registered(object_info, baseline):
    """All 7 TS_CosyVoice3 nodes are visible to the live server."""
    expected_ids = {n["node_id"] for n in baseline["nodes"]}
    actual_ids = {k for k in object_info if k.startswith("TS_CosyVoice3_")}
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    assert not missing, f"Missing nodes from /object_info: {sorted(missing)}"
    assert not extra, f"Unexpected TS_CosyVoice3_* nodes from /object_info: {sorted(extra)}"
    assert len(actual_ids) == 7, f"Expected exactly 7 nodes, got {len(actual_ids)}"


@pytest.mark.parametrize(
    "expected",
    json.load(open(BASELINE_PATH, "r", encoding="utf-8"))["nodes"],
    ids=lambda e: e["node_id"],
)
def test_node_contract_via_http(object_info, expected):
    """Each live node matches the V1 baseline (display, category, outputs, inputs)."""
    node_id = expected["node_id"]
    assert node_id in object_info, f"{node_id} not registered on the live server"
    info = object_info[node_id]

    assert info.get("display_name") == expected["display_name"], (
        f"{node_id}: display_name drift — baseline={expected['display_name']!r}, "
        f"live={info.get('display_name')!r}"
    )
    assert info.get("category") == expected["category"], (
        f"{node_id}: category drift — baseline={expected['category']!r}, "
        f"live={info.get('category')!r}"
    )
    assert bool(info.get("output_node")) == expected["is_output_node"], (
        f"{node_id}: output_node drift — baseline={expected['is_output_node']}, "
        f"live={info.get('output_node')}"
    )

    expected_output_types = [o["type"] for o in expected["outputs"]]
    actual_outputs = list(info.get("output", []))
    assert actual_outputs == expected_output_types, (
        f"{node_id}: outputs drift — baseline={expected_output_types}, live={actual_outputs}"
    )

    expected_output_names = [o["name"] for o in expected["outputs"]]
    actual_output_names = list(info.get("output_name", []))
    assert actual_output_names == expected_output_names, (
        f"{node_id}: output_name drift — baseline={expected_output_names}, live={actual_output_names}"
    )

    inputs_payload = info.get("input", {})
    required_live = inputs_payload.get("required", {}) or {}
    optional_live = inputs_payload.get("optional", {}) or {}

    for required_name in expected["inputs"].get("required", {}):
        assert required_name in required_live, (
            f"{node_id}: required input {required_name!r} missing on live server. "
            f"Live required={list(required_live)}"
        )

    for optional_name in expected["inputs"].get("optional", {}):
        assert optional_name in optional_live or optional_name in required_live, (
            f"{node_id}: optional input {optional_name!r} missing on live server. "
            f"Live optional={list(optional_live)}, required={list(required_live)}"
        )


def test_cosyvoice_model_type_threading(object_info):
    """ModelLoader output type matches what synthesis nodes expect on the input side."""
    loader = object_info["TS_CosyVoice3_ModelLoader"]
    assert "COSYVOICE_MODEL" in loader.get("output", []), (
        "ModelLoader is not advertising COSYVOICE_MODEL as an output type"
    )

    consumers = [
        "TS_CosyVoice3_Instruct2",
        "TS_CosyVoice3_SpeakerInstruct2",
        "TS_CosyVoice3_CrossLingual",
        "TS_CosyVoice3_VoiceConversion",
        "TS_CosyVoice3_SaveSpeaker",
        "TS_CosyVoice3_Dialog",
    ]
    for consumer_id in consumers:
        info = object_info[consumer_id]
        required = info.get("input", {}).get("required", {}) or {}
        assert "model" in required, f"{consumer_id}: missing required 'model' input"
        # ComfyUI represents the input type as the first element of the schema tuple/list.
        type_payload = required["model"]
        if isinstance(type_payload, (list, tuple)) and type_payload:
            type_token = type_payload[0]
        else:
            type_token = type_payload
        assert type_token == "COSYVOICE_MODEL", (
            f"{consumer_id}: 'model' input expects {type_token!r}, "
            f"but ModelLoader outputs COSYVOICE_MODEL"
        )


def _post_prompt(body: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    """POST to /prompt and normalize HTTPError into (status, body) so callers can branch."""
    try:
        response = _http_post("/prompt", body)
        return 200, (response if isinstance(response, dict) else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def test_prompt_validation_recognizes_our_v3_loader():
    """
    Sanity: ComfyUI's /prompt parser knows our V3 ModelLoader exists.

    We post a 1-node graph whose only flaw is that nothing in it is an output
    node. ComfyUI should reject the prompt with the well-known 'prompt_no_outputs'
    error, NOT with 'unknown class_type' or 'invalid_prompt'. That proves
    /object_info registration is wired all the way into the prompt validator.
    """
    body = {
        "prompt": {
            "1": {
                "class_type": "TS_CosyVoice3_ModelLoader",
                "inputs": {
                    "model_version": "Fun-CosyVoice3-0.5B",
                    "download_source": "HuggingFace",
                    "device": "auto",
                    "fp16": False,
                },
            },
        },
        "client_id": "ts-cosyvoice-e2e",
    }

    status, payload = _post_prompt(body)
    assert status == 400, f"Expected HTTP 400 for output-less prompt, got {status}: {payload!r}"
    assert isinstance(payload, dict), f"Expected JSON error body, got: {payload!r}"
    error = payload.get("error") or {}
    assert error.get("type") == "prompt_no_outputs", (
        f"Expected error.type='prompt_no_outputs', got {error!r}. "
        "If ComfyUI returned a different error, our V3 node may not be registered correctly."
    )


def test_prompt_validation_reports_missing_required_input_via_node_errors():
    """
    Build a 2-node graph using our SaveSpeaker (is_output_node=True) so the
    prompt parses past 'prompt_no_outputs', then deliberately omit the required
    'reference_audio' input. ComfyUI should report a per-node validation error
    pointing at SaveSpeaker. That proves the V3 schema is the source of truth
    for required/optional input flags on the live server.
    """
    body = {
        "prompt": {
            "1": {
                "class_type": "TS_CosyVoice3_ModelLoader",
                "inputs": {
                    "model_version": "Fun-CosyVoice3-0.5B",
                    "download_source": "HuggingFace",
                    "device": "auto",
                    "fp16": False,
                },
            },
            "2": {
                "class_type": "TS_CosyVoice3_SaveSpeaker",
                "inputs": {
                    "model": ["1", 0],
                    # reference_audio intentionally missing — V3 schema marks it required
                    "reference_text": "",
                    "speaker_name": "ts_e2e_probe",
                },
            },
        },
        "client_id": "ts-cosyvoice-e2e",
    }

    status, payload = _post_prompt(body)
    assert isinstance(payload, dict), f"Expected JSON body, got: {payload!r}"

    # ComfyUI may answer in two shapes for missing-required-input:
    #   - HTTP 400 with error.type == 'invalid_prompt'
    #   - HTTP 200/400 with non-empty node_errors and prompt NOT queued
    node_errors = payload.get("node_errors") or {}
    if node_errors:
        node_2_errors = node_errors.get("2", {})
        assert node_2_errors, (
            f"Expected validation error on node '2' (SaveSpeaker), got node_errors={node_errors}"
        )
        rendered = json.dumps(node_2_errors).lower()
        assert "reference_audio" in rendered or "required" in rendered, (
            f"SaveSpeaker validation didn't mention the missing required input: {node_2_errors}"
        )
        return

    # Fallback path: top-level error with invalid_prompt.
    error = payload.get("error") or {}
    assert error, f"Expected non-empty error or node_errors, got: {payload!r}"
    error_type = error.get("type") or ""
    error_blob = json.dumps(payload).lower()
    assert error_type in {"invalid_prompt", "prompt_outputs_failed_validation"} or (
        "reference_audio" in error_blob or "required" in error_blob
    ), (
        f"Expected ComfyUI to fail validation with a recognizable error, got: {payload!r}"
    )

    # Defensive: if the prompt somehow got queued, cancel it so we don't
    # accidentally trigger a real run.
    prompt_id = payload.get("prompt_id")
    if prompt_id:
        try:
            _http_post("/interrupt", {})
        except Exception:
            pass
