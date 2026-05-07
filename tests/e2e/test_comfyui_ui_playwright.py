"""
End-to-end UI tests against a live ComfyUI frontend, driven by Playwright Chromium.

What we check:
- The ComfyUI page loads and exposes its global app/LiteGraph runtime.
- LiteGraph.registered_node_types contains all 7 TS_CosyVoice3 nodes.
- For each node, the runtime LiteGraph node class advertises the same
  category, output types/order, and required input names that V1 baseline JSON
  declares (i.e. what the user actually sees and connects in the canvas).
- A real graph instantiation works: we instantiate the V3 ModelLoader on the
  canvas via app.graph.add(LiteGraph.createNode(...)) and confirm the node
  reports widgets and outputs that match the schema.
- A screenshot of ComfyUI with a TS node visible is saved as proof for the user.

Skipped automatically if:
- The ComfyUI server is unreachable (same gate as the HTTP tests).
- Playwright cannot launch Chromium (browser not installed, sandbox issue, etc).

Run with::

    "<comfy-portable-python>" -m pytest -v tests/e2e/test_comfyui_ui_playwright.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest


COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get("COMFYUI_HTTP_TIMEOUT", "30"))

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "contracts",
    "v1_baseline.json",
)
SCREENSHOT_DIR = Path(os.path.dirname(__file__)) / "_artifacts"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _server_alive() -> bool:
    try:
        urllib.request.urlopen(f"{COMFYUI_BASE_URL}/system_stats", timeout=HTTP_TIMEOUT).read()
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError):
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not _server_alive(),
        reason=f"ComfyUI server not reachable at {COMFYUI_BASE_URL}",
    ),
]


playwright = pytest.importorskip("playwright.sync_api", reason="playwright not installed")


@pytest.fixture(scope="module")
def baseline() -> dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def browser_page():
    """Launch headless Chromium, open ComfyUI, and yield a Page object."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:
            pytest.skip(f"Cannot launch Chromium: {exc}")

        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.set_default_timeout(60_000)

        try:
            page.goto(COMFYUI_BASE_URL, wait_until="domcontentloaded")
        except Exception as exc:
            browser.close()
            pytest.skip(f"Cannot open ComfyUI page: {exc}")

        # ComfyUI exposes window.LiteGraph and window.app once its bundle has booted.
        page.wait_for_function(
            "() => !!window.LiteGraph && !!window.LiteGraph.registered_node_types",
            timeout=60_000,
        )
        # Some builds finish hooking up the graph asynchronously; give it a moment.
        page.wait_for_function(
            "() => Object.keys(window.LiteGraph.registered_node_types || {})"
            ".some(k => k.startsWith('TS_CosyVoice3_'))",
            timeout=60_000,
        )

        yield page

        try:
            context.close()
            browser.close()
        except Exception:
            pass


def test_litegraph_registered_all_seven_nodes(browser_page, baseline):
    """LiteGraph.registered_node_types contains all 7 TS_CosyVoice3 classes."""
    registered = browser_page.evaluate(
        "() => Object.keys(window.LiteGraph.registered_node_types || {})"
        ".filter(k => k.startsWith('TS_CosyVoice3_')).sort()"
    )
    expected = sorted(n["node_id"] for n in baseline["nodes"])
    assert registered == expected, (
        f"LiteGraph registered nodes don't match baseline.\n"
        f"  expected: {expected}\n"
        f"  actual:   {registered}"
    )


@pytest.mark.parametrize(
    "expected",
    json.load(open(BASELINE_PATH, "r", encoding="utf-8"))["nodes"],
    ids=lambda e: e["node_id"],
)
def test_litegraph_node_runtime_contract(browser_page, expected):
    """For each TS node, the LiteGraph runtime class exposes the expected category and outputs."""
    node_id = expected["node_id"]
    payload = browser_page.evaluate(
        """(nodeId) => {
            const ctor = window.LiteGraph.registered_node_types[nodeId];
            if (!ctor) return null;
            const inst = new ctor();
            const outputs = (inst.outputs || []).map(o => ({type: o.type, name: o.name}));
            const inputs = (inst.inputs || []).map(i => ({type: i.type, name: i.name}));
            return {
                category: ctor.category || inst.category,
                title: ctor.title || inst.title,
                outputs,
                inputs,
            };
        }""",
        node_id,
    )
    assert payload is not None, f"LiteGraph constructor for {node_id} not found"

    assert payload["category"] == expected["category"], (
        f"{node_id}: category drift — baseline={expected['category']!r}, "
        f"runtime={payload['category']!r}"
    )

    expected_output_types = [o["type"] for o in expected["outputs"]]
    actual_output_types = [o["type"] for o in payload["outputs"]]
    assert actual_output_types == expected_output_types, (
        f"{node_id}: outputs drift — baseline={expected_output_types}, runtime={actual_output_types}"
    )

    # Required inputs from baseline must be present as connectable slots in LiteGraph.
    # ComfyUI only renders connection-typed inputs as slots; widget-only inputs
    # (numbers, strings, booleans, combos) live on inst.widgets, so we check both.
    runtime_input_names = {i["name"] for i in payload["inputs"] if i.get("name")}
    runtime_input_payload = browser_page.evaluate(
        """(nodeId) => {
            const ctor = window.LiteGraph.registered_node_types[nodeId];
            const inst = new ctor();
            const widgets = (inst.widgets || []).map(w => w.name);
            return widgets;
        }""",
        node_id,
    )
    runtime_widget_names = set(runtime_input_payload or [])
    all_runtime_inputs = runtime_input_names | runtime_widget_names

    for required_name in expected["inputs"].get("required", {}):
        assert required_name in all_runtime_inputs, (
            f"{node_id}: required input {required_name!r} not visible in LiteGraph "
            f"(slots={runtime_input_names}, widgets={runtime_widget_names})"
        )


def test_can_instantiate_loader_on_canvas(browser_page):
    """Instantiate the V3 ModelLoader on the live canvas to verify end-to-end wiring."""
    result = browser_page.evaluate(
        """() => {
            const node = LiteGraph.createNode('TS_CosyVoice3_ModelLoader');
            if (!node) return {ok: false, reason: 'createNode returned null'};
            window.app.graph.add(node);
            const widgetNames = (node.widgets || []).map(w => w.name);
            const outputs = (node.outputs || []).map(o => o.type);
            return {ok: true, widgets: widgetNames, outputs};
        }"""
    )
    assert result.get("ok"), f"Could not instantiate ModelLoader on canvas: {result}"
    assert "model_version" in result["widgets"], (
        f"ModelLoader widgets missing 'model_version': {result['widgets']}"
    )
    assert "fp16" in result["widgets"], (
        f"ModelLoader widgets missing 'fp16': {result['widgets']}"
    )
    assert result["outputs"] == ["COSYVOICE_MODEL"], (
        f"ModelLoader outputs drift: {result['outputs']}"
    )


def test_screenshot_with_ts_node_visible(browser_page):
    """Take a screenshot showing a TS node on the canvas. Saved for human review."""
    # Make sure at least one TS node is on canvas (idempotent — this also clears noise).
    browser_page.evaluate(
        """() => {
            window.app.graph.clear();
            const node = LiteGraph.createNode('TS_CosyVoice3_ModelLoader');
            node.pos = [200, 200];
            window.app.graph.add(node);
            const synth = LiteGraph.createNode('TS_CosyVoice3_Instruct2');
            synth.pos = [600, 200];
            window.app.graph.add(synth);
            window.app.graph.setDirtyCanvas(true, true);
        }"""
    )
    # Give the canvas one redraw tick.
    browser_page.wait_for_timeout(500)

    artifact = SCREENSHOT_DIR / "comfyui_with_ts_nodes.png"
    browser_page.screenshot(path=str(artifact), full_page=False)
    assert artifact.exists() and artifact.stat().st_size > 0, "Screenshot was not saved"
