"""
Generate the pack's embedded help pages and locale tables.

Two inputs, one output set — so the shipped artefacts cannot drift from the code:

- ``define_schema()`` of every registered node supplies the structure: input ids,
  types, defaults, ranges, combo options, and the Russian tooltips.
- ``tools/ts_node_docs_source.py`` supplies the English tooltips and the
  long-form prose that the schema cannot express.

Written artefacts:

- ``web/docs/<node_id>/en.md`` and ``web/docs/<node_id>/ru.md`` — ComfyUI serves
  these as the node's embedded help (``WEB_DIRECTORY`` is declared in
  ``__init__.py``).
- ``locales/en/nodeDefs.json`` and ``locales/ru/nodeDefs.json`` — merged by
  ComfyUI into ``/api/i18n`` and applied by the frontend, keyed by ``node_id``.

Run from the pack root with the ComfyUI portable interpreter::

    "<comfy>/python/python.exe" tools/generate_node_docs.py

``tests/test_generated_docs.py`` calls ``build_artifacts()`` and fails if what is
committed differs from what this module would write, so regenerating is not
something a contributor can forget.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PACK_ROOT not in sys.path:
    sys.path.insert(0, PACK_ROOT)

try:
    # Imported as `tools.generate_node_docs` (test suite, pack root on sys.path).
    from .ts_node_docs_source import NODE_DOCS
except ImportError:
    # Run directly as `python tools/generate_node_docs.py`; the pack root is on
    # sys.path from above, so address the sibling through the package.
    from tools.ts_node_docs_source import NODE_DOCS  # type: ignore[no-redef]


# Combo widgets whose option list is long (emotion presets, languages) get an
# excerpt rather than a wall of values; the full list is visible in the widget.
_MAX_LISTED_OPTIONS = 6

_LABELS = {
    "en": {
        "inputs": "Inputs",
        "outputs": "Outputs",
        "notes": "Notes",
        "input": "Input",
        "output": "Output",
        "type": "Type",
        "required": "Required",
        "default": "Default",
        "description": "Description",
        "yes": "yes",
        "no": "no",
        "options": "Options",
        "range": "Range",
        "more": "and {n} more",
        "generated": (
            "This page is generated from the node schema and "
            "`tools/ts_node_docs_source.py`. Do not edit it by hand."
        ),
    },
    "ru": {
        "inputs": "Входы",
        "outputs": "Выходы",
        "notes": "Примечания",
        "input": "Вход",
        "output": "Выход",
        "type": "Тип",
        "required": "Обязательный",
        "default": "По умолчанию",
        "description": "Описание",
        "yes": "да",
        "no": "нет",
        "options": "Варианты",
        "range": "Диапазон",
        "more": "и ещё {n}",
        "generated": (
            "Страница генерируется из схемы ноды и "
            "`tools/ts_node_docs_source.py`. Не редактируйте её вручную."
        ),
    },
}


def load_schemas() -> list[Any]:
    """Return every registered node schema, in registration order."""
    import importlib

    pack = importlib.import_module("__init__")
    extension = asyncio.run(pack.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    return [node.define_schema() for node in nodes]


def _escape_cell(text: str) -> str:
    """Make a string safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_default(spec: Any) -> str:
    default = getattr(spec, "default", None)
    if default is None:
        return "—"
    if isinstance(default, bool):
        return "`true`" if default else "`false`"
    if isinstance(default, str):
        return f"`{default}`" if default else "—"
    return f"`{default}`"


def _format_constraints(spec: Any, labels: dict[str, str]) -> str:
    """Describe a widget's accepted values: combo options or a numeric range."""
    options = getattr(spec, "options", None)
    if options:
        shown = [str(option) for option in options[:_MAX_LISTED_OPTIONS]]
        rendered = ", ".join(f"`{option}`" for option in shown)
        if len(options) > _MAX_LISTED_OPTIONS:
            rendered += ", " + labels["more"].format(n=len(options) - _MAX_LISTED_OPTIONS)
        return f"{labels['options']}: {rendered}."

    minimum = getattr(spec, "min", None)
    maximum = getattr(spec, "max", None)
    if minimum is None and maximum is None:
        return ""
    step = getattr(spec, "step", None)
    rendered = f"{labels['range']}: `{minimum}`–`{maximum}`"
    if step is not None:
        rendered += f", step `{step}`"
    return rendered + "."


def _validate_against_schema(schema: Any, doc: dict[str, Any]) -> None:
    """Fail loudly when the prose source and the schema have drifted apart."""
    node_id = schema.node_id

    schema_inputs = {spec.id for spec in schema.inputs}
    documented_inputs = set(doc["inputs"])
    missing = schema_inputs - documented_inputs
    extra = documented_inputs - schema_inputs
    if missing:
        raise ValueError(f"{node_id}: inputs missing from ts_node_docs_source: {sorted(missing)}")
    if extra:
        raise ValueError(f"{node_id}: ts_node_docs_source documents unknown inputs: {sorted(extra)}")

    schema_outputs = {out.display_name for out in schema.outputs}
    documented_outputs = set(doc["outputs"])
    missing_out = schema_outputs - documented_outputs
    extra_out = documented_outputs - schema_outputs
    if missing_out:
        raise ValueError(f"{node_id}: outputs missing from ts_node_docs_source: {sorted(missing_out)}")
    if extra_out:
        raise ValueError(f"{node_id}: ts_node_docs_source documents unknown outputs: {sorted(extra_out)}")

    for spec in schema.inputs:
        if not (spec.tooltip or "").strip():
            raise ValueError(f"{node_id}: input {spec.id!r} has no tooltip in define_schema()")
    if not (schema.description or "").strip():
        raise ValueError(f"{node_id}: schema has no description")

    for required_key in ("description_ru", "intro_en", "intro_ru", "notes_en", "notes_ru"):
        if not doc.get(required_key):
            raise ValueError(f"{node_id}: ts_node_docs_source is missing {required_key!r}")


def _tooltip_for(spec: Any, doc: dict[str, Any], lang: str) -> str:
    """The Russian tooltip lives in the schema; the English one in the prose source."""
    if lang == "ru":
        return (spec.tooltip or "").strip()
    return doc["inputs"][spec.id].strip()


def _render_markdown(schema: Any, doc: dict[str, Any], lang: str) -> str:
    labels = _LABELS[lang]
    lines: list[str] = [f"# {schema.display_name}", ""]
    lines.append(doc[f"intro_{lang}"].strip())
    lines.append("")

    lines.append(f"## {labels['inputs']}")
    lines.append("")
    lines.append(
        f"| {labels['input']} | {labels['type']} | {labels['required']} | "
        f"{labels['default']} | {labels['description']} |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for spec in schema.inputs:
        description = _tooltip_for(spec, doc, lang)
        constraints = _format_constraints(spec, labels)
        if constraints:
            description = f"{description} {constraints}"
        required = labels["no"] if spec.optional else labels["yes"]
        lines.append(
            f"| `{spec.id}` | {spec.io_type} | {required} | "
            f"{_format_default(spec)} | {_escape_cell(description)} |"
        )
    lines.append("")

    lines.append(f"## {labels['outputs']}")
    lines.append("")
    lines.append(f"| {labels['output']} | {labels['type']} | {labels['description']} |")
    lines.append("| --- | --- | --- |")
    for out in schema.outputs:
        text = doc["outputs"][out.display_name][lang]
        lines.append(f"| `{out.display_name}` | {out.io_type} | {_escape_cell(text)} |")
    lines.append("")

    notes = doc[f"notes_{lang}"]
    if notes:
        lines.append(f"## {labels['notes']}")
        lines.append("")
        for note in notes:
            lines.append(f"- {note.strip()}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"<sub>{labels['generated']}</sub>")
    lines.append("")
    return "\n".join(lines)


def _render_node_defs(schemas: list[Any], lang: str) -> dict[str, Any]:
    """Build the locale table ComfyUI merges into /api/i18n, keyed by node_id."""
    node_defs: dict[str, Any] = {}
    for schema in schemas:
        doc = NODE_DOCS[schema.node_id]
        entry: dict[str, Any] = {
            # This string is the node-library subtitle and the search-result line,
            # so both locales use their one-line description — the English one
            # from the schema, the Russian one from the prose source. Using the
            # long intro paragraph here (as an earlier version did for Russian)
            # made the Russian entry three times longer than the English one.
            "description": (
                schema.description.strip() if lang == "en" else doc["description_ru"].strip()
            ),
            "inputs": {
                spec.id: {"tooltip": _tooltip_for(spec, doc, lang)}
                for spec in schema.inputs
            },
            "outputs": {
                str(index): {"tooltip": doc["outputs"][out.display_name][lang].strip()}
                for index, out in enumerate(schema.outputs)
            },
        }
        node_defs[schema.node_id] = entry
    return node_defs


def build_artifacts(schemas: list[Any] | None = None) -> dict[str, str]:
    """
    Render every generated file as ``{repo-relative path: file contents}``.

    Writing is left to ``main()`` so the test suite can diff what *would* be
    written against what is committed, without touching the working tree.
    """
    if schemas is None:
        schemas = load_schemas()

    documented = set(NODE_DOCS)
    registered = {schema.node_id for schema in schemas}
    if documented != registered:
        raise ValueError(
            "ts_node_docs_source and the registered node list disagree: "
            f"undocumented={sorted(registered - documented)}, "
            f"unknown={sorted(documented - registered)}"
        )

    artifacts: dict[str, str] = {}
    for schema in schemas:
        doc = NODE_DOCS[schema.node_id]
        _validate_against_schema(schema, doc)
        for lang in ("en", "ru"):
            path = f"web/docs/{schema.node_id}/{lang}.md"
            artifacts[path] = _render_markdown(schema, doc, lang)

    for lang in ("en", "ru"):
        payload = _render_node_defs(schemas, lang)
        artifacts[f"locales/{lang}/nodeDefs.json"] = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )

    return artifacts


def main() -> int:
    artifacts = build_artifacts()
    for relative_path, content in sorted(artifacts.items()):
        absolute_path = os.path.join(PACK_ROOT, relative_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        with open(absolute_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        print(f"wrote {relative_path}")
    print(f"{len(artifacts)} file(s) generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
