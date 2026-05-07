# ADR 0001: Migrate to ComfyUI V3 schema

- **Status:** Accepted
- **Date:** 2026-05-07

## Context

`comfyui-ts-cosyvoice` v0.6.0 currently registers all 7 nodes via the ComfyUI **V1 schema**:
`INPUT_TYPES` + `RETURN_TYPES` + `FUNCTION` + `CATEGORY`, exported through the
`NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` dicts in `__init__.py`.

ComfyUI now ships **V3 schema** (`IO.ComfyNode` + `define_schema` + `execute`), exposed via
`comfy_api.v0_0_2`. V3 brings:

- typed inputs/outputs with explicit `IO.<Type>.Input(...)` / `IO.<Type>.Output(...)`,
- structured `IO.NodeOutput`,
- `search_aliases` for renames,
- `NodeReplace` for migration of historically renamed nodes,
- structured `validate_inputs` / `fingerprint_inputs` lifecycle,
- official `ComfyExtension` registration via `comfy_entrypoint()`.

Staying on V1 is supported but means losing typed-IO ergonomics, structured aliases,
and official discoverability for new ComfyUI features.

## Decision

Migrate all **7 nodes** of `comfyui-ts-cosyvoice` to V3 incrementally, with hybrid
registration during the transition (V1 + V3 in `__init__.py` simultaneously, one node
per commit).

Constraints:

- Pin imports to **`comfy_api.v0_0_2`**, never `comfy_api.latest`. The `latest` alias
  is intentionally unstable and will break the pack on the next ComfyUI release.
- Preserve every public contract — `node_id` strings, input names/types/order,
  output types/order — so user workflow JSONs saved before the migration continue to
  load and execute identically.
- Custom IO type `COSYVOICE_MODEL` is declared **once** in
  `nodes/_v3_types.py` and imported by every node that consumes it.
- V3 classes are locked (`comfy_api.internal.lock_class`); class-level mutable state
  is not possible. The model cache (currently a module-level dict in
  `utils/ts_model_manager.py`) stays where it is — the loader node will read/write
  through that module rather than via class attributes.
- Vendored upstream code (`cosyvoice/`, `matcha/`) is untouched. The migration is a
  thin V3 wrapper around the same inference logic.

## Consequences

- Minimum required ComfyUI version rises to one that ships stable `comfy_api.v0_0_2`.
  `pyproject.toml` gets a `requires-comfyui` floor.
- `__init__.py` gains a `_PackExtension` `ComfyExtension` and a `comfy_entrypoint()`
  factory, alongside the legacy `NODE_CLASS_MAPPINGS`. By Phase 4 the legacy mappings
  are removed entirely.
- `CLAUDE.md` is updated twice: once to lift the "no V1→V3 migration without explicit
  request" rule and document V3 patterns alongside V1 (Phase 1, transitional), and
  again at the end to drop the V1-only sections (Phase 4).
- Pack version bumps from `0.6.0` to `0.7.0` (SemVer minor + `BREAKING:` note in
  `CHANGELOG.md`, since the schema change is breaking with respect to the
  `requires-comfyui` floor even though node IDs are preserved).
- Workflow JSON regression tests are not yet provided by the user. Static contract
  preservation is verified via `tests/test_node_contracts.py` against
  `tests/contracts/v1_baseline.json`. End-to-end workflow regression is left for the
  user to verify in their own ComfyUI before merging Phase 4.

## Alternatives considered

- **Stay on V1 indefinitely.** Rejected. ComfyUI's long-term direction is V3; new
  features will land on V3 first.
- **Big-bang rewrite of all 7 nodes in one commit.** Rejected. Inverts workflow
  compatibility — any single broken contract would break all user workflows
  simultaneously, with no easy revert.
- **Migrate only the loader, leave synthesis nodes on V1.** Rejected. The
  loader's only purpose is to feed the synthesis nodes, so a partial migration would
  not deliver any user-visible benefit and would double the maintenance burden.
- **Rename nodes during migration.** Rejected. Preserving `node_id` is the highest
  priority constraint; we use V3's `search_aliases=[]` to leave room for future
  renames without breaking JSONs.
