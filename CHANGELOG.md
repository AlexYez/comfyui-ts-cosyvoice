# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-07-15

Quality release: no node was renamed, no input/output was added, removed or
reordered, and the speaker-preset file format is unchanged — existing workflows
and saved `.pt` presets keep working as-is. The one behaviour change is on the
error path (see *Changed*).

### Security
- **Speaker presets can no longer escape the preset directory.** `speaker_name` in
  `TS CosyVoice Save Speaker` was joined into a path with no validation, so a name
  like `../../evil` wrote outside `models/cosyvoice/speaker/`. Names are now
  validated (no separators, no traversal, no Windows-reserved or illegal
  characters) and every resolved path is checked for containment. Unicode names
  (e.g. Cyrillic) remain valid.
- **Presets load without executing pickle code.** `torch.load` now runs with
  `weights_only=True`. Presets only ever contained tensors, so previously saved
  presets load unchanged.

### Fixed
- **Voice conversion no longer fails on sources longer than 24 s when librosa's
  silence detection is unusable.** On installs where librosa's numba backend
  rejects the installed NumPy, `librosa.effects.split` raised and took the whole
  conversion down. Silence detection is an optimisation for *where* to cut, so it
  now degrades to fixed-size chunks with a warning.
- **Speaker presets load onto the device the model was actually loaded on.** The
  node hardcoded `cuda if available`, which produced a device mismatch when the
  Model Loader had been pinned to CPU on a CUDA machine. It now follows
  `model["device"]`.
- **Saved speakers no longer wipe each other.** `TS CosyVoice Speaker Text To
  Voice` replaced the shared model frontend's whole `spk2info` table, dropping
  speakers registered by other nodes or earlier runs; entries are now merged in
  key-by-key and copied before prompt tokens are written.
- **Missing optional dependencies no longer prevent the pack from loading.**
  `soundfile` was imported at module level by a helper every synthesis node pulls
  in, so one absent library stopped ComfyUI from registering *any* of the 7 nodes.
  Heavy imports (`soundfile`, `torch`, `torchaudio`) moved into the functions that
  use them; a missing `soundfile` now raises an actionable error at call time.
- **Emotion preset edits apply without a ComfyUI restart.** The mtime-invalidated
  preset cache was defeated by a module-level snapshot; the effective instruction
  is now resolved per call.
- Chunk splitting picked the *last* usable silence rather than the one nearest the
  chunk limit (the computed best-candidate distance was never applied).
- The startup banner reported version 0.7.0 two releases after the fact; the
  version now has a single runtime source, enforced by a test.

### Changed
- **Synthesis nodes now report errors instead of returning one second of silence.**
  `Text to Voice`, `Cross-Language`, `Speaker Text To Voice`, `Voice To Voice` and
  `Dialog` caught every exception and returned silent audio, so a failed run looked
  like a successful one and the graph continued with empty audio. They now raise,
  matching what `Model Loader` and `Save Speaker` always did — a failure surfaces
  on the node in the UI. The graceful "no parseable dialog lines" path in `Dialog`
  still returns silence plus an explanatory `message`, as before.
- `Voice To Voice` and `Dialog` log through the project logger instead of `print()`.

### Internal
- New `utils/ts_speaker_io.py` centralises preset naming, validation, save/load and
  frontend injection.
- Removed dead helpers (`load_audio_from_path`, `ensure_stereo`, `normalize_audio`,
  `check_model_exists`) and the duplicate CosyVoice3 detector in
  `ts_whisper_utils`; `is_cosyvoice3_model_info` is now the single implementation.
- Test suite extended from 7 contract tests to 130+: preset I/O and traversal,
  audio shape/rate conventions, prompt formatting, VC chunking and crossfade,
  import-time resilience, and version consistency.

## [0.8.1] - 2026-06-09

### Fixed
- **Windows install:** replaced the declared `WeTextProcessing` dependency with
  `wetext` in `requirements.txt` and `pyproject.toml`. `WeTextProcessing` pulled
  in `pynini==2.1.6`, which builds from source and fails under MSVC on native
  Windows (GCC-only compiler flags such as `-Wno-register` / `-funsigned-char`
  passed to `cl.exe`). The vendored CosyVoice frontend
  (`cosyvoice/cli/frontend.py`) has always imported `wetext`, not
  `WeTextProcessing` — so the old dependency was simultaneously broken on Windows
  and never actually used (no `tn` / `itn` import anywhere in the pack). `wetext`
  does not depend on `pynini`; its only compiled dependency, `kaldifst`, ships
  prebuilt Windows wheels (cp310–cp313 `win_amd64`), so the pack now installs
  cleanly on Windows portable ComfyUI without a C++ toolchain.

## [0.8.0] - 2026-05-13

### Changed
- Project license switched to **MIT** (standalone pure `LICENSE` file so GitHub
  detects it correctly).
- Dev/runtime split: development-only files (`tests/`, `project_memory/`,
  `pytest.ini`, `CLAUDE.md`) are kept out of the published pack and live only in
  the development checkout.

### Added
- `.comfyignore` to exclude dev-only files from the ComfyRegistry distribution.

## [0.7.0] - 2026-05-07

### Changed
- **BREAKING (schema):** all 7 nodes migrated from ComfyUI **V1 schema**
  (`INPUT_TYPES` + `RETURN_TYPES` + `FUNCTION` + `CATEGORY`) to **V3 schema**
  (`IO.ComfyNode` + `define_schema` + `execute` + `IO.NodeOutput`),
  pinned to `comfy_api.v0_0_2`. Public contracts are preserved:
  - All 7 `node_id` strings unchanged (`TS_CosyVoice3_ModelLoader`,
    `TS_CosyVoice3_Instruct2`, `TS_CosyVoice3_SpeakerInstruct2`,
    `TS_CosyVoice3_CrossLingual`, `TS_CosyVoice3_VoiceConversion`,
    `TS_CosyVoice3_SaveSpeaker`, `TS_CosyVoice3_Dialog`).
  - Display names, categories, input names/types/order, output types/order
    and `is_output_node` flag preserved bit-for-bit against the V1 baseline.
  - Existing user workflow JSON files continue to load and execute identically.
- Minimum required ComfyUI version bumped to `>=0.3.40` (the floor for stable
  `comfy_api.v0_0_2`).
- Registration moves from `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`
  to a single `_PackExtension` registered through `comfy_entrypoint()`.
- Custom IO type `COSYVOICE_MODEL` is now declared once in
  `nodes/_v3_types.py` and imported by every consumer.

### Added
- `tests/contracts/v1_baseline.json` — frozen V1 contract snapshot used as the
  source of truth for migration verification.
- `tests/test_node_contracts.py` — pytest suite asserting that every node's
  V3 schema preserves `node_id`, display name, category, `is_output_node`,
  input names, output types and order from the V1 baseline.
- `nodes/_v3_types.py` — single declaration site for `IO.Custom("COSYVOICE_MODEL")`.

### Removed
- Legacy `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` dicts in
  `__init__.py`. Use the V3 `comfy_entrypoint()` instead — ComfyUI registers
  the pack automatically.

### Notes
- No inference logic was modified during the migration. Every `execute`
  classmethod runs the exact same code path as the previous V1 `FUNCTION`
  method. Behavioral parity is asserted by the contract tests; end-to-end
  audio output equality must be re-verified by users in their own ComfyUI
  before relying on the new release.
- Vendored upstream code (`cosyvoice/`, `matcha/`) was not touched.

## [0.6.0] - prior to migration window

Last V1-schema release. See git history for the per-feature changes that
shipped in this and earlier 0.x releases.
