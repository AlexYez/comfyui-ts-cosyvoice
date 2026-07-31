# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-07-31

Minor release: acts on a ComfyUI node-pack audit and on a follow-up read-only
code audit. No node was renamed, no input or output was added, removed or
reordered, and the speaker-preset and `COSYVOICE_MODEL` formats are unchanged —
saved workflows and existing `.pt` presets keep working. The contract test now
verifies that positionally, including input order, defaults and ranges.

### Added
- **Embedded help.** `WEB_DIRECTORY` is declared and every node ships a help page
  at `web/docs/<node_id>/{en,ru}.md`; the per-node help button previously 404'd.
- **Localisation.** `locales/{en,ru}/nodeDefs.json` carry descriptions, input and
  output tooltips. Russian tooltips stay in the schema as the fallback; English
  now reaches an English UI instead of showing Cyrillic.
- **Example workflow discoverability.** `workflows/` is now `example_workflows/`
  — the folder name ComfyUI prefers — with the same-named `.jpg` thumbnail the
  template browser shows on its card.
- `essentials_category = "Audio"` on all seven nodes (additive; nothing moves in
  the normal category menu).
- `validate_inputs` on Save Speaker rejects an unusable preset name during graph
  validation instead of after the model loader has already run.

### Fixed
- **Cancel works.** No loop read ComfyUI's interrupt flag, so cancelling a long
  voice conversion did nothing and the node ran to completion. Checks now sit in
  the streaming loop, the voice-conversion chunk loop and the dialog line loop.
- **`seed = -1` means what its tooltip says.** With unchanged inputs ComfyUI
  served the cached audio forever, so `-1` produced one take and then never
  varied. The seeded nodes now fingerprint their inputs.
- **Speaker Text To Voice notices a re-saved preset** instead of returning the
  voice from before the re-save.
- The README version badge, which had been stuck on `0.7.0`.
- The example workflow no longer ships the deprecated `SaveAudioMP3`, and its
  `speaker_preset` points at the preset its own Save Speaker node writes.

### Changed
- **Model integrity checks are recorded, not repeated.** CRC-ing the checkpoints
  and opening both ONNX models used to run on the first load after every ComfyUI
  start; the result is now recorded next to the weights and re-verified only when
  a file changes.
- **Whisper transcripts are cached by audio content**, so the dialog node stops
  re-transcribing unchanged speaker references on every run.
- The model cache takes a lock per cache key rather than one global lock held for
  the duration of a multi-gigabyte download.
- A failed auto-transcription is now logged as an explicit warning naming the
  cause, and the Save Speaker help says where to look for it.

### Internal
- `ruff.toml` (vendored code excluded) and everything it flagged.
- An API-pin canary so a future `comfy_api` divergence is a decision, not drift.
- Guard tests tying the shipped documentation to actual behaviour, added after an
  audit found help pages describing the pre-fix `seed` semantics.
- Browser-driven round-trip tests proving `widgets_values` stays positionally
  stable for every node.

## [0.9.1] - 2026-07-19

Patch release: quality-audit follow-up. No node was renamed, no input/output was
added, removed or reordered, and the speaker-preset and `COSYVOICE_MODEL` formats
are unchanged — existing workflows and saved `.pt` presets keep working.

### Fixed
- **Speaker preset device (regression from 0.9.0).** `TS CosyVoice Speaker Text To
  Voice` loaded the preset onto `model["device"]` — the *requested* device — which
  the vendored runtime does not honour (it always runs on cuda-if-available). If
  the loader was pinned to CPU on a CUDA box, preset tensors and model weights
  landed on different devices and inference failed. The preset now loads onto the
  device the model actually runs on (`frontend.device`), via a single
  `get_runtime_device` helper.
- **A malformed `configs/ts_emotion_presets.json` no longer breaks the pack.**
  Emotion-preset options are read at import time; a JSON syntax error used to
  propagate through node import and stop ComfyUI from registering any of the 7
  nodes. Parsing now degrades to the built-in custom-instruction option with a
  warning.
- **Model cache no longer stores duplicate copies in VRAM.** The cache key hashed
  the requested device string, so `auto` (→ `cuda:0`) and an explicit `cuda`
  loaded and cached the same weights twice; likewise `fp16` on a CPU-only box
  (silently downgraded to fp32) duplicated the fp32 model. The key now depends
  only on what changes the loaded weights: model version and the effective fp16
  flag.
- **Temp WAV files no longer leak when writing fails.** `NamedTemporaryFile`
  created the file before `soundfile.write`; a write error (full disk, bad array)
  left an empty file with no path returned to clean it up. The temp file is now
  removed on write failure.
- **Dialog temp files are cleaned up on partial failure.** If preparing speaker
  audio failed part-way through, temp files already written were leaked; the list
  is now caller-owned so the node's `finally` always sees them.
- **AUDIO with a batch of more than one clip is handled explicitly.** A `[B>1, C, T]`
  input previously reached `soundfile` as a 3-D array and failed with a cryptic
  error; the first clip is now used, with a debug log.
- Exception chains preserved (`raise ... from`) in the model loader and downloader.
- Cross-language synthesis no longer re-runs mono/resample on an already-prepared
  reference (behaviour identical, one redundant pass removed).

### Changed
- The Model Loader `device` widget tooltip now states that device selection is
  best-effort (the CosyVoice runtime places the model itself). `model["device"]`
  now reports the actual runtime device rather than the requested one.

### Internal
- New helpers: `get_runtime_device` (adapter), `_drop_batch_dim` / `_write_temp_wav`
  (audio utils).
- Test suite grown to 147 (+ the earlier 131): emotion-preset resilience,
  runtime-device resolution, model-cache keying and device recording, temp-file
  cleanup on write failure, and batch-dim handling.

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
