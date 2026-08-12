# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1] - 2026-08-10

### Fixed
- **A half-removed ONNX Runtime was let through as if it were fine.**
  `require_onnxruntime()` exists so a missing or broken ONNX Runtime produces a
  pack-level error naming the package to install, instead of a failure from inside
  vendored code. It only checked that `import onnxruntime` succeeded — which is not
  enough, and the gap was hit in practice.

  `onnxruntime` and `onnxruntime-gpu` are separate distributions that install into the
  *same* `site-packages/onnxruntime/` directory, so uninstalling either deletes files
  the other needs, `__init__.py` included. pip leaves the directory behind, Python
  imports it as a namespace package, and the import succeeds with `__file__` set to
  `None`. The first real call then raised

      module 'onnxruntime' has no attribute 'SessionOptions'

  from `cosyvoice/cli/frontend.py` — exactly the obscure failure the check was added
  to prevent. Made worse when the uninstall runs while ComfyUI still holds the DLLs
  open: pip cannot delete a locked file, renames it to `~ackend` and reports success.

  The check now verifies that `SessionOptions`, `InferenceSession` and
  `get_available_providers` are actually present, and the error names what is missing,
  reports the module path (calling out `None` as a half-removed install), explains why
  a successful import proves nothing, and gives the repair — including that ComfyUI
  must be fully stopped first, or the reinstall is partial again.

Behaviour of the nodes is unchanged; this is diagnostics only.

## [1.4.0] - 2026-08-10

A second external audit, focused on what happens when things go wrong. Every finding
was reproduced before being changed; two turned out to be narrower than reported and
are noted as such.

### Fixed — verification no longer opens on failure
The trust boundary had three ways to let unverified data through, and the third was
the worst:

- **A missing or malformed `model_manifest.json` became an empty dict and a warning**,
  after which loading continued. It is part of the pack, so its absence means a damaged
  or tampered install — now a refusal, with the reason. Same for a manifest that has no
  entry for the model being loaded, or an entry with no `config` files: "nothing to
  check" is no longer reported as "checked".
- **A verified file that was simply absent was skipped.** An interrupted download, or
  two mirrors merged into one directory, produced exactly that shape and reached the
  loader with `cosyvoice3.yaml` unverified. Missing verified files are now refused, and
  the model directory is cleared before the fallback download source runs so the
  mixture cannot form in the first place.
- **A weight-hash mismatch was recorded as a passed deep check.** The warning appeared
  once and every later load reported "integrity already recorded" for files known not
  to match — strictly worse than the warn policy it was implementing. A load that could
  not be fully verified is no longer recorded, so the mismatch is reported every time.
- **PyTorch older than 2.6 is refused.** The vendored loader calls `torch.load` without
  `weights_only` on downloaded checkpoints, and before 2.6 that runs the pickle
  machinery. Enforced as a precondition rather than by patching the vendored tree.
  Verified that the torch in use here (2.11) does default to `weights_only=True`.
- **`spk2info.pt` is checked before the vendored frontend reads it.** It is the one
  `.pt` on the load path the manifest cannot cover — upstream does not publish it and
  the pack creates an empty one — so it is parsed with `weights_only=True` first. The
  documentation claiming hashes for "every file this package reads" has been corrected
  to say so.

### Fixed — publishing is gated on validation
The publish workflow triggered on a `pyproject.toml` push and called the publish action
immediately, while the install check ran as a separate workflow with no ordering
between them. A release that could not be installed could reach the registry before the
parallel check noticed, and a registry version cannot be withdrawn. The install check is
now a reusable workflow and `publish-node` has `needs: validate`.

### Fixed — a batch of audio is no longer silently discarded
`AUDIO` is `[B, C, T]`, and anything with `B > 1` was reduced to `waveform[0]` — logged
at DEBUG in the shared helper and not logged at all in voice conversion. A five-clip
batch produced one clip and reported success. It is now refused with an error naming the
count and what to do instead; `B == 1`, which is every graph that was producing correct
results, is unaffected. Two local tests that pinned the old behaviour as expected have
been rewritten to state why the expectation changed, and the limit is documented in the
README and in each affected node's help page.

### Fixed — one bad preset entry no longer unregisters the whole pack
`load_emotion_presets` checked that `presets` was a list but then called `.get` on each
item. `{"presets": [null]}` raised AttributeError, and because the widget options are
built at *import* time the error propagated out of `__init__.py` — so a single stray
`null` in a user-edited JSON file meant ComfyUI registered none of the seven nodes.
Non-object entries are now skipped with one summary warning.

### Fixed — saving a speaker preset is atomic
`torch.save` wrote straight to the final path, so re-saving an existing name and then
losing the process — a crash, a full disk, an exception inside `torch.save` — replaced a
working preset with a truncated one. It now writes to a temporary file in the same
directory, fsyncs, and `os.replace`s. Fault-injection tests cover the failed overwrite,
the leftover temporary file, and that the destination is never unlinked first.

### Performance
- **Voice conversion no longer re-copies the whole result at every seam.** Each chunk
  was joined onto one growing tensor, and `crossfade_join` allocates a fresh tensor and
  copies both sides, so total copying grew as O(N²) on exactly the long sources the
  chunking exists to support. `BoundedSolaJoiner` keeps only the tail that can affect
  the next splice — `max_overlap + search + fade` samples — and concatenates once,
  reducing total copying roughly 25-fold at 100 chunks. Measured on the shipped
  settings (24 s chunks, 1 s overlap): 20 min of audio 362 ms → 136 ms, 40 min 824 ms →
  313 ms. The output is *bit-identical* to the previous implementation, which is
  asserted against it for constant, zero and mixed overlaps.
  The join loop also checks for Cancel now, which it could not before.
- **Dialog no longer holds ~7× the final audio.** Every turn was kept in a mix list and
  in the speaking stem, plus an equal-length silence shared by the other three stems,
  and then a full mix and four full-length stems were built while all of that was still
  alive — about 2.25 GiB of output waveforms for a one-hour dialog at 24 kHz, before the
  model. Only `(speaker, waveform)` is kept now, and the five outputs are allocated once
  and filled by offset.

### Notes on two findings
- The audit asked for weight-hash mismatches to be **fail-closed**. They remain a loud,
  now-repeated warning, which was a deliberate choice in 1.2.0: someone whose weights
  predate the pinned revision has a working install. What made that choice unsafe was
  the sidecar recording it as verified, which is fixed above; and with PyTorch ≥ 2.6
  enforced, a tampered checkpoint can no longer execute code. Say the word and it
  becomes a refusal.
- ModelScope still resolves to `master`. Its API publishes no immutable ref for this
  repository (`ModelRevisions: None`), so the hashes are the control there. Unchanged
  and already documented in the manifest.

## [1.3.0] - 2026-08-08

Supply-chain and shared-environment release, from an external audit. Nothing about
synthesis changed; what changed is what the pack installs into somebody else's
ComfyUI, and what it is willing to load and execute.

Every finding below was reproduced before it was fixed, and two turned out to be
worse or narrower than reported.

### Fixed — the pack no longer claims ownership of its host's runtime
- **`torch` and `torchaudio` are no longer declared as dependencies.** ComfyUI
  installs them as a matched pair built against a specific CUDA/ROCm/MPS runtime, and
  a `torch>=2.0.0` line lets pip decide that pair should be replaced — which is how
  adding a custom node turns a working CUDA install into a CPU one. ComfyUI cannot
  start without them, so there was nothing to guard against in the first place.
- **`onnxruntime` is no longer required either.** `onnxruntime` and `onnxruntime-gpu`
  are separate distributions that both provide the module named `onnxruntime`, and
  ONNX Runtime supports exactly one variant per environment. Requiring the CPU build
  installed it beside a user's GPU build, leaving which one wins undefined. It is now
  the extras `onnx-cpu` / `onnx-cuda`, and both manifests plus the README say which
  to pick.
- **The silent CPU fallback is now loud.** `cosyvoice/cli/frontend.py` asks for
  `CUDAExecutionProvider` whenever torch reports CUDA. ONNX Runtime does not treat a
  missing provider as an error — it falls back to the CPU and continues, so the speech
  tokenizer encodes every reference clip on the CPU of a GPU machine with only ORT's
  own warning to show for it. `utils/ts_onnx_providers.py` reports the providers it
  actually has, raises a pack-level error naming the package to install when ORT is
  absent, and warns once per process when CUDA was expected and is not there. Running
  this on the development machine immediately found exactly that case: torch with CUDA
  13.0 alongside CPU-only `onnxruntime` 1.26.0.
- `requires-python = ">=3.10"` is declared, and CI now tests that floor as well as
  Windows, which it did not cover before.
- **`modelscope` is optional.** HuggingFace is the default source and serves the same
  files under the same hashes, so ModelScope is a mirror rather than a second source
  of truth. Selecting it without the package gives an error that says so.

### Fixed — the model is pinned and its contents are verified
- **The model was fetched from a mutable branch and then executed.** Downloads used
  `revision='main'` / `revision='master'`, and the resulting `cosyvoice3.yaml` goes to
  HyperPyYAML, whose `!new:` and `!apply:` tags import modules and construct objects.
  The previous checks — file sizes, ZIP CRCs, "does ONNX open this" — establish that a
  download finished, not that it is the download that was vetted.
  HuggingFace is now pinned to commit `29e01c4e`, and `model_manifest.json` records
  sha256 for all 15 files the pack reads.
- **ModelScope cannot be pinned, and the manifest says so** instead of leaving a bare
  `master` looking like an oversight: its API reports `ModelRevisions: None` and
  `master` is the only branch. All 10 files both mirrors expose hashes for match
  byte-for-byte, so the manifest covers ModelScope too — the hashes are the control
  there, not the revision.
- Verification splits by consequence. Executed and parsed files (`cosyvoice3.yaml`,
  the tokenizer configs) are **refused** on mismatch — measured at 0.06 s, so it runs
  on every load. Weights and ONNX graphs **warn and continue**: 9.74 GB hashes in
  7.7 s, so it is recorded in the existing `.ts_validation.json` sidecar and not
  repeated, and someone whose copy predates the pinned revision keeps a working
  install rather than being locked out.
- A refusal deliberately deletes nothing. `get_model_path` treats a validation *error
  string* as grounds to wipe the model directory and re-download, so a hash mismatch
  is raised as `ModelIntegrityError` on a separate path — an integrity check must not
  delete gigabytes belonging to a user whose model is merely newer.
- Verified by tampering: appending `!name:os.system` to a copy of `cosyvoice3.yaml` is
  refused with the file named, and flipping a single newline is caught.

### Fixed — the pack was shadowing two ComfyUI core modules
- `__init__.py` prepended the pack root to `sys.path`, and the pack contains `utils/`
  and `nodes/` while ComfyUI's root contains `utils/` and `nodes.py`. Measured in
  ComfyUI's own interpreter, `import utils` and `import nodes` resolved to *ours*
  after the pack loaded. It went unnoticed only because ComfyUI imports both before
  custom nodes load; any node loading later and importing them fresh would have got
  the pack's. The root is now **appended**, which leaves `cosyvoice`/`matcha`
  resolvable — nothing else defines those — and the core names alone.
- Four intra-package imports written as `from utils.X import Y` depended on that
  shadowing and would have broken model loading outright once it was removed. They now
  use the relative-first pattern the rest of the package already used. Verified by
  loading the pack the way ComfyUI loads it: 7 nodes registered, both core modules
  intact, vendored runtime imports.

### Fixed — models are released from VRAM
- `_MODEL_CACHE` had no eviction, so every distinct precision or checkpoint
  combination kept its own multi-gigabyte copy for the process lifetime. One CosyVoice
  model is now kept resident: the previous one is released *before* the next loads, so
  the peak is one model rather than two. `unload_cosyvoice_model()` frees explicitly —
  measured dropping VRAM from 3.44 GB to 0.01 GB.
- The cached dict is **not** cleared on eviction. An earlier version of this change did
  clear it, and the existing test suite caught what that means: it is the same object
  that flows between nodes as `COSYVOICE_MODEL`, so emptying it turns a downstream
  synthesis node's model into `{}` mid-run.

### Added — third-party licence notices, required before further distribution
- The repository shipped only its own MIT licence while redistributing Apache-2.0 and
  MIT code belonging to others. Added `THIRD_PARTY_NOTICES.md` plus full texts in
  `licenses/`: Apache-2.0 for `cosyvoice/` and the original MIT with its copyright
  line for `matcha/`. The notices list all twelve copyright holders found in the
  vendored sources — Alibaba, Antgroup, Mobvoi, Johns Hopkins and Shigeki Karita, the
  ESPnet/WeNet lineage the transformer code descends from.
- The exact upstream commits are **not recorded**, and the notices say so rather than
  printing an unverified hash: both trees arrived in one commit with no revision noted.
  What can be shown is that they are unmodified — `git diff` against that import is
  empty and no `TS-PATCH` marker exists — and a test keeps that claim honest.

### Security hygiene
- All GitHub Actions are pinned to full commit SHAs. `Comfy-Org/publish-node-action@main`
  was a moving branch running with `REGISTRY_ACCESS_TOKEN`.
- Log and error strings no longer use an em-dash, which renders as `?` in a
  legacy-codepage Windows console. Tooltips keep theirs — those are delivered to the
  browser as UTF-8.

### Notes
- `diffusers`, `hydra-core` and `pydantic` are declared but appear nowhere in the
  import graph (`diffusers` only in comments). They are left in place: removing a
  dependency can only be validated by exercising all seven nodes, and only voice
  conversion was run.

## [1.2.0] - 2026-08-08

Two things at once: audio quality in voice conversion, and the pack becoming
installable again on a current ComfyUI. The second matters more — as of 1.0.0 the
pack could not load a model at all on ComfyUI's own NumPy 2.5, and the fix was to
stop needing numba rather than to hold somebody else's NumPy back.

Backwards compatible under the 1.0.0 promise: no node, no existing input and no
output changed. Five optional
inputs are **appended** to the end of their nodes' widget lists — `widgets_values`
is positional and applied from index 0, so graphs saved with 1.0.0 keep mapping
correctly. The contract baseline was regenerated deliberately and still asserts
that every pre-existing input kept its name, order, type and default.

### Fixed
- **Chunk seams no longer drop out.** The join between converted chunks faded one
  chunk out to zero, faded the next in from zero and concatenated them. Measured
  on a constant-amplitude signal it reached exactly `0.0` at every boundary — a
  40 ms dropout roughly every 24 seconds of converted audio. It is now a real
  overlap-add crossfade that holds unity gain across the seam.
- **Silence-aware splitting works again.** It called `librosa.effects.split`,
  whose numba backend refuses to load against modern NumPy, so the whole path was
  dead and every cut landed at a fixed offset — frequently mid-word. Reimplemented
  as framed RMS in torch, with no optional dependency to fail.
- **Pitch shifting no longer runs at 16 kHz.** It was applied *after* the source
  was reduced for the tokenizer, which is the worst place for it. It now runs at up
  to the model's own rate, before that reduction.

### Added
- **Overlapping chunks with aligned joins.** Each chunk after the first re-reads
  one second of its predecessor, so timbre and prosody no longer restart at every
  boundary. Because a model's output length is not an exact function of its input
  length, the overlap is *measured* by normalised cross-correlation (SOLA) rather
  than assumed, then crossfaded — the approach long-form and streaming voice
  conversion implementations use.
- **Formant-preserving pitch shift** via WORLD F0 rescaling, so ±12 semitones no
  longer also makes the speaker sound smaller or larger. Verified to track the
  requested shift within 0.4% across the full range. `pyworld` was already a
  declared dependency but unused at inference time; a phase vocoder remains the
  fallback.
- `diffusion_steps` on voice conversion. The flow decoder ran a hardcoded 10 Euler
  steps; 25–40 resolves more detail at proportional cost. Implemented by wrapping
  the decoder at runtime rather than patching vendored code, and scoped to the one
  call — these settings live on the shared cached model and would otherwise change
  every other node in the graph.
- `guidance_strength` on voice conversion, exposing the classifier-free guidance
  rate that was fixed at 0.7 in the model configuration.
- `normalize_output` and `target_rms_dbfs` on voice conversion, so different
  reference clips stop producing wildly different output levels. This is RMS
  normalisation with a -1 dBFS peak ceiling and is labelled as such — it is not an
  EBU R128 loudness measurement.
- `llm_checkpoint` on the loader, selecting the GRPO reinforcement-learning LLM
  that ships as `llm.rl.pt` inside the same model folder. There is no separate
  `..._RL` repository (ModelScope 404, HuggingFace 401), so this loads the
  checkpoint over the default after construction; the choice is part of the cache
  key, so switching reloads rather than silently reusing the other weights.

### Fixed after a follow-up audit
- **A newer NumPy took the whole pack down, with no hint of why.** The vendored
  speech tokenizer imports `openai-whisper` at module level, whisper imports numba,
  and numba refuses to load against a NumPy it does not support — so model loading
  failed in *every* node with a traceback naming only numba. Reproduced on
  ComfyUI 0.31.1 with NumPy 2.5.1 and numba 0.61.0. The first fix was to declare
  the ceiling numba implies (`numpy<2.2`); that was **replaced before release** by
  removing the pack's need for numba altogether — see "The pack no longer needs
  numba" below. Pinning NumPy down would have downgraded a library the rest of the
  user's ComfyUI depends on.
- **Chunk alignment could shift timing by half a second.** The SOLA search radius
  was derived as half the overlap — ±500 ms for a 1 s overlap — while the matching
  window came from the crossfade and was only 20 ms, under two pitch periods of a
  low voice. For quasi-periodic speech a wide search does not find a better answer,
  it admits more wrong ones. Measured on a sustained vowel with two independent
  generations, worst-case misalignment was **554 ms**; with the search and window
  now set as absolute durations (80 ms each, sized from the model's own 40 ms token
  grid) it is **154 ms**, and 71 ms when ambiguity is the only error source.
  `tests/test_sola_production_config.py` checks the shipped values — the previous
  alignment tests used a wider window and a fifth of the search, so the
  configuration that actually shipped was never checked.
- **The last pitch-shift fallback was unguarded.** With pyworld absent and
  torchaudio failing, the librosa branch raised a raw
  `ImportError: Numba needs NumPy 2.1 or less` from inside librosa. It now raises
  one error naming all three failed shifters and how to proceed.
- `apply_rl_llm_checkpoint` loads with `weights_only=True`, per the pack's own
  policy for downloaded `.pt` files.
- The flow-decoder wrapper is no longer installed when the requested step count
  equals the vendored default, which was every run at the node's default.
- `_join_converted_chunks` rejects a mismatched overlap list instead of silently
  falling back to a plain crossfade, which would have left the overlap duplicated
  in the output.
- Removed the README badge claiming "29 tests passing": the suite is dev-local, so
  nothing in the published repository can substantiate a count, and the link
  pointed at a gitignored folder.

### macOS and cross-platform
A dedicated compatibility audit found that the pack ran on macOS but promised
things it could not deliver there, and that one dependency could block the install
outright.

- **`pyworld` is no longer a hard dependency.** It publishes wheels for Windows
  only (0.3.5: `win_amd64`/`win32` plus an sdist), so requiring it made
  `pip install -r requirements.txt` build a C++ extension — failing outright on a
  Mac without the Xcode Command Line Tools, before ComfyUI even started. It is now
  an optional extra (`world-pitch`); the pitch shift already fell back to a phase
  vocoder without it, and a test now pins that the fallback works and that the pack
  still registers all seven nodes.
- **The loader stops advertising acceleration it does not use.** The vendored
  runtime has no Metal/MPS path — every device decision in it is
  `cuda if available else cpu` — so on Apple Silicon everything runs on the CPU.
  Unsupported devices are now reduced to CPU with an explicit log line, instead of
  reporting "Auto-selected accelerator: mps" two lines before "Device: cpu". The
  `mps` option stays selectable: removing a combo value would break saved
  workflows. Tooltips, help pages and README say what actually happens.
- **The RL checkpoint loads onto the device the model really uses.** It was handed
  the *requested* device, so on a Mac it staged a multi-hundred-megabyte checkpoint
  through MPS only to copy it into CPU parameters — wasteful, and a real allocation
  risk in unified memory.
- **Silence detection allocates 1x the signal instead of 5x.** `unfold(...).pow(2)`
  materialised frames × frame_length floats: measured at 288 MB for a 10-minute
  source and 864 MB for 30 minutes, on top of the 1.5 GB model. Average pooling over
  the squared signal produces frame-for-frame identical values (verified to float32
  noise) from one copy.
- Preset listing matches the `.pt` extension case-insensitively, so a file named
  `Voice.PT` on a case-preserving filesystem is no longer invisible.
- `ffmpeg` is documented as what Whisper auto-transcription requires — macOS does
  not ship it, and without it a preset is saved with an empty reference text.
- New `.github/workflows/platform_install.yml` installs the requirements on
  macOS 14 (Apple Silicon) and Linux and smoke-checks the portable helpers. It does
  not run the suite — `tests/` is dev-local — but it would have caught the
  `pyworld` problem on its own.

### The pack no longer needs numba, and no longer constrains your NumPy
- **Fixed: the pack could not load a model on an up-to-date ComfyUI at all.**
  Current ComfyUI ships NumPy 2.5. numba refuses to import against NumPy 2.2 or
  newer, and the vendored tree reached for two numba-carrying packages at *module*
  level on the inference path: `cosyvoice/cli/frontend.py` imports `whisper` for
  `log_mel_spectrogram`, `cosyvoice/tokenizer/tokenizer.py` imports
  `whisper.tokenizer`, and the shipped `cosyvoice3.yaml` resolves
  `feat_extractor: !name:matcha.utils.audio.mel_spectrogram`, whose module does
  `from librosa.filters import mel`. Any one of those raised, so every node in the
  pack failed at model load with a traceback about NumPy.
- **Changed: the NumPy ceiling is gone.** The previous release pinned
  `numpy>=1.24.0,<2.2` to keep numba working. That was the wrong thing to
  constrain: this pack installs into somebody else's ComfyUI, and the pin does not
  quietly go unsatisfied — pip downgrades a library the rest of that install
  depends on. `numpy` is now floored only.
- **Added `utils/ts_mel.py`.** All three of those imports existed for mel
  spectrograms, which are ordinary arithmetic. This module computes librosa's
  `filters.mel` (slaney scale and normalisation) and ports whisper's
  `log_mel_spectrogram`, in torch and NumPy. It is not an approximation: it agrees
  with librosa's own implementation to within one float32 ULP at every parameter
  set the pack uses, including the shipped model's exact `feat_extractor`
  (`sr=24000, n_fft=1920, n_mels=80`). Where whisper is installed at all, its
  precomputed filterbank asset is read straight off disk so the speech tokenizer's
  input stays bit-identical to upstream.
- **Added `utils/ts_runtime_shims.py`.** It registers those stand-ins in
  `sys.modules` before the vendored import runs. **No vendored file is patched** —
  `cosyvoice/` and `matcha/` stay byte-identical to upstream, which is what keeps
  pulling upstream fixes cheap. The shims are installed *only* where the real
  package cannot be imported; where it can, nothing is touched and behaviour is
  exactly upstream's.
- **Changed: `openai-whisper` and `librosa` are now optional extras.** Keeping
  either as a hard requirement reimposes the NumPy cap transitively through numba.
  Nothing on the synthesis path uses them any more. What they still buy:
  `openai-whisper` (extra `transcription`) auto-fills `reference_text` in Save
  Speaker and Dialog; `librosa` is the last-resort pitch shifter behind `pyworld`
  and `torchaudio`. Both nodes already degraded gracefully without transcription
  and still do — now with a message that says which install is missing and that
  synthesis is unaffected, instead of an `AttributeError`.
- **Fixed: `scipy` and `matplotlib` were never declared, and one of them stopped
  being guaranteed.** A dependency audit that resolved every import in the tree
  against the declared manifests found both missing. `scipy` is imported by
  `cosyvoice/hifigan/generator.py` for the HiFT vocoder — it used to arrive
  transitively because librosa requires it, so making librosa optional would have
  left a fresh install unable to load a model at all. `matplotlib` is reached
  through `cosyvoice/hifigan/hifigan.py` → `matcha/hifigan/xutils.py` while the
  shipped `cosyvoice3.yaml` is parsed, and was missing already. Every other
  declared package lists both under an extra only, which pip does not install. A
  test now pins each with the reason.
- Verified end to end on the failing configuration: ComfyUI 0.31.1, Python 3.12.9,
  **NumPy 2.5.1**, with nothing installed or downgraded — model loads in ~22 s and
  a five-second voice conversion completes at RTF 0.52 with numba never imported.

### Internal
- New `utils/ts_audio_dsp.py` holds the signal-level helpers (silence detection,
  crossfade, SOLA alignment, RMS normalisation) as pure functions, tested against
  analytic expectations rather than "it ran".
- `tests/test_platform_compat.py` covers the paths that only differ off Windows:
  the pyworld-absent pitch shift, device normalisation, case-insensitive preset
  listing, and the absence of `subprocess`/absolute-path assumptions anywhere in
  `nodes/` or `utils/`.
- The chunking tests that changed meaning were updated, not relaxed: chunks now
  overlap, so they assert the chunks tile the source exactly once, and the join
  assertion now pins the absence of the dropout it used to permit.
- `tests/test_flow_overrides.py` pins that both quality overrides are restored on
  success, on exception and on cancellation.
- `tests/test_mel.py` checks the mel front-end against the implementations it
  replaces rather than against invariants. librosa cannot be *imported* in the
  environment this targets, so the reference is lifted out of librosa's own source
  with `ast` and executed against nothing but NumPy — a reference that only worked
  where librosa already works would never run where it matters. It also pins a
  defect found while writing it: asset lookup used `importlib.util.find_spec`,
  which consults `sys.modules` and therefore reported the pack's own whisper
  stand-in, silently skipping the genuine filterbank on disk in exactly the
  environment it was meant to serve.
- `tests/test_runtime_shims.py` is mostly about the shims doing *nothing*: a stub
  shadowing a working whisper would silently disable transcription, which is worse
  than the failure being fixed.
- `tests/test_platform_compat.py` had two tests that pinned the old NumPy cap.
  They are inverted rather than deleted — they now assert that neither manifest
  caps NumPy and that neither numba-carrying package is a hard requirement.
- The macOS/Linux CI job additionally asserts that the mel front-end works and
  that numba never enters the import graph, run in the same state a user's install
  is in now that neither package is required.

## [1.0.0] - 2026-07-31

Stable release. **No functional change from 0.10.0** — the code, the generated
help pages and the locale tables are identical. What changes is the promise:
the pack's public surface is now considered stable, and the items below will not
change without a major version and a documented migration.

Frozen from this release on:

- the seven `node_id` values, which are what a saved workflow stores;
- input **names**, order and defaults — `widgets_values` is a positional array,
  so inserting or reordering a serialising input rewrites every saved graph;
- output names, order and types;
- the `COSYVOICE_MODEL` dictionary passed from the loader to every other node;
- the on-disk speaker-preset format (`{name: model_input}` in a `.pt`, loaded
  with `weights_only=True`), including presets written by 0.6.x and later.

`tests/test_node_contracts.py` checks all of the above against
`tests/contracts/v1_baseline.json` on every run, and the browser round-trip
tests verify the positional guarantee through the real ComfyUI frontend.

Not covered by the stability promise: log wording, tooltips and help-page prose,
`essentials_category`, the contents of `configs/ts_emotion_presets.json`, and the
vendored `cosyvoice/` and `matcha/` trees, which track upstream.

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
