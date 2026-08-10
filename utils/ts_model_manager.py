"""
Model Manager for TS CosyVoice3
Handles model downloading, caching, and loading
"""

import json
import os
import shutil
import threading
import zipfile
from typing import Any, Dict

import torch

try:
    from .ts_cosyvoice_adapter import get_runtime_device
    from .ts_logging import get_logger, log_banner, log_exception
    from .ts_model_manifest import (
        pinned_revision,
        require_safe_torch_load,
        validate_tensors_only,
        verify_config_files,
        verify_weight_files,
    )
    from .ts_onnx_providers import report_provider_status, require_onnxruntime
except (ImportError, ValueError):
    from ts_cosyvoice_adapter import get_runtime_device
    from ts_logging import get_logger, log_banner, log_exception
    from ts_model_manifest import (
        pinned_revision,
        require_safe_torch_load,
        validate_tensors_only,
        verify_config_files,
        verify_weight_files,
    )
    from ts_onnx_providers import report_provider_status, require_onnxruntime

# Global model cache. Module-level rather than class-level: V3 node classes are
# locked, so `cls.cache = ...` would raise.
_MODEL_CACHE = {}

# One lock per cache key, so loading model A never blocks a caller that wants
# model B. A single global lock was held for the whole download — potentially
# minutes and ~1.5 GB — which serialised unrelated loads.
#
# ComfyUI runs one prompt at a time today, so the plain `if key in cache` this
# replaces was not actively losing races; a bare check-then-load would still
# download and load the same weights twice the moment anything does run two
# loaders concurrently.
_MODEL_CACHE_KEY_LOCKS: "dict[str, threading.Lock]" = {}

# Guards only the creation of the per-key locks above — never held during I/O.
_MODEL_CACHE_LOCK_REGISTRY = threading.Lock()


def _lock_for_cache_key(cache_key: str) -> threading.Lock:
    """Return the lock guarding one cache key, creating it once."""
    with _MODEL_CACHE_LOCK_REGISTRY:
        lock = _MODEL_CACHE_KEY_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _MODEL_CACHE_KEY_LOCKS[cache_key] = lock
        return lock
LOGGER = get_logger("TS CosyVoice Model Manager")

# Model configurations
MODEL_CONFIGS = {
    "Fun-CosyVoice3-0.5B": {
        "modelscope_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "huggingface_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "config_file": "cosyvoice3.yaml",
        "speech_tokenizer_file": "speech_tokenizer_v3.onnx",
        "recommended": True,
    },
}

# The GRPO reinforcement-learning post-trained LLM ships *inside* the same
# repository as an extra checkpoint rather than as a separate model: there is no
# `..._RL` repo (ModelScope 404, HuggingFace 401). The vendored loader hardcodes
# `llm.pt`, so selecting this one means loading it over the top afterwards —
# exactly what the vendored `CosyVoice3Model.load` does for the default.
RL_LLM_CHECKPOINT_NAME = "llm.rl.pt"

LLM_CHECKPOINT_STANDARD = "standard"
LLM_CHECKPOINT_RL = "reinforcement-learning"
LLM_CHECKPOINT_CHOICES = [LLM_CHECKPOINT_STANDARD, LLM_CHECKPOINT_RL]

COMMON_MODEL_FILES = {
    "campplus.onnx",
    "flow.pt",
    "hift.pt",
    "llm.pt",
}

MODEL_FILE_MIN_SIZES = {
    "cosyvoice3.yaml": 256,
    "campplus.onnx": 1_000_000,
    "speech_tokenizer_v3.onnx": 1_000_000,
    "llm.pt": 100_000_000,
    "flow.pt": 50_000_000,
    "hift.pt": 10_000_000,
}


_LOGGED_MODELS_DIR: str | None = None


def get_models_directory() -> str:
    """Get the base models directory for CosyVoice models"""
    global _LOGGED_MODELS_DIR

    # Try to get ComfyUI models directory
    try:
        import folder_paths
        base_models_dir = folder_paths.models_dir
    except Exception as e:
        LOGGER.warning("[TS CosyVoice Model Manager] Could not import folder_paths: %s", e)
        LOGGER.warning("[TS CosyVoice Model Manager] Using fallback models directory")
        base_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))

    cosyvoice_dir = os.path.join(base_models_dir, "cosyvoice")
    os.makedirs(cosyvoice_dir, exist_ok=True)

    # This helper is called from several places per load; log the resolved path once.
    if _LOGGED_MODELS_DIR != cosyvoice_dir:
        LOGGER.info("[TS CosyVoice Model Manager] Models directory: %s", cosyvoice_dir)
        _LOGGED_MODELS_DIR = cosyvoice_dir

    return cosyvoice_dir


def get_download_cache_directory() -> str:
    """Get a shared cache directory for resumable downloads."""
    cache_dir = os.path.join(get_models_directory(), ".download_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_expected_model_files(model_version: str) -> set[str]:
    """Get the required top-level files for a specific model root."""
    config = MODEL_CONFIGS[model_version]
    return set(COMMON_MODEL_FILES) | {
        config["config_file"],
        config["speech_tokenizer_file"],
    }


def find_model_root(model_dir: str, model_version: str | None = None) -> str | None:
    """Find the actual directory containing a complete downloaded model."""
    if not os.path.exists(model_dir):
        return None

    expected_files = get_expected_model_files(model_version) if model_version else None
    config_files = {"cosyvoice.yaml", "cosyvoice2.yaml", "cosyvoice3.yaml"}

    for root, _, files in os.walk(model_dir):
        file_set = set(files)
        if expected_files is not None:
            if expected_files.issubset(file_set):
                return root
            continue

        if COMMON_MODEL_FILES.issubset(file_set) and config_files.intersection(file_set):
            return root

    return None


def _validate_file_size(file_path: str, min_size: int) -> str | None:
    """Validate that a file exists and is large enough to avoid truncated downloads."""
    if not os.path.isfile(file_path):
        return f"Missing file: {file_path}"

    file_size = os.path.getsize(file_path)
    if file_size < min_size:
        return f"File is too small ({file_size} bytes): {file_path}"

    return None


def _validate_torch_archive(file_path: str) -> str | None:
    """Validate a torch checkpoint archive without fully loading model weights."""
    if not zipfile.is_zipfile(file_path):
        return f"Checkpoint is not a valid torch zip archive: {file_path}"

    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                return f"Corrupted checkpoint member '{bad_member}' in {file_path}"
            if not archive.namelist():
                return f"Checkpoint archive is empty: {file_path}"
    except zipfile.BadZipFile as e:
        return f"Corrupted checkpoint archive {file_path}: {e}"

    return None


def _validate_onnx_file(file_path: str) -> str | None:
    """Validate that an ONNX model can be opened by ONNX Runtime."""
    try:
        import onnxruntime

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        onnxruntime.InferenceSession(
            file_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        return f"Invalid ONNX model {file_path}: {e}"

    return None


# Sidecar recording which deep integrity checks passed, and for exactly which
# files. Lives inside the model root, so clear_model_directory() drops it with
# the rest when a model is re-downloaded.
VALIDATION_RECORD_NAME = ".ts_validation.json"

# What the deep pass actually verifies. Recorded verbatim so the file states what
# was checked rather than implying a stronger guarantee than was made.
_DEEP_CHECKS = ["torch_archive_crc", "onnx_session_load", "sha256_weights_manifest"]


def _file_signature(model_root: str, file_names: "set[str]") -> "dict[str, list[int]]":
    """Size and mtime of every required file — enough to notice a replaced file."""
    signature: dict[str, list[int]] = {}
    for file_name in sorted(file_names):
        stat_result = os.stat(os.path.join(model_root, file_name))
        signature[file_name] = [stat_result.st_size, stat_result.st_mtime_ns]
    return signature


def _read_validation_record(model_root: str) -> "dict[str, Any] | None":
    record_path = os.path.join(model_root, VALIDATION_RECORD_NAME)
    try:
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        # No record, unreadable, or malformed: fall back to validating for real.
        return None
    return record if isinstance(record, dict) else None


def _write_validation_record(model_root: str, model_version: str, signature: "dict[str, list[int]]") -> None:
    record = {
        "model_version": model_version,
        "checks_passed": _DEEP_CHECKS,
        "files": signature,
    }
    try:
        with open(os.path.join(model_root, VALIDATION_RECORD_NAME), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    except OSError as e:
        # A read-only model dir is not a reason to fail a working model; we just
        # pay for the deep check again next time.
        LOGGER.warning("[TS CosyVoice Model Manager] Could not write validation record: %s", e)


def _deep_checks_already_passed(
    model_root: str, model_version: str, signature: "dict[str, list[int]]"
) -> bool:
    """True when this exact set of files already passed the expensive checks."""
    record = _read_validation_record(model_root)
    if record is None:
        return False
    return (
        record.get("model_version") == model_version
        and record.get("checks_passed") == _DEEP_CHECKS
        and record.get("files") == signature
    )


def validate_model_root(model_root: str, model_version: str) -> str | None:
    """
    Run structural and integrity validation for a downloaded model root.

    The cheap structural checks (presence, size, config contents) run every time.
    The expensive ones — CRC-ing multi-hundred-megabyte torch archives and
    opening both ONNX models — used to run on the first load after every ComfyUI
    start, re-proving something that had not changed. They now run once per set
    of files and the result is recorded next to the weights; replacing any file
    invalidates the record and the deep pass runs again.
    """
    config = MODEL_CONFIGS[model_version]
    required_files = get_expected_model_files(model_version)

    for file_name in sorted(required_files):
        file_path = os.path.join(model_root, file_name)
        min_size = MODEL_FILE_MIN_SIZES.get(file_name, 1)
        size_error = _validate_file_size(file_path, min_size)
        if size_error:
            return size_error

    config_path = os.path.join(model_root, config["config_file"])
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config_text = config_file.read()
    except OSError as e:
        return f"Failed to read config file {config_path}: {e}"

    if "sample_rate" not in config_text or "get_tokenizer" not in config_text:
        return f"Config file is incomplete or invalid: {config_path}"

    try:
        signature = _file_signature(model_root, required_files)
    except OSError as e:
        return f"Could not stat model files in {model_root}: {e}"

    if _deep_checks_already_passed(model_root, model_version, signature):
        LOGGER.info(
            "[TS CosyVoice Model Manager] Integrity already recorded for these files; "
            "skipping checkpoint CRC and ONNX load"
        )
        return None

    LOGGER.info(
        "[TS CosyVoice Model Manager] Verifying model integrity (first time for these files)..."
    )

    for checkpoint_name in ("llm.pt", "flow.pt", "hift.pt"):
        archive_error = _validate_torch_archive(os.path.join(model_root, checkpoint_name))
        if archive_error:
            return archive_error

    for onnx_name in ("campplus.onnx", config["speech_tokenizer_file"]):
        onnx_error = _validate_onnx_file(os.path.join(model_root, onnx_name))
        if onnx_error:
            return onnx_error

    # Weights and ONNX graphs against the manifest. Several gigabytes, so it belongs
    # in the once-per-file-set pass rather than on every load. It warns rather than
    # refusing — see utils/ts_model_manifest.py for why — but the result must gate the
    # record below.
    mismatched_weights = verify_weight_files(model_root, model_version)

    if mismatched_weights:
        # Deliberately NOT recording success. Writing the record here meant the
        # warning appeared exactly once and every later load skipped the check
        # entirely, reporting "integrity already recorded" for files known not to
        # match. A load that could not be fully verified must stay unverified, so the
        # warning is repeated on every load instead of being silently absorbed.
        LOGGER.warning(
            "[TS CosyVoice Model Manager] Not recording an integrity pass: %d file(s) "
            "do not match %s. This will be re-checked and reported on every load until "
            "the files match the manifest or the manifest is regenerated.",
            len(mismatched_weights),
            "model_manifest.json",
        )
        return None

    _write_validation_record(model_root, model_version, signature)
    return None


def clear_model_directory(model_dir: str):
    """Remove existing downloaded files for a clean re-download."""
    if not os.path.exists(model_dir):
        return

    for entry in os.listdir(model_dir):
        entry_path = os.path.join(model_dir, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)
        else:
            try:
                os.remove(entry_path)
            except FileNotFoundError:
                pass


def ensure_runtime_support_files(model_root: str):
    """Create runtime support files that some public releases omit."""
    os.makedirs(model_root, exist_ok=True)
    spk2info_path = os.path.join(model_root, "spk2info.pt")
    if not os.path.exists(spk2info_path):
        try:
            torch.save({}, spk2info_path)
            LOGGER.info("[TS CosyVoice Model Manager] Created missing runtime file: %s", spk2info_path)
        except Exception as e:
            LOGGER.warning("[TS CosyVoice Model Manager] Could not create %s: %s", spk2info_path, e)


def download_model_modelscope(model_id: str, local_dir: str, revision: str = "master") -> str:
    """
    Download model from ModelScope.

    ModelScope publishes no immutable ref for this repository — its API reports
    ``ModelRevisions: None`` and ``master`` is the only branch — so unlike the
    HuggingFace path this cannot be pinned to a commit. What makes the result
    trustworthy is the sha256 verification that follows, not the revision.
    """
    log_banner(
        LOGGER,
        "[TS CosyVoice Model Manager] Downloading from ModelScope",
        Model=model_id,
        Revision=revision,
        Target=local_dir,
    )

    try:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "[TS CosyVoice Model Manager] The ModelScope download source needs the "
                "`modelscope` package, which is not installed. Either set "
                "download_source to HuggingFace (the default, and what this pack "
                "requires), or install it:\n"
                '    pip install "comfyui-ts-cosyvoice[modelscope]"\n'
                "It is optional because ModelScope is a mirror: HuggingFace serves the "
                "same files, verified against the same hashes."
            ) from exc

        model_path = snapshot_download(
            model_id=model_id,
            revision=revision,
            cache_dir=get_download_cache_directory(),
            local_dir=local_dir,
            local_files_only=False,
            enable_file_lock=True,
            max_workers=4,
        )

        log_banner(
            LOGGER,
            "[TS CosyVoice Model Manager] Download complete",
            Path=model_path,
        )

        return model_path

    except Exception as e:
        log_exception(LOGGER, "[TS CosyVoice Model Manager] ModelScope download failed", e)
        raise


def download_model_huggingface(model_id: str, local_dir: str, revision: str = "main") -> str:
    """
    Download model from HuggingFace.

    ``revision`` is the full commit hash recorded in ``model_manifest.json``, not a
    branch name. Fetching from ``main`` meant the bytes could change under us between
    two installs, and ``cosyvoice3.yaml`` is executed by HyperPyYAML once loaded.
    """
    log_banner(
        LOGGER,
        "[TS CosyVoice Model Manager] Downloading from HuggingFace",
        Model=model_id,
        Revision=revision,
        Target=local_dir,
    )

    try:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=get_download_cache_directory(),
            local_dir=local_dir,
            local_files_only=False,
            max_workers=4,
        )

        log_banner(
            LOGGER,
            "[TS CosyVoice Model Manager] Download complete",
            Path=model_path,
        )

        return model_path

    except Exception as e:
        log_exception(LOGGER, "[TS CosyVoice Model Manager] HuggingFace download failed", e)
        raise


def get_model_path(
    model_version: str,
    download_source: str = "ModelScope",
) -> str:
    """
    Get the path to a CosyVoice model, downloading if necessary

    Args:
        model_version: Model version name (e.g., "Fun-CosyVoice3-0.5B")
        download_source: "ModelScope" or "HuggingFace"
    Returns:
        Path to the model directory
    """
    if model_version not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model version: {model_version}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_version]
    models_dir = get_models_directory()
    model_dir = os.path.join(models_dir, model_version)

    existing_root = find_model_root(model_dir, model_version)
    if existing_root:
        ensure_runtime_support_files(existing_root)
        validation_error = validate_model_root(existing_root, model_version)
        if validation_error:
            LOGGER.warning("[TS CosyVoice Model Manager] Existing model validation failed: %s", validation_error)
            LOGGER.warning("[TS CosyVoice Model Manager] Clearing invalid model directory: %s", model_dir)
            clear_model_directory(model_dir)
        else:
            # Deliberately outside the validation_error channel above: that channel
            # wipes the model directory and re-downloads. A hash mismatch must not
            # delete gigabytes of someone's working install, so verify_config_files
            # raises instead, leaving everything on disk untouched.
            checked = verify_config_files(existing_root, model_version)
            log_banner(
                LOGGER,
                "[TS CosyVoice Model Manager] Model already exists",
                Path=existing_root,
                Verified=(
                    f"{checked} config files match model_manifest.json"
                    if checked
                    else "no manifest entries - contents not verified"
                ),
            )
            return existing_root

    os.makedirs(model_dir, exist_ok=True)

    def _download_from_source(source_name: str) -> str:
        revision = pinned_revision(model_version, source_name)
        if source_name == "ModelScope":
            model_id = config["modelscope_id"]
            download_model_modelscope(model_id, model_dir, revision=revision or "master")
        else:
            model_id = config["huggingface_id"]
            download_model_huggingface(model_id, model_dir, revision=revision or "main")

        resolved_root = find_model_root(model_dir, model_version)
        if not resolved_root:
            raise RuntimeError(f"Downloaded files are incomplete for {model_version} in {model_dir}")
        ensure_runtime_support_files(resolved_root)
        validation_error = validate_model_root(resolved_root, model_version)
        if validation_error:
            raise RuntimeError(f"Downloaded model validation failed: {validation_error}")
        # A fresh download at a pinned revision should match the manifest exactly. If
        # it does not, trying the other mirror is a reasonable next step, so this is
        # left to propagate into the alternate-source fallback below.
        verify_config_files(resolved_root, model_version)
        return resolved_root

    # Download model
    try:
        return _download_from_source(download_source)

    except Exception as e:
        LOGGER.warning("[TS CosyVoice Model Manager] Primary download source failed, trying alternate: %s", e)

        try:
            alternate_source = "HuggingFace" if download_source == "ModelScope" else "ModelScope"
            # Start the alternate source from an empty directory. A failed download
            # leaves partial files behind, and letting the second source fill in
            # around them produces a directory assembled from two snapshots — which
            # would pass the presence and size checks while being a mixture. The hash
            # verification would now catch it, but the cheaper fix is not to create
            # the mixture in the first place.
            LOGGER.info(
                "[TS CosyVoice Model Manager] Clearing partial download before trying %s, "
                "so the two sources cannot be mixed into one directory",
                alternate_source,
            )
            clear_model_directory(model_dir)
            return _download_from_source(alternate_source)

        except Exception as e2:
            raise RuntimeError(
                f"Failed to download model from both sources. Last error: {e2!s}"
            ) from e2


def load_cosyvoice_model(
    model_path: str,
    device: torch.device | None = None,
    fp16: bool = False,
) -> Any:
    """
    Load a CosyVoice model from disk

    Args:
        model_path: Path to model directory
        device: Target device (will auto-detect if None)

    Returns:
        Loaded CosyVoice model instance
    """
    log_banner(
        LOGGER,
        "[TS CosyVoice Model Manager] Loading CosyVoice model",
        Path=model_path,
    )

    # Before the vendored frontend reaches its own module-level `import onnxruntime`:
    # raise a pack-level error naming the package to install, and warn when torch has
    # CUDA but ONNX Runtime cannot offer the CUDA provider. ONNX Runtime treats a
    # missing provider as a silent fall back to CPU, which would put the speech
    # tokenizer on the CPU of a GPU machine with nothing but ORT's own warning to say
    # so.
    require_onnxruntime()
    report_provider_status()

    # The vendored loader calls torch.load without weights_only on downloaded
    # checkpoints, so an older PyTorch would unpickle them. Refuse rather than patch
    # the vendored tree.
    require_safe_torch_load()

    # spk2info.pt is the one .pt on the load path the manifest cannot cover: upstream
    # does not ship it, this pack creates an empty one, and the vendored frontend
    # torch.loads whatever is present. Prove it is data before that happens.
    spk2info_root = find_model_root(model_path) or model_path
    validate_tensors_only(
        os.path.join(spk2info_root, "spk2info.pt"), "the speaker-info file"
    )

    try:
        # Import from vendored cosyvoice package (bundled with this node pack)
        import sys
        vendored_path = os.path.join(os.path.dirname(__file__), "..")
        if vendored_path not in sys.path:
            sys.path.append(vendored_path)

        # The vendored tree imports whisper and librosa.filters at module level
        # purely to compute mel spectrograms, and both pull numba, which refuses
        # to load against NumPy 2.2+. Substitute the pack's own mel front-end for
        # whichever of them cannot import, before the vendored import runs. A
        # no-op where they import fine. See utils/ts_runtime_shims.py.
        try:
            from .ts_runtime_shims import install_cosyvoice_import_shims
        except ImportError:
            from ts_runtime_shims import install_cosyvoice_import_shims

        install_cosyvoice_import_shims()

        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise _diagnose_runtime_import_failure(exc) from exc

        # Determine device
        if device is None:
            import comfy.model_management
            device = comfy.model_management.get_torch_device()

        LOGGER.info("[TS CosyVoice Model Manager] Target device: %s", device)
        LOGGER.info("[TS CosyVoice Model Manager] FP16 enabled: %s", fp16)

        model_dir = find_model_root(model_path)

        if model_dir is None:
            raise FileNotFoundError(f"Could not find cosyvoice.yaml/cosyvoice3.yaml in {model_path}")

        LOGGER.info("[TS CosyVoice Model Manager] Found model root: %s", model_dir)

        # Use AutoModel to automatically detect and load the correct model type
        LOGGER.info("[TS CosyVoice Model Manager] Using AutoModel to load from: %s", model_dir)
        model = AutoModel(model_dir=model_dir, load_trt=False, fp16=fp16)

        # Note: CosyVoice handles device placement internally, no need to call .to()

        log_banner(LOGGER, "[TS CosyVoice Model Manager] Model loaded successfully")

        return model

    except Exception as e:
        log_exception(LOGGER, "[TS CosyVoice Model Manager] ERROR loading model", e)
        raise


# numba refuses to import against a NumPy newer than it supports, and the vendored
# CosyVoice frontend imports `whisper` at module level — whisper in turn imports
# numba. So a NumPy that is merely *newer* than numba expects takes the entire pack
# down at model-load time, with a traceback that names numba and mentions neither
# this pack nor how to fix it.
_NUMBA_SIGNATURES = ("numba", "numpy")


def _diagnose_runtime_import_failure(exc: ImportError) -> RuntimeError:
    """
    Turn an unhelpful vendored import failure into an actionable one.

    Args:
        exc: the original ImportError raised while importing the CosyVoice runtime.

    Returns:
        A RuntimeError to raise in its place. The original is kept as the cause.
    """
    message = str(exc).lower()
    if all(token in message for token in _NUMBA_SIGNATURES):
        installed = "unknown"
        try:
            import numpy

            installed = numpy.__version__
        except Exception:
            pass
        return RuntimeError(
            "[TS CosyVoice Model Manager] The CosyVoice runtime could not be imported "
            f"because numba rejects the installed NumPy ({installed}).\n"
            "The pack normally works around this: the only things on the synthesis "
            "path that needed numba were two mel spectrograms, and it substitutes its "
            "own implementation for them (utils/ts_runtime_shims.py). Reaching this "
            "error means something else in the import chain now wants numba too, "
            "which the shims do not cover.\n"
            "Either report it with the traceback below, or install a NumPy that numba "
            "accepts (numpy<2.2) into the ComfyUI Python as a stopgap.\n"
            f"Original error: {exc}"
        )

    return RuntimeError(
        "[TS CosyVoice Model Manager] The CosyVoice runtime could not be imported. "
        "Install the pack requirements into the ComfyUI Python and restart ComfyUI:\n"
        "    pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )


def apply_rl_llm_checkpoint(model: Any, model_root: str, device: "torch.device | None" = None) -> None:
    """
    Swap in the reinforcement-learning LLM checkpoint shipped alongside the default.

    Mirrors what the vendored ``CosyVoice3Model.load`` does for ``llm.pt``: load the
    state dict and apply it strictly. Done after the model is fully constructed, so
    no vendored file has to be patched to point at a different filename.

    Args:
        model: the loaded runtime returned by ``load_cosyvoice_model``.
        model_root: directory the weights were found in.
        device: map_location for the checkpoint; defaults to the model's own device.

    Raises:
        FileNotFoundError: if the RL checkpoint is not in the downloaded model.
        RuntimeError: if it exists but does not fit this model.
    """
    checkpoint_path = os.path.join(model_root, RL_LLM_CHECKPOINT_NAME)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"[TS CosyVoice Model Manager] The reinforcement-learning checkpoint "
            f"{RL_LLM_CHECKPOINT_NAME} is not present in {model_root}. It ships with "
            f"Fun-CosyVoice3-0.5B-2512; delete the model folder and let the loader "
            f"download it again, or switch llm_checkpoint back to 'standard'."
        )

    llm_module = getattr(getattr(model, "model", None), "llm", None)
    if llm_module is None:
        raise RuntimeError(
            "[TS CosyVoice Model Manager] Could not reach model.llm to apply the "
            "reinforcement-learning checkpoint; switch llm_checkpoint to 'standard'."
        )

    map_location = device if device is not None else getattr(model.model, "device", "cpu")
    try:
        # weights_only=True per the pack's own policy: this file is downloaded, and
        # it only ever contains tensors, so there is no reason to run pickle.
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        llm_module.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise RuntimeError(
            f"[TS CosyVoice Model Manager] Failed to apply {RL_LLM_CHECKPOINT_NAME}: {e}"
        ) from e

    LOGGER.info(
        "[TS CosyVoice Model Manager] Applied reinforcement-learning LLM checkpoint: %s",
        RL_LLM_CHECKPOINT_NAME,
    )


def _torch_modules_of(model: Any) -> "list[torch.nn.Module]":
    """
    The nn.Modules a loaded CosyVoice runtime owns, so they can be moved off the GPU.

    Walks one level into the runtime and its inner model rather than assuming the
    attribute layout: upstream has moved these around before, and a rename should
    cost a less thorough unload, not an AttributeError during cleanup.
    """
    found: list[torch.nn.Module] = []
    seen: set[int] = set()

    def collect(container: Any) -> None:
        if container is None:
            return
        for name in dir(container):
            if name.startswith("__"):
                continue
            try:
                value = getattr(container, name)
            except Exception:
                continue
            if isinstance(value, torch.nn.Module) and id(value) not in seen:
                seen.add(id(value))
                found.append(value)

    collect(model)
    collect(getattr(model, "model", None))
    return found


def unload_cosyvoice_model(cache_key: str | None = None) -> int:
    """
    Drop cached models and give their VRAM back.

    Args:
        cache_key: evict just this entry, or every entry when None.

    Returns:
        How many cache entries were evicted.

    The cache had no eviction at all, so every distinct precision or checkpoint
    combination accumulated its own multi-gigabyte copy for the lifetime of the
    process; toggling fp16 twice meant three resident models.

    Caveat worth stating plainly: this moves the modules to CPU, so a *different*
    node still holding this exact model object would find it on the CPU. ComfyUI
    executes one prompt at a time and the loader runs before the synthesis nodes it
    feeds, so in practice the evicted model is one nothing is using. A graph with two
    loader nodes configured differently and both feeding synthesis in the same run is
    the case to be aware of.
    """
    # Detach the entries under the lock, then do the slow part outside it. Moving
    # several gigabytes to CPU is I/O, and this lock also gates handing out per-key
    # locks — holding it across the transfer would stall an unrelated load.
    with _MODEL_CACHE_LOCK_REGISTRY:
        keys = [cache_key] if cache_key is not None else list(_MODEL_CACHE)
        detached = [(key, _MODEL_CACHE.pop(key)) for key in keys if key in _MODEL_CACHE]

    evicted = 0
    for key, model_info in detached:
        evicted += 1
        for module in _torch_modules_of(model_info.get("model")):
            try:
                module.to("cpu")
            except Exception as exc:
                LOGGER.warning(
                    "[TS CosyVoice Model Manager] Could not move a module to CPU "
                    "while unloading %s: %s",
                    key,
                    exc,
                )
        # The dict is deliberately left intact rather than cleared. It is the same
        # object get_cached_model handed to the loader node, and it flows down the
        # graph as COSYVOICE_MODEL — emptying it would turn a downstream synthesis
        # node's model into `{}` mid-run, i.e. a KeyError instead of, at worst, a
        # model that is now on the CPU. Dropping the cache's own reference is enough
        # to free anything nobody else is holding.
        LOGGER.info("[TS CosyVoice Model Manager] Unloaded cached model: %s", key)

    if evicted:
        import gc

        gc.collect()
        try:
            import comfy.model_management as mm

            mm.soft_empty_cache()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return evicted


def _evict_other_models(keep_key: str) -> None:
    """
    Enforce one resident CosyVoice model, freeing VRAM *before* the next one loads.

    Order matters: releasing first is the whole point. Freeing afterwards would still
    need both copies resident simultaneously, which is exactly the peak that runs a
    card out of memory.
    """
    stale = [key for key in list(_MODEL_CACHE) if key != keep_key]
    for key in stale:
        LOGGER.info(
            "[TS CosyVoice Model Manager] Releasing %s before loading %s "
            "(one CosyVoice model is kept resident at a time)",
            key,
            keep_key,
        )
        unload_cosyvoice_model(key)


def get_cached_model(
    model_version: str,
    download_source: str = "ModelScope",
    device: torch.device | None = None,
    fp16: bool = False,
    llm_checkpoint: str = LLM_CHECKPOINT_STANDARD,
) -> Dict[str, Any]:
    """
    Get a CosyVoice model, using cache if available

    Args:
        model_version: Model version name
        download_source: Download source
        device: Target device
        fp16: Enable half precision loading when supported
    Returns:
        Dictionary containing model and metadata
    """
    # The vendored runtime ignores the requested device (it always loads on
    # cuda-if-available) and downgrades fp16 to fp32 when CUDA is absent. So the
    # cache key must key only on what actually changes the loaded weights:
    # the model version and the *effective* fp16 flag. Keying on the requested
    # device string would store several identical models under different keys
    # (e.g. "auto" -> cuda:0 vs "cuda" -> cuda), doubling VRAM.
    effective_fp16 = bool(fp16) and torch.cuda.is_available()
    # The checkpoint choice changes the loaded weights, so it must be part of the
    # key — otherwise toggling it would silently return the previously loaded LLM.
    cache_key = f"{model_version}_fp16_{int(effective_fp16)}_llm_{llm_checkpoint}"

    # Fast path: already loaded, no lock needed.
    cached_info = _MODEL_CACHE.get(cache_key)
    if cached_info is not None:
        LOGGER.info("[TS CosyVoice Model Manager] Using cached model: %s", model_version)
        return cached_info

    with _lock_for_cache_key(cache_key):
        # Re-check under the lock: another thread may have finished loading while
        # we waited, and loading again would cost a second copy in VRAM.
        cached_info = _MODEL_CACHE.get(cache_key)
        if cached_info is not None:
            LOGGER.info("[TS CosyVoice Model Manager] Using cached model: %s", model_version)
            return cached_info

        return _load_and_cache_model(
            cache_key, model_version, download_source, device, fp16, llm_checkpoint
        )


def _load_and_cache_model(
    cache_key: str,
    model_version: str,
    download_source: str,
    device: torch.device | None,
    fp16: bool,
    llm_checkpoint: str = LLM_CHECKPOINT_STANDARD,
) -> Dict[str, Any]:
    """Download, load and cache a model. Callers must hold this key's lock."""
    # Get model path (download if needed)
    model_path = get_model_path(model_version, download_source)

    # Free any other resident model first, so peak usage is one model rather than
    # two. Done after the download resolves and before the weights are loaded.
    _evict_other_models(cache_key)

    # Load model
    model = load_cosyvoice_model(model_path, device, fp16=fp16)

    if llm_checkpoint == LLM_CHECKPOINT_RL:
        model_root = find_model_root(model_path) or model_path
        # The *actual* device, not the requested one. On Apple Silicon the loader
        # resolves `auto` to mps, but the vendored runtime has no mps path and
        # runs on cpu — passing the request would stage a multi-hundred-megabyte
        # checkpoint through mps only to copy it into cpu parameters, which on a
        # unified-memory Mac is both wasteful and a real allocation risk.
        apply_rl_llm_checkpoint(model, model_root, get_runtime_device(model))

    # Detect model version for nodes to use
    version_lower = model_version.lower()
    is_cosyvoice3 = "cosyvoice3" in version_lower or "fun-cosyvoice3" in version_lower
    is_cosyvoice2 = False

    # Record the device the model actually runs on, not the requested one — the
    # runtime pins its own device and the request is best-effort (see loader).
    actual_device = get_runtime_device(model)

    # Create model info dict
    model_info = {
        "model": model,
        "model_name": model_version,
        "model_version": model_version,
        "model_path": model_path,
        "device": actual_device,
        "fp16": fp16,
        "sample_rate": model.sample_rate,  # Use actual model sample rate (24000 for v2/v3, 22050 for v1)
        "llm_checkpoint": llm_checkpoint,
        "is_cosyvoice3": is_cosyvoice3,
        "is_cosyvoice2": is_cosyvoice2,
    }

    LOGGER.info("[TS CosyVoice Model Manager] Model sample rate: %s Hz", model.sample_rate)
    LOGGER.info(
        "[TS CosyVoice Model Manager] Model type: %s",
        "CosyVoice3" if is_cosyvoice3 else "Unknown",
    )

    # Cache model
    _MODEL_CACHE[cache_key] = model_info

    return model_info
