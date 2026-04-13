"""
Model Manager for TS CosyVoice3
Handles model downloading, caching, and loading
"""

import os
import shutil
import zipfile
from typing import Optional, Dict, Any

import torch

try:
    from .ts_logging import get_logger, log_banner, log_exception
except (ImportError, ValueError):
    from ts_logging import get_logger, log_banner, log_exception

# Global model cache
_MODEL_CACHE = {}
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


def get_models_directory() -> str:
    """Get the base models directory for CosyVoice models"""
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

    LOGGER.info("[TS CosyVoice Model Manager] Models directory: %s", cosyvoice_dir)
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


def find_model_root(model_dir: str, model_version: Optional[str] = None) -> Optional[str]:
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


def validate_model_root(model_root: str, model_version: str) -> str | None:
    """Run structural and integrity validation for a downloaded model root."""
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
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_text = config_file.read()
    except OSError as e:
        return f"Failed to read config file {config_path}: {e}"

    if "sample_rate" not in config_text or "get_tokenizer" not in config_text:
        return f"Config file is incomplete or invalid: {config_path}"

    for checkpoint_name in ("llm.pt", "flow.pt", "hift.pt"):
        archive_error = _validate_torch_archive(os.path.join(model_root, checkpoint_name))
        if archive_error:
            return archive_error

    for onnx_name in ("campplus.onnx", config["speech_tokenizer_file"]):
        onnx_error = _validate_onnx_file(os.path.join(model_root, onnx_name))
        if onnx_error:
            return onnx_error

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


def download_model_modelscope(model_id: str, local_dir: str) -> str:
    """Download model from ModelScope"""
    log_banner(
        LOGGER,
        "[TS CosyVoice Model Manager] Downloading from ModelScope",
        Model=model_id,
        Target=local_dir,
    )

    try:
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=model_id,
            revision='master',
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


def download_model_huggingface(model_id: str, local_dir: str) -> str:
    """Download model from HuggingFace"""
    log_banner(
        LOGGER,
        "[TS CosyVoice Model Manager] Downloading from HuggingFace",
        Model=model_id,
        Target=local_dir,
    )

    try:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=model_id,
            revision='main',
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


def check_model_exists(model_dir: str, model_version: Optional[str] = None) -> bool:
    """Check if model files exist in the directory"""
    model_root = find_model_root(model_dir, model_version)
    if model_root is None or model_version is None:
        return model_root is not None
    return validate_model_root(model_root, model_version) is None


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
            log_banner(
                LOGGER,
                "[TS CosyVoice Model Manager] Model already exists",
                Path=existing_root,
            )
            return existing_root

    os.makedirs(model_dir, exist_ok=True)

    def _download_from_source(source_name: str) -> str:
        if source_name == "ModelScope":
            model_id = config["modelscope_id"]
            download_model_modelscope(model_id, model_dir)
        else:
            model_id = config["huggingface_id"]
            download_model_huggingface(model_id, model_dir)

        resolved_root = find_model_root(model_dir, model_version)
        if not resolved_root:
            raise RuntimeError(f"Downloaded files are incomplete for {model_version} in {model_dir}")
        ensure_runtime_support_files(resolved_root)
        validation_error = validate_model_root(resolved_root, model_version)
        if validation_error:
            raise RuntimeError(f"Downloaded model validation failed: {validation_error}")
        return resolved_root

    # Download model
    try:
        return _download_from_source(download_source)

    except Exception as e:
        LOGGER.warning("[TS CosyVoice Model Manager] Primary download source failed, trying alternate: %s", e)

        try:
            alternate_source = "HuggingFace" if download_source == "ModelScope" else "ModelScope"
            return _download_from_source(alternate_source)

        except Exception as e2:
            raise RuntimeError(f"Failed to download model from both sources. Last error: {str(e2)}")


def load_cosyvoice_model(
    model_path: str,
    device: Optional[torch.device] = None,
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

    try:
        # Import from vendored cosyvoice package (bundled with this node pack)
        import sys
        vendored_path = os.path.join(os.path.dirname(__file__), "..")
        if vendored_path not in sys.path:
            sys.path.insert(0, vendored_path)

        from cosyvoice.cli.cosyvoice import AutoModel

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


def get_cached_model(
    model_version: str,
    download_source: str = "ModelScope",
    device: Optional[torch.device] = None,
    fp16: bool = False,
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
    cache_key = f"{model_version}_{device}_fp16_{int(fp16)}"

    # Check cache
    if cache_key in _MODEL_CACHE:
        LOGGER.info("[TS CosyVoice Model Manager] Using cached model: %s", model_version)
        return _MODEL_CACHE[cache_key]

    # Get model path (download if needed)
    model_path = get_model_path(model_version, download_source)

    # Load model
    model = load_cosyvoice_model(model_path, device, fp16=fp16)

    # Detect model version for nodes to use
    version_lower = model_version.lower()
    is_cosyvoice3 = "cosyvoice3" in version_lower or "fun-cosyvoice3" in version_lower
    is_cosyvoice2 = False

    # Create model info dict
    model_info = {
        "model": model,
        "model_name": model_version,
        "model_version": model_version,
        "model_path": model_path,
        "device": device,
        "fp16": fp16,
        "sample_rate": model.sample_rate,  # Use actual model sample rate (24000 for v2/v3, 22050 for v1)
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
