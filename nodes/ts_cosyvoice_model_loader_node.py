"""
TS CosyVoice3 Model Loader Node
Downloads and loads CosyVoice models with automatic weight management
"""

import torch
from typing import Tuple, Dict, Any
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_model_manager import get_cached_model, MODEL_CONFIGS
    from ..utils.ts_logging import get_logger, log_banner, log_exception
except (ImportError, ValueError):
    from utils.ts_model_manager import get_cached_model, MODEL_CONFIGS
    from utils.ts_logging import get_logger, log_banner, log_exception


LOGGER = get_logger("TS CosyVoice Model Loader")


def _resolve_target_device(device: str) -> torch.device:
    """Resolve requested device with GPU-first auto selection and safe fallback."""
    if device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("[TS CosyVoice Model Loader] CUDA requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        if device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
            LOGGER.warning("[TS CosyVoice Model Loader] MPS requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device)

    accelerator_checks = []
    if torch.cuda.is_available():
        accelerator_checks.append(torch.device("cuda", torch.cuda.current_device()))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        accelerator_checks.append(torch.device("xpu", torch.xpu.current_device()))
    if hasattr(torch, "npu") and torch.npu.is_available():
        accelerator_checks.append(torch.device("npu", torch.npu.current_device()))
    if hasattr(torch, "mlu") and torch.mlu.is_available():
        accelerator_checks.append(torch.device("mlu", torch.mlu.current_device()))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator_checks.append(torch.device("mps"))

    if accelerator_checks:
        target_device = accelerator_checks[0]
        LOGGER.info("[TS CosyVoice Model Loader] Auto-selected accelerator: %s", target_device)
        return target_device

    try:
        import comfy.model_management
        target_device = comfy.model_management.get_torch_device()
        LOGGER.info("[TS CosyVoice Model Loader] Auto-selected Comfy device fallback: %s", target_device)
        return target_device
    except Exception as e:
        LOGGER.warning("[TS CosyVoice Model Loader] Auto device detection fallback failed: %s", e)

    LOGGER.info("[TS CosyVoice Model Loader] Falling back to CPU")
    return torch.device("cpu")


class TS_CosyVoice3_ModelLoader:
    """
    Load CosyVoice models with automatic downloading and caching
    """

    RETURN_TYPES = ("COSYVOICE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "TS CosyVoice3/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (list(MODEL_CONFIGS.keys()), {
                    "default": "Fun-CosyVoice3-0.5B",
                    "description": "CosyVoice model version to load",
                    "tooltip": "Выберите версию модели CosyVoice для загрузки."
                }),
                "download_source": (["HuggingFace", "ModelScope"], {
                    "default": "HuggingFace",
                    "description": "Source to download model from",
                    "tooltip": "Источник, из которого будет скачана модель."
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "description": "Device to load model on",
                    "tooltip": "Устройство для загрузки и запуска модели; auto предпочитает GPU."
                }),
            },
            "optional": {
                "fp16": ("BOOLEAN", {
                    "default": False,
                    "description": "Enable FP16 model loading on supported accelerators",
                    "tooltip": "Включает FP16 на поддерживаемых ускорителях для снижения расхода видеопамяти."
                }),
            }
        }

    def load_model(
        self,
        model_version: str,
        download_source: str = "ModelScope",
        device: str = "auto",
        fp16: bool = False,
    ) -> Tuple[Dict[str, Any]]:
        """
        Load a CosyVoice model

        Args:
            model_version: Model version to load
            download_source: Download source (ModelScope or HuggingFace)
            device: Target device
            fp16: Enable FP16 model loading when supported

        Returns:
            Tuple containing model info dict
        """
        log_banner(
            LOGGER,
            "[TS CosyVoice Model Loader] Loading model...",
            Version=model_version,
            Source=download_source,
            Device=device,
            FP16=fp16,
        )

        try:
            # Determine device
            target_device = _resolve_target_device(device)

            # Get cached model
            model_info = get_cached_model(
                model_version=model_version,
                download_source=download_source,
                device=target_device,
                fp16=fp16,
            )

            log_banner(
                LOGGER,
                "[TS CosyVoice Model Loader] Model loaded successfully!",
                Model=model_info["model_name"],
                Device=model_info["device"],
                FP16=model_info.get("fp16", False),
                Path=model_info["model_path"],
            )

            return (model_info,)

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            log_exception(LOGGER, "[TS CosyVoice Model Loader] ERROR", e)
            raise RuntimeError(error_msg)
