"""
TS CosyVoice3 Model Loader Node (V3 schema).
Downloads and loads CosyVoice models with automatic weight management.

Migrated from V1 to V3 on 2026-05-07.
Public contract preserved:
- node_id: TS_CosyVoice3_ModelLoader
- inputs (required): model_version, download_source, device
- inputs (optional): fp16
- outputs: (COSYVOICE_MODEL,) named "model"
The model cache (_MODEL_CACHE in utils.ts_model_manager) stays where it is —
V3 forbids class-level mutable state but module-level cache is fine.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch

from comfy_api.v0_0_2 import IO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.ts_model_manager import get_cached_model, MODEL_CONFIGS
    from ..utils.ts_logging import get_logger, log_banner, log_exception
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from utils.ts_model_manager import get_cached_model, MODEL_CONFIGS
    from utils.ts_logging import get_logger, log_banner, log_exception
    from nodes._v3_types import CosyVoiceModel


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


class TS_CosyVoice3_ModelLoader(IO.ComfyNode):
    """Load CosyVoice models with automatic downloading and caching."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_CosyVoice3_ModelLoader",
            display_name="TS CosyVoice Model Loader",
            category="TS CosyVoice3/Loaders",
            description="Load a CosyVoice model with automatic download and caching.",
            inputs=[
                IO.Combo.Input(
                    "model_version",
                    options=list(MODEL_CONFIGS.keys()),
                    default="Fun-CosyVoice3-0.5B",
                    tooltip="Выберите версию модели CosyVoice для загрузки.",
                ),
                IO.Combo.Input(
                    "download_source",
                    options=["HuggingFace", "ModelScope"],
                    default="HuggingFace",
                    tooltip="Источник, из которого будет скачана модель.",
                ),
                IO.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu", "mps"],
                    default="auto",
                    tooltip="Устройство для загрузки и запуска модели; auto предпочитает GPU.",
                ),
                IO.Boolean.Input(
                    "fp16",
                    default=False,
                    optional=True,
                    tooltip="Включает FP16 на поддерживаемых ускорителях для снижения расхода видеопамяти.",
                ),
            ],
            outputs=[
                CosyVoiceModel.Output(display_name="model"),
            ],
            search_aliases=[],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        model_version: str,
        download_source: str = "ModelScope",
        device: str = "auto",
        fp16: bool = False,
    ) -> IO.NodeOutput:
        """Load a CosyVoice model and return its info dict as COSYVOICE_MODEL."""
        log_banner(
            LOGGER,
            "[TS CosyVoice Model Loader] Loading model...",
            Version=model_version,
            Source=download_source,
            Device=device,
            FP16=fp16,
        )

        try:
            target_device = _resolve_target_device(device)

            model_info: Dict[str, Any] = get_cached_model(
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

            return IO.NodeOutput(model_info)

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            log_exception(LOGGER, "[TS CosyVoice Model Loader] ERROR", e)
            raise RuntimeError(error_msg)
