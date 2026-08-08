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
    from ..utils.ts_logging import get_logger, log_banner, log_exception
    from ..utils.ts_model_manager import (
        LLM_CHECKPOINT_CHOICES,
        LLM_CHECKPOINT_STANDARD,
        MODEL_CONFIGS,
        get_cached_model,
    )
    from ._v3_types import CosyVoiceModel
except (ImportError, ValueError):
    from nodes._v3_types import CosyVoiceModel
    from utils.ts_logging import get_logger, log_banner, log_exception
    from utils.ts_model_manager import (
        LLM_CHECKPOINT_CHOICES,
        LLM_CHECKPOINT_STANDARD,
        MODEL_CONFIGS,
        get_cached_model,
    )


LOGGER = get_logger("TS CosyVoice Model Loader")


# What the vendored CosyVoice runtime actually implements. Every place it picks a
# device does `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` —
# cosyvoice/cli/model.py:36, :244, :390 and cosyvoice/cli/frontend.py:80. There is
# no MPS, XPU, NPU or MLU path anywhere in it, so on Apple Silicon the model runs
# on the CPU no matter what this node is asked for.
_RUNTIME_SUPPORTED_DEVICE_TYPES = ("cuda", "cpu")


def _normalise_to_supported_device(target_device: torch.device) -> torch.device:
    """
    Reduce an unsupported accelerator to CPU, and say so.

    Reporting the accelerator we detected would be a lie: the runtime ignores it.
    An earlier version logged "Auto-selected accelerator: mps" immediately before
    "Model loaded successfully! Device: cpu", which reads like a bug. The option
    itself stays in the widget — removing a combo value would break saved
    workflows that selected it.
    """
    if target_device.type in _RUNTIME_SUPPORTED_DEVICE_TYPES:
        return target_device

    LOGGER.warning(
        "[TS CosyVoice Model Loader] Device '%s' is not supported by the CosyVoice "
        "runtime, which implements CUDA and CPU only — the model will run on CPU. "
        "This is an upstream limitation, not a configuration problem.",
        target_device,
    )
    return torch.device("cpu")


def _resolve_target_device(device: str) -> torch.device:
    """Resolve requested device with GPU-first auto selection and safe fallback."""
    if device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("[TS CosyVoice Model Loader] CUDA requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        if device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
            LOGGER.warning("[TS CosyVoice Model Loader] MPS requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        return _normalise_to_supported_device(torch.device(device))

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
        target_device = _normalise_to_supported_device(accelerator_checks[0])
        LOGGER.info("[TS CosyVoice Model Loader] Auto-selected device: %s", target_device)
        return target_device

    try:
        import comfy.model_management
        target_device = _normalise_to_supported_device(comfy.model_management.get_torch_device())
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
                    tooltip="Предпочитаемое устройство (best-effort). Рантайм CosyVoice "
                            "реализует только CUDA и CPU: mps на Apple Silicon и "
                            "прочие ускорители сводятся к CPU — это ограничение "
                            "upstream, а не ошибка настройки.",
                ),
                IO.Boolean.Input(
                    "fp16",
                    default=False,
                    optional=True,
                    tooltip="Включает FP16 для снижения расхода видеопамяти. Требует "
                            "CUDA: без неё рантайм сам возвращается к FP32, в том "
                            "числе на Apple Silicon.",
                ),
                # Appended after fp16 — the end of the existing widget list — so
                # saved graphs keep mapping their shorter widgets_values arrays.
                IO.Combo.Input(
                    "llm_checkpoint",
                    options=list(LLM_CHECKPOINT_CHOICES),
                    default=LLM_CHECKPOINT_STANDARD,
                    optional=True,
                    tooltip="Какой LLM-чекпоинт загружать. 'reinforcement-learning' "
                            "использует llm.rl.pt из той же папки модели — вариант "
                            "после GRPO post-training с лучшими метриками. "
                            "Переключение перезагружает модель.",
                ),
            ],
            outputs=[
                CosyVoiceModel.Output(display_name="model"),
            ],
            search_aliases=[],
            is_output_node=False,
            essentials_category="Audio",
        )

    @classmethod
    def execute(
        cls,
        model_version: str,
        download_source: str = "ModelScope",
        device: str = "auto",
        fp16: bool = False,
        llm_checkpoint: str = LLM_CHECKPOINT_STANDARD,
    ) -> IO.NodeOutput:
        """Load a CosyVoice model and return its info dict as COSYVOICE_MODEL."""
        log_banner(
            LOGGER,
            "[TS CosyVoice Model Loader] Loading model...",
            Version=model_version,
            Source=download_source,
            Device=device,
            FP16=fp16,
            LLM=llm_checkpoint,
        )

        try:
            target_device = _resolve_target_device(device)

            model_info: Dict[str, Any] = get_cached_model(
                model_version=model_version,
                download_source=download_source,
                device=target_device,
                fp16=fp16,
                llm_checkpoint=llm_checkpoint,
            )

            log_banner(
                LOGGER,
                "[TS CosyVoice Model Loader] Model loaded successfully!",
                Model=model_info["model_name"],
                Device=model_info["device"],
                FP16=model_info.get("fp16", False),
                LLM=model_info.get("llm_checkpoint", "standard"),
                Path=model_info["model_path"],
            )

            return IO.NodeOutput(model_info)

        except Exception as e:
            error_msg = f"Error loading model: {e!s}"
            log_exception(LOGGER, "[TS CosyVoice Model Loader] ERROR", e)
            raise RuntimeError(error_msg) from e
