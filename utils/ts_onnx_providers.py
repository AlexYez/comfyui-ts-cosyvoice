"""
Say out loud which ONNX Runtime is installed and what it can actually accelerate.

The pack does not declare an ONNX Runtime as a required dependency: ``onnxruntime``
and ``onnxruntime-gpu`` are separate distributions that both provide the module
named ``onnxruntime``, and ONNX Runtime supports exactly one of them per
environment. Declaring the CPU build would install it beside somebody's
onnxruntime-gpu and leave the winner undefined.

The cost of not declaring it is that two situations must be reported clearly instead
of guessed at:

- **Nothing installed.** The vendored frontend imports ``onnxruntime`` at module
  level, so without it the failure is an ImportError from inside vendored code that
  names neither the pack nor the fix.

- **CPU build on a CUDA machine.** ``cosyvoice/cli/frontend.py`` asks for
  ``CUDAExecutionProvider`` whenever ``torch.cuda.is_available()``. ONNX Runtime does
  not treat an unavailable provider as an error — it falls back to CPU and carries
  on. The speech tokenizer then runs on the CPU on a machine that has a GPU, and the
  only hint is a warning buried in ORT's own output. That is the quiet
  half-performance case this module exists to make loud.

Reporting only. Nothing here installs anything or changes provider selection.
"""

from __future__ import annotations

try:
    from .ts_logging import get_logger
except ImportError:  # loaded as a top-level module
    from ts_logging import get_logger

LOGGER = get_logger("TS CosyVoice3 ONNX")
LOG_PREFIX = "[TS CosyVoice3 ONNX]"

CUDA_PROVIDER = "CUDAExecutionProvider"

INSTALL_HINT = (
    "Install exactly one ONNX Runtime into the Python that runs ComfyUI:\n"
    "    pip install onnxruntime-gpu      # NVIDIA GPU\n"
    "    pip install onnxruntime          # CPU only, or Apple Silicon\n"
    "Do not install both: they provide the same module name and ONNX Runtime "
    "supports one variant per environment."
)

# Reported once per process. This is called on every model load, and a warning that
# repeats on every load is a warning people learn to scroll past.
_ANNOUNCED = False


def available_providers() -> list[str]:
    """Execution providers ONNX Runtime reports, or [] if it is not installed."""
    try:
        import onnxruntime
    except ImportError:
        return []
    try:
        return list(onnxruntime.get_available_providers())
    except Exception as exc:  # pragma: no cover - malformed ORT install
        LOGGER.warning("%s Could not query execution providers: %s", LOG_PREFIX, exc)
        return []


# What the vendored frontend actually calls. Importing the module is not enough to
# know these exist: see require_onnxruntime.
_REQUIRED_ATTRIBUTES = ("SessionOptions", "InferenceSession", "get_available_providers")


def require_onnxruntime() -> list[str]:
    """
    Return the available providers, raising a pack-level error if ORT is unusable.

    Called before the vendored code reaches its own module-level ``import
    onnxruntime``, so the user sees which package to install rather than an
    ImportError from inside a vendored file.

    A successful import is **not** sufficient, which an earlier version of this
    assumed. ``onnxruntime`` and ``onnxruntime-gpu`` install into the *same*
    ``onnxruntime/`` directory, so uninstalling one removes files the other needs --
    including ``__init__.py``. What is left is a directory Python imports happily as a
    namespace package: ``import onnxruntime`` succeeds, ``__file__`` is None, and the
    first real call fails with ``module 'onnxruntime' has no attribute
    'SessionOptions'`` from inside vendored code. Observed in the wild after
    `pip uninstall onnxruntime` was run while ComfyUI still held the DLLs open, which
    also leaves ``~ackend``-style stubs behind where pip could not delete a locked
    file. So the attributes are checked, not just the import.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError(
            f"{LOG_PREFIX} ONNX Runtime is not installed, and the CosyVoice speech "
            f"tokenizer cannot run without it.\n{INSTALL_HINT}"
        ) from exc

    missing = [name for name in _REQUIRED_ATTRIBUTES if not hasattr(onnxruntime, name)]
    if missing:
        location = getattr(onnxruntime, "__file__", None)
        where = list(getattr(onnxruntime, "__path__", []) or [])
        raise RuntimeError(
            f"{LOG_PREFIX} ONNX Runtime imports but is not usable: it is missing "
            f"{', '.join(missing)}.\n"
            f"  Module file: {location!r}"
            + (f"\n  Directory:   {where[0]}" if where else "")
            + (
                "\n  A module file of None means Python found the directory but no "
                "__init__.py, i.e. a half-removed install rather than a working one."
                if location is None
                else ""
            )
            + "\n  This is what installing both variants, or uninstalling one of them, "
            "leaves behind: they share the same onnxruntime/ directory, so removing "
            "either deletes files the other needs.\n"
            "  Fix it with ComfyUI fully stopped, or the files stay locked and the "
            "reinstall is partial again:\n"
            "    pip uninstall -y onnxruntime onnxruntime-gpu\n"
            "    (delete any leftover site-packages/onnxruntime/ directory)\n"
            "    pip install onnxruntime-gpu      # or onnxruntime, for CPU only"
        )

    return available_providers()


def _cuda_expected() -> bool:
    """Whether the vendored frontend will ask for the CUDA provider."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch is a ComfyUI hard requirement
        return False


def report_provider_status(force: bool = False) -> dict[str, object]:
    """
    Log what ONNX Runtime can do here, warning when CUDA was expected but is absent.

    Returns the same facts as a dict so callers and tests can assert on them rather
    than parsing log output.
    """
    global _ANNOUNCED

    providers = available_providers()
    cuda_expected = _cuda_expected()
    cuda_available = CUDA_PROVIDER in providers
    status = {
        "providers": providers,
        "cuda_expected": cuda_expected,
        "cuda_available": cuda_available,
        "degraded": bool(cuda_expected and not cuda_available),
    }

    if _ANNOUNCED and not force:
        return status

    if not providers:
        LOGGER.warning("%s ONNX Runtime is not installed. %s", LOG_PREFIX, INSTALL_HINT)
    elif status["degraded"]:
        LOGGER.warning(
            "%s Torch reports CUDA, but ONNX Runtime does not offer %s - available: "
            "%s. The vendored speech tokenizer asks for the CUDA provider and ONNX "
            "Runtime will silently fall back to the CPU, so reference-audio encoding "
            "runs on the CPU on a machine that has a GPU. This usually means the "
            "CPU-only `onnxruntime` package is installed; `pip uninstall onnxruntime` "
            "then `pip install onnxruntime-gpu` fixes it.",
            LOG_PREFIX,
            CUDA_PROVIDER,
            ", ".join(providers),
        )
    else:
        LOGGER.info(
            "%s Execution providers: %s%s",
            LOG_PREFIX,
            ", ".join(providers),
            "" if cuda_expected else " (torch reports no CUDA, so CPU is expected here)",
        )

    _ANNOUNCED = True
    return status


def reset_for_tests() -> None:
    """Forget that the status was already reported."""
    global _ANNOUNCED
    _ANNOUNCED = False
