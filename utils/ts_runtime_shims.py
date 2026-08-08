"""
Make the vendored CosyVoice runtime importable without numba.

The vendored tree reaches for numba-carrying packages at *module* level in three
places on the inference path, all of them for a mel spectrogram:

    cosyvoice/cli/frontend.py:20      import whisper
    cosyvoice/cli/frontend.py:125     whisper.log_mel_spectrogram(speech, n_mels=128)
    cosyvoice/tokenizer/tokenizer.py:7  from whisper.tokenizer import Tokenizer
    matcha/utils/audio.py:8           from librosa.filters import mel

numba refuses to import against NumPy newer than 2.1 and current ComfyUI ships
NumPy 2.5, so on an up-to-date install every one of those raises and the pack
cannot load a model at all.

The fix is deliberately *not* a patch to ``cosyvoice/`` or ``matcha/``. Those are
vendored upstream copies and keeping them byte-identical is what makes pulling
upstream fixes cheap. Instead this module registers stand-ins in ``sys.modules``
before the vendored code is imported, so the vendored import statements resolve
against ours. The replacements live in :mod:`utils.ts_mel`.

Two properties matter more than anything else here:

- **Inert when the real thing works.** If ``import whisper`` succeeds, nothing is
  registered and behaviour is exactly upstream's. A stub that shadowed a working
  whisper would silently break auto-transcription, which is a worse bug than the
  one being fixed.
- **Detectable.** Anything installed here carries :data:`SHIM_ATTRIBUTE`, so code
  that needs the *full* package (transcription needs whisper's decoder, not just
  its front-end) can tell a stand-in from the real thing and say so.

Only ``whisper``'s front-end and ``librosa.filters.mel`` are provided. The stubs
are not general-purpose: ask them for anything else and they raise.
"""

from __future__ import annotations

import sys
import types

from utils.ts_logging import get_logger

LOGGER = get_logger("TS CosyVoice3 Runtime")
LOG_PREFIX = "[TS CosyVoice3 Runtime]"

#: Marker attribute set on every module this file registers.
SHIM_ATTRIBUTE = "__ts_cosyvoice_shim__"

_INSTALLED: dict[str, bool] = {}


def is_shimmed(module_name: str) -> bool:
    """True if ``module_name`` in ``sys.modules`` is one of our stand-ins."""
    module = sys.modules.get(module_name)
    return module is not None and getattr(module, SHIM_ATTRIBUTE, False) is True


def _real_module_imports(module_name: str) -> bool:
    """
    Whether ``module_name`` can actually be imported here.

    Catches ``BaseException`` on purpose: the numba guard raises ``ImportError``,
    but a broken native extension elsewhere in the chain can surface as almost
    anything, and a failure to probe must never take down node registration.
    """
    if is_shimmed(module_name):
        return False
    try:
        __import__(module_name)
    except BaseException:  # see docstring
        return False
    return True


def _new_shim(module_name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(module_name)
    setattr(module, SHIM_ATTRIBUTE, True)
    module.__doc__ = (
        f"Minimal stand-in for {module_name!r} installed by comfyui-ts-cosyvoice "
        "so the CosyVoice runtime can load without numba. Not the real package."
    )
    if package:
        # A package needs __path__ for `import parent.child` to be attempted.
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_whisper_shim() -> None:
    """
    Register a ``whisper`` package exposing only what the vendored code touches.

    ``log_mel_spectrogram`` is the real work. ``whisper.tokenizer.Tokenizer`` is
    needed only so ``cosyvoice/tokenizer/tokenizer.py`` imports; it belongs to
    the CosyVoice1 ``get_tokenizer`` path, and CosyVoice3 builds its tokenizer
    through ``get_qwen_tokenizer`` -> ``AutoTokenizer``, so it is never
    constructed. It raises if it ever is, rather than pretending.
    """
    from utils import ts_mel

    whisper_module = _new_shim("whisper", package=True)
    whisper_module.log_mel_spectrogram = ts_mel.log_mel_spectrogram  # type: ignore[attr-defined]
    whisper_module.mel_filters = ts_mel.whisper_mel_filters  # type: ignore[attr-defined]
    whisper_module.N_FFT = ts_mel.N_FFT  # type: ignore[attr-defined]
    whisper_module.HOP_LENGTH = ts_mel.HOP_LENGTH  # type: ignore[attr-defined]
    whisper_module.SAMPLE_RATE = ts_mel.SAMPLE_RATE  # type: ignore[attr-defined]

    tokenizer_module = _new_shim("whisper.tokenizer")

    class Tokenizer:  # placeholder, see _install_whisper_shim
        """Unreachable on the CosyVoice3 path; present only to satisfy an import."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "[TS CosyVoice3] whisper's Tokenizer was requested, but only a "
                "stand-in is available because openai-whisper cannot be imported "
                "in this environment. This is the CosyVoice1 tokenizer path; "
                "CosyVoice3 does not use it. Install a working openai-whisper if "
                "you need it."
            )

    tokenizer_module.Tokenizer = Tokenizer  # type: ignore[attr-defined]
    whisper_module.tokenizer = tokenizer_module  # type: ignore[attr-defined]

    sys.modules["whisper"] = whisper_module
    sys.modules["whisper.tokenizer"] = tokenizer_module


def _install_librosa_filters_shim() -> None:
    """
    Register ``librosa.filters`` with a compatible ``mel``.

    Scoped to the submodule on purpose: ``import librosa`` succeeds on its own
    (librosa 0.10 resolves submodules lazily) and only ``librosa.filters`` pulls
    numba. If the real top-level package is importable it is left in place and
    only the one failing submodule is substituted, so anything else a caller
    reaches for is still genuinely librosa.
    """
    from utils import ts_mel

    if "librosa" not in sys.modules:
        try:
            __import__("librosa")
        except BaseException:  # fall back to a stub package
            sys.modules["librosa"] = _new_shim("librosa", package=True)

    filters_module = _new_shim("librosa.filters")
    filters_module.mel = ts_mel.librosa_compatible_mel  # type: ignore[attr-defined]

    sys.modules["librosa.filters"] = filters_module
    parent = sys.modules.get("librosa")
    if parent is not None:
        parent.filters = filters_module  # type: ignore[attr-defined]


def install_cosyvoice_import_shims() -> dict[str, bool]:
    """
    Ensure the vendored CosyVoice runtime can be imported, shimming only if needed.

    Safe and cheap to call repeatedly: each target is probed once and the result
    remembered, so this does not re-import packages on every model load.

    Returns:
        ``{"whisper": bool, "librosa.filters": bool}`` -- True where a stand-in
        is now in place, False where the real package is being used.
    """
    if not _INSTALLED:
        if _real_module_imports("whisper"):
            _INSTALLED["whisper"] = False
        else:
            _install_whisper_shim()
            _INSTALLED["whisper"] = True

        if _real_module_imports("librosa.filters"):
            _INSTALLED["librosa.filters"] = False
        else:
            _install_librosa_filters_shim()
            _INSTALLED["librosa.filters"] = True

        shimmed = sorted(name for name, used in _INSTALLED.items() if used)
        if shimmed:
            LOGGER.info(
                "%s %s could not be imported (commonly numba vs the installed "
                "NumPy); using the pack's built-in mel front-end instead. "
                "Synthesis and voice conversion are unaffected.",
                LOG_PREFIX,
                " and ".join(shimmed),
            )
            if "whisper" in shimmed:
                LOGGER.warning(
                    "%s automatic reference-text transcription needs a working "
                    "openai-whisper and stays unavailable; type reference_text "
                    "manually, or install openai-whisper against a NumPy it "
                    "supports (numba requires numpy<2.2).",
                    LOG_PREFIX,
                )

    return dict(_INSTALLED)


def reset_for_tests() -> None:
    """Forget the memoised probe results. Test-support only."""
    _INSTALLED.clear()
