"""
Whisper auto-transcription helpers for TS CosyVoice3.

Whisper is optional: if it is missing or transcription fails, callers fall back
to an empty reference text rather than failing the node (§9). What changed is
that the failure is no longer silent — ``transcribe_audio_with_status`` hands the
reason back so a node can tell the user their preset was saved with a degraded
(empty) transcript instead of only whispering it into the log.

Transcripts are cached by audio *content*, not by path: every caller writes the
reference to a fresh temp file, so a path-keyed cache would never hit. The Dialog
node transcribes up to four references, and without this it re-transcribed all of
them on every single run even though the references had not changed.

Model-family detection lives in `ts_cosyvoice_adapter.is_cosyvoice3_model_info` —
this module used to carry a second, subtly different copy of it.
"""

import hashlib
import os

try:
    from .ts_logging import get_logger
except (ImportError, ValueError):
    from ts_logging import get_logger


_WHISPER_MODEL = None
LOGGER = get_logger("TS CosyVoice3 Whisper")

# Content hash -> transcript. Bounded because a long session can transcribe many
# different references; the reference clips themselves are never held here.
_TRANSCRIPT_CACHE: "dict[str, str]" = {}
_TRANSCRIPT_CACHE_LIMIT = 64

_READ_BLOCK_SIZE = 1 << 20


def get_whisper_download_dir() -> str:
    import folder_paths

    whisper_dir = os.path.join(folder_paths.models_dir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    return whisper_dir


def get_whisper_model(log_prefix: str):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # The model loader may have registered a stand-in for whisper so that the
        # CosyVoice runtime could import without numba. It provides the mel
        # front-end only -- there is no decoder behind it. Say that plainly
        # instead of letting `whisper.load_model` fail with an AttributeError.
        try:
            from .ts_runtime_shims import is_shimmed
        except ImportError:
            from ts_runtime_shims import is_shimmed

        if is_shimmed("whisper"):
            LOGGER.warning(
                "[%s] Automatic transcription is unavailable: openai-whisper "
                "cannot be imported in this environment (commonly numba vs the "
                "installed NumPy), so only the pack's built-in mel front-end is "
                "in place. Enter reference_text manually, or install "
                "openai-whisper against a NumPy numba supports (numpy<2.2). "
                "Synthesis itself is unaffected.",
                log_prefix,
            )
            return None

        try:
            import whisper

            LOGGER.info("[%s] Loading Whisper model for auto-transcription...", log_prefix)
            _WHISPER_MODEL = whisper.load_model("base", download_root=get_whisper_download_dir())
            LOGGER.info("[%s] Whisper model loaded successfully", log_prefix)
        except Exception as e:
            LOGGER.error("[%s] Failed to load Whisper: %s", log_prefix, e)
            return None
    return _WHISPER_MODEL


def _audio_content_key(audio_path: str) -> "str | None":
    """Hash the audio bytes; returns None when the file cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(audio_path, "rb") as handle:
            for block in iter(lambda: handle.read(_READ_BLOCK_SIZE), b""):
                digest.update(block)
    except OSError as e:
        LOGGER.debug("[TS CosyVoice3 Whisper] Could not hash %s: %s", os.path.basename(audio_path), e)
        return None
    return digest.hexdigest()


def _remember(content_key: "str | None", transcript: str) -> None:
    if not content_key or not transcript:
        return
    if len(_TRANSCRIPT_CACHE) >= _TRANSCRIPT_CACHE_LIMIT:
        _TRANSCRIPT_CACHE.pop(next(iter(_TRANSCRIPT_CACHE)))
    _TRANSCRIPT_CACHE[content_key] = transcript


def transcribe_audio_with_status(audio_path: str, log_prefix: str) -> "tuple[str, str | None]":
    """
    Transcribe reference audio.

    Returns:
        ``(transcript, failure_reason)``. ``failure_reason`` is None on success;
        otherwise it is a short, user-facing explanation of why the transcript is
        empty, so the calling node can surface it instead of silently saving a
        weaker preset.
    """
    content_key = _audio_content_key(audio_path)
    if content_key is not None:
        cached = _TRANSCRIPT_CACHE.get(content_key)
        if cached is not None:
            LOGGER.info("[%s] Reusing cached transcript for identical reference audio", log_prefix)
            return cached, None

    model = get_whisper_model(log_prefix)
    if model is None:
        return "", (
            "Whisper is not available, so the reference text could not be transcribed "
            "automatically. Install it with: pip install openai-whisper"
        )

    try:
        result = model.transcribe(audio_path, language=None)
    except Exception as e:
        LOGGER.error("[%s] Whisper transcription failed: %s", log_prefix, e)
        return "", f"Whisper transcription failed: {e}"

    transcript = str(result.get("text", "")).strip()
    detected_lang = result.get("language", "unknown")
    LOGGER.info("[%s] Whisper detected language: %s", log_prefix, detected_lang)

    if not transcript:
        return "", "Whisper produced an empty transcript for this reference audio."

    _remember(content_key, transcript)
    return transcript, None


def transcribe_audio(audio_path: str, log_prefix: str) -> str:
    """Transcribe reference audio, returning an empty string on any failure."""
    transcript, _ = transcribe_audio_with_status(audio_path, log_prefix)
    return transcript
