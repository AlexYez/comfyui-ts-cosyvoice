"""
Whisper auto-transcription helpers for TS CosyVoice3.

Whisper is optional: if it is missing or transcription fails, the callers fall back
to an empty reference text rather than failing the node (§9).

Model-family detection lives in `ts_cosyvoice_adapter.is_cosyvoice3_model_info` —
this module used to carry a second, subtly different copy of it.
"""

import os

try:
    from .ts_logging import get_logger
except (ImportError, ValueError):
    from ts_logging import get_logger


_WHISPER_MODEL = None
LOGGER = get_logger("TS CosyVoice3 Whisper")


def get_whisper_download_dir() -> str:
    import folder_paths

    whisper_dir = os.path.join(folder_paths.models_dir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    return whisper_dir


def get_whisper_model(log_prefix: str):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper

            LOGGER.info("[%s] Loading Whisper model for auto-transcription...", log_prefix)
            _WHISPER_MODEL = whisper.load_model("base", download_root=get_whisper_download_dir())
            LOGGER.info("[%s] Whisper model loaded successfully", log_prefix)
        except Exception as e:
            LOGGER.error("[%s] Failed to load Whisper: %s", log_prefix, e)
            return None
    return _WHISPER_MODEL


def transcribe_audio(audio_path: str, log_prefix: str) -> str:
    try:
        model = get_whisper_model(log_prefix)
        if model is None:
            return ""

        result = model.transcribe(audio_path, language=None)
        transcript = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        LOGGER.info("[%s] Whisper detected language: %s", log_prefix, detected_lang)
        return transcript
    except Exception as e:
        LOGGER.error("[%s] Whisper transcription failed: %s", log_prefix, e)
        return ""
