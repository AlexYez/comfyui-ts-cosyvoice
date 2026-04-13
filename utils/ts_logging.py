import logging
from typing import Any


_FORMAT = "%(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    logger.error("%s: %s", message, exc)
    logger.debug("Traceback", exc_info=exc)


def preview_text(value: str, limit: int = 50) -> str:
    return f"{value[:limit]}..." if len(value) > limit else value


def log_banner(logger: logging.Logger, title: str, **fields: Any) -> None:
    logger.info("%s", "=" * 60)
    logger.info("%s", title)
    for key, value in fields.items():
        logger.info("%s: %s", key, value)
    logger.info("%s", "=" * 60)
