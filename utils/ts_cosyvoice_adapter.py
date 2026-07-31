from typing import Any

SYSTEM_PROMPT = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"


def is_cosyvoice3_model_info(model_info: dict[str, Any]) -> bool:
    """Detect whether the loaded runtime points to a CosyVoice3 family model."""
    version = str(model_info.get("model_version", "")).lower()
    return bool(model_info.get("is_cosyvoice3")) or "cosyvoice3" in version or "fun-cosyvoice3" in version


def get_runtime_device(cosyvoice_model: Any) -> Any:
    """
    Return the device the loaded model actually runs on.

    The vendored CosyVoice runtime hardcodes ``cuda`` when it is available and
    ignores the loader's device request (``AutoModel`` does not accept a device
    argument; ``frontend``/``model`` pin their own device). So the frontend's
    ``device`` is the only truthful source for placing preset tensors and reading
    the effective device — the loader's requested value can disagree with it.

    Falls back to ``"cpu"`` if the attribute is unavailable.
    """
    frontend = getattr(cosyvoice_model, "frontend", None)
    device = getattr(frontend, "device", None)
    if device is not None:
        return device
    return "cpu"


def format_instruct_text(instruct_text: str, is_cosyvoice3: bool) -> str:
    """Format instruct text according to the active CosyVoice generation API."""
    raw_instruct = instruct_text.strip()
    if raw_instruct.startswith(SYSTEM_PROMPT):
        raw_instruct = raw_instruct[len(SYSTEM_PROMPT):].lstrip("\n")
    if raw_instruct.endswith(END_OF_PROMPT):
        raw_instruct = raw_instruct[:-len(END_OF_PROMPT)].rstrip()

    if is_cosyvoice3:
        return f"{SYSTEM_PROMPT}\n{raw_instruct}{END_OF_PROMPT}"
    return f"{raw_instruct}{END_OF_PROMPT}"


def format_cross_lingual_text(text: str, is_cosyvoice3: bool, target_language: str) -> str:
    """Format cross-lingual synthesis text for the selected runtime generation path."""
    if is_cosyvoice3:
        return f"{SYSTEM_PROMPT}{END_OF_PROMPT}{text}"

    if target_language == "auto":
        return text

    lang_tags = {
        "en": "<|en|>",
        "zh": "<|zh|>",
        "ja": "<|jp|>",
        "ko": "<|ko|>",
        "yue": "<|yue|>",
        "de": "<|de|>",
        "es": "<|es|>",
        "fr": "<|fr|>",
        "it": "<|it|>",
        "ru": "<|ru|>",
    }
    return f"{lang_tags.get(target_language, '')}{text}"


def apply_speaker_prompt_tokens(cosyvoice_model: Any, spk_id: str, formatted_instruct: str) -> None:
    """Write formatted instruct tokens into a loaded speaker preset entry."""
    prompt_text_token, prompt_text_token_len = cosyvoice_model.frontend._extract_text_token(formatted_instruct)
    cosyvoice_model.frontend.spk2info[spk_id]["prompt_text"] = prompt_text_token
    cosyvoice_model.frontend.spk2info[spk_id]["prompt_text_len"] = prompt_text_token_len
