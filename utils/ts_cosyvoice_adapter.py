import contextlib
from typing import Any

try:
    from .ts_logging import get_logger
except (ImportError, ValueError):
    from ts_logging import get_logger

SYSTEM_PROMPT = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"

LOGGER = get_logger("TS CosyVoice3 Adapter")

# Installed on the vendored flow decoder. Prefixed so they cannot collide with
# anything upstream adds later.
_STEPS_OVERRIDE_ATTR = "_ts_n_timesteps_override"
_ORIGINAL_FORWARD_ATTR = "_ts_original_forward"

# The literal the vendored flow decoder is called with (cosyvoice/flow/flow.py).
# Asking for exactly this is a no-op, so the wrapper is not installed for it.
VENDORED_DIFFUSION_STEPS = 10


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


def get_flow_decoder(cosyvoice_model: Any) -> Any:
    """
    Return the conditional flow-matching decoder, or None if it cannot be reached.

    The path is ``model.flow.decoder`` inside the vendored runtime. It is resolved
    defensively rather than assumed: if upstream reorganises this, the quality
    knobs below must degrade to "use the model's defaults", never crash a
    conversion that would otherwise have worked.
    """
    decoder = getattr(getattr(getattr(cosyvoice_model, "model", None), "flow", None), "decoder", None)
    if decoder is None:
        LOGGER.warning(
            "[TS CosyVoice3 Adapter] Could not reach model.flow.decoder; "
            "diffusion-step and CFG overrides will be ignored"
        )
    return decoder


def _install_steps_override(decoder: Any) -> bool:
    """
    Make the decoder read its step count from an attribute we control.

    ``n_timesteps`` is passed as a literal at the call site in the vendored
    ``flow.py``, so there is no attribute to set and no argument to thread
    through. Rather than patch vendored code — which would have to be re-applied
    on every upstream pull — the decoder's own ``forward`` is wrapped once.
    ``torch.nn.Module.__call__`` resolves ``self.forward`` through the instance
    dict, so the wrapper is picked up by the normal call path.
    """
    if getattr(decoder, _ORIGINAL_FORWARD_ATTR, None) is not None:
        return True

    original_forward = getattr(decoder, "forward", None)
    if original_forward is None:
        return False

    def _forward_with_override(*args: Any, **kwargs: Any) -> Any:
        override = getattr(decoder, _STEPS_OVERRIDE_ATTR, None)
        if override:
            if "n_timesteps" in kwargs:
                kwargs["n_timesteps"] = override
            elif len(args) >= 3:
                # Positional (mu, mask, n_timesteps, ...) — kept for robustness;
                # the current vendored call site uses keywords.
                args = (args[0], args[1], override, *args[3:])
            else:
                kwargs["n_timesteps"] = override
        return original_forward(*args, **kwargs)

    setattr(decoder, _ORIGINAL_FORWARD_ATTR, original_forward)
    decoder.forward = _forward_with_override
    return True


@contextlib.contextmanager
def flow_inference_overrides(
    cosyvoice_model: Any,
    diffusion_steps: int | None = None,
    cfg_rate: float | None = None,
):
    """
    Temporarily raise the flow decoder's quality settings for one inference.

    Both values are properties of the shared, module-cached model instance, so
    they MUST be scoped: leaving them set would silently change every other node
    in the graph that uses the same loader output. Everything is restored on the
    way out, including on failure and on cancellation.

    Args:
        cosyvoice_model: the loaded runtime (``model_info["model"]``).
        diffusion_steps: Euler steps for the flow decoder. The vendored default is
            10, which favours speed; more steps trade time for detail. ``None`` or
            a non-positive value leaves the default alone.
        cfg_rate: classifier-free guidance strength. The shipped configuration
            uses 0.7; higher follows the reference more closely at the risk of
            artefacts. ``None`` leaves it alone.

    Yields:
        True if the overrides were installed, False if the decoder was not
        reachable and the model's own defaults are in effect.
    """
    decoder = get_flow_decoder(cosyvoice_model)
    if decoder is None:
        yield False
        return

    # Only patch when there is something to change. The node's default equals the
    # vendored literal, so without this check every run permanently installed a
    # forward wrapper on the shared cached model for no effect.
    wants_steps = (
        diffusion_steps is not None
        and int(diffusion_steps) > 0
        and int(diffusion_steps) != VENDORED_DIFFUSION_STEPS
    )
    wants_cfg = cfg_rate is not None

    previous_cfg = getattr(decoder, "inference_cfg_rate", None)
    steps_installed = False

    try:
        if wants_steps:
            steps_installed = _install_steps_override(decoder)
            if steps_installed:
                setattr(decoder, _STEPS_OVERRIDE_ATTR, int(diffusion_steps))
                LOGGER.info(
                    "[TS CosyVoice3 Adapter] Flow decoder steps: %s (default 10)",
                    int(diffusion_steps),
                )
            else:
                LOGGER.warning(
                    "[TS CosyVoice3 Adapter] Could not override diffusion steps; "
                    "using the model default"
                )

        if wants_cfg and previous_cfg is not None:
            decoder.inference_cfg_rate = float(cfg_rate)
            LOGGER.info(
                "[TS CosyVoice3 Adapter] Flow decoder CFG rate: %s (was %s)",
                float(cfg_rate),
                previous_cfg,
            )
        elif wants_cfg:
            LOGGER.warning(
                "[TS CosyVoice3 Adapter] Decoder exposes no inference_cfg_rate; "
                "the CFG setting is ignored"
            )

        yield True
    finally:
        # Restore unconditionally: the model instance is shared through the cache.
        if steps_installed:
            setattr(decoder, _STEPS_OVERRIDE_ATTR, None)
        if wants_cfg and previous_cfg is not None:
            decoder.inference_cfg_rate = previous_cfg
