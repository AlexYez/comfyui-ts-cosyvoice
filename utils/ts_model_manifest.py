"""
Content verification for downloaded model files, against ``model_manifest.json``.

Why this exists: the pack hands ``cosyvoice3.yaml`` to HyperPyYAML, whose ``!new:``
and ``!apply:`` tags import modules and construct objects. That file is *executed*.
The checks that were here before — file sizes, ZIP CRCs of the torch archives, and
"does onnxruntime open this graph" — establish that a download completed intact.
None of them establish that it is the download we vetted, and until now the model
was fetched from a mutable branch (``main``/``master``).

Two policies, because they answer different questions:

``config``
    Files that are executed, imported, or parsed as configuration before any tensor
    is touched. A mismatch is refused outright — this is the code-execution path,
    and "probably fine" is not a useful answer there.

``weights``
    Tensor and graph payloads. A mismatch is reported loudly and execution
    continues. Someone whose copy predates the pinned revision has a working
    install; taking it away would be a worse outcome than telling them clearly that
    their weights are not the pinned ones.

A refusal here deliberately does **not** delete anything. ``get_model_path`` treats
a validation *error string* as grounds to wipe the model directory and re-download,
so a hash mismatch is raised as ``ModelIntegrityError`` instead — a user whose model
is merely newer than the manifest must not have several gigabytes deleted from under
them by an integrity check.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from typing import Any

try:
    from .ts_logging import get_logger
except ImportError:  # loaded as a top-level module
    from ts_logging import get_logger

LOGGER = get_logger("TS CosyVoice3 Model Manifest")
LOG_PREFIX = "[TS CosyVoice3 Model Manifest]"

MANIFEST_NAME = "model_manifest.json"
CONFIG_POLICY = "config"
WEIGHTS_POLICY = "weights"

_READ_CHUNK = 8 * 1024 * 1024


class ModelIntegrityError(RuntimeError):
    """A file that gets executed does not match the manifest."""


def manifest_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MANIFEST_NAME)


@lru_cache(maxsize=1)
def _read_manifest() -> tuple[dict[str, Any], str | None]:
    """
    Return ``(manifest, unavailable_reason)``.

    Reading never raises, so importing the pack cannot fail on a damaged manifest and
    ComfyUI still registers the nodes. The *reason* is carried alongside so that the
    place which actually needs the guarantee can refuse with it — see
    :func:`require_manifest_entry`. Returning a bare empty dict, as this used to, made
    "the manifest is broken" indistinguishable from "there is nothing to check", and
    the load continued either way.
    """
    path = manifest_path()
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except OSError as exc:
        return {}, f"{MANIFEST_NAME} could not be read ({exc.__class__.__name__}: {exc})"
    except ValueError as exc:
        return {}, f"{MANIFEST_NAME} is not valid JSON ({exc})"
    if not isinstance(manifest, dict):
        return {}, f"{MANIFEST_NAME} does not contain a JSON object at the top level"
    if not isinstance(manifest.get("models"), dict):
        return {}, f"{MANIFEST_NAME} has no 'models' object"
    return manifest, None


def load_manifest() -> dict[str, Any]:
    """The manifest contents, or an empty dict if it could not be read."""
    return _read_manifest()[0]


def manifest_unavailable_reason() -> str | None:
    """Why the manifest cannot be used, or None when it is fine."""
    return _read_manifest()[1]


def require_manifest_entry(model_version: str) -> dict[str, Any]:
    """
    Return the manifest entry for this model, or refuse.

    This is the fail-closed gate. The manifest ships *with the pack*, so it being
    absent, malformed, or silent about a model the pack offers is not a normal
    condition — it means the install is damaged or has been tampered with. Continuing
    in that state would load and execute ``cosyvoice3.yaml`` with no idea whether it
    is the file that was vetted, which is precisely the guarantee this file exists to
    provide.
    """
    reason = manifest_unavailable_reason()
    if reason is None:
        entry = _model_entry(model_version)
        if entry.get("files"):
            return entry
        reason = f"{MANIFEST_NAME} has no file hashes for model version {model_version!r}"

    raise ModelIntegrityError(
        f"{LOG_PREFIX} Refusing to load: the model cannot be verified.\n"
        f"  {reason}\n"
        f"  The manifest is part of the pack and records the sha256 of every model "
        f"file that gets read, including cosyvoice3.yaml — which HyperPyYAML executes "
        f"via its !new: and !apply: tags. Loading without it would mean trusting "
        f"whatever is on disk.\n"
        f"  Reinstall or re-pull the pack to restore {MANIFEST_NAME}. If you are adding "
        f"a new model version, add it with tools/generate_model_manifest.py first."
    )


def _model_entry(model_version: str) -> dict[str, Any]:
    entry = load_manifest().get("models", {}).get(model_version)
    return entry if isinstance(entry, dict) else {}


def _source_key(download_source: str) -> str:
    return "modelscope" if str(download_source).strip().lower() == "modelscope" else "huggingface"


def pinned_revision(model_version: str, download_source: str) -> str | None:
    """
    The revision to download, or None to let the client choose its default.

    HuggingFace is pinned to a full commit hash. ModelScope publishes no immutable
    ref for this repository, so it stays on ``master`` and the hashes below are what
    make that download trustworthy — see ``revision_note`` in the manifest.
    """
    source = _model_entry(model_version).get(_source_key(download_source))
    if not isinstance(source, dict):
        return None
    revision = source.get("revision")
    return revision if isinstance(revision, str) and revision else None


def pinned_repo_id(model_version: str, download_source: str) -> str | None:
    source = _model_entry(model_version).get(_source_key(download_source))
    if not isinstance(source, dict):
        return None
    repo_id = source.get("repo_id")
    return repo_id if isinstance(repo_id, str) and repo_id else None


def manifest_files(model_version: str, policy: str | None = None) -> dict[str, dict]:
    files = _model_entry(model_version).get("files")
    if not isinstance(files, dict):
        return {}
    if policy is None:
        return files
    return {
        name: spec
        for name, spec in files.items()
        if isinstance(spec, dict) and spec.get("policy") == policy
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _mismatches(model_root: str, model_version: str, policy: str) -> list[tuple[str, str, str]]:
    """
    Hash every manifest file of this policy that is present on disk.

    Files listed in the manifest but absent are skipped rather than reported: not
    every entry is required for every configuration (``llm.rl.pt`` only matters when
    the reinforcement-learning checkpoint is selected). Missing *required* files are
    already caught by the size checks in ``validate_model_root``.
    """
    found: list[tuple[str, str, str]] = []
    for name, spec in sorted(manifest_files(model_version, policy).items()):
        expected = spec.get("sha256")
        if not isinstance(expected, str) or not expected:
            continue
        path = os.path.join(model_root, name.replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        try:
            actual = sha256_file(path)
        except OSError as exc:
            LOGGER.warning("%s Could not read %s for verification: %s", LOG_PREFIX, name, exc)
            continue
        if actual != expected:
            found.append((name, expected, actual))
    return found


def verify_config_files(model_root: str, model_version: str) -> int:
    """
    Verify the executed and parsed files. Raises on any mismatch.

    Cheap enough to run on every load: the whole ``config`` set is a few megabytes,
    dominated by the tokenizer vocabulary.

    Returns the number of files checked, so callers can say so rather than implying
    a verification that never happened.
    """
    require_manifest_entry(model_version)

    entries = manifest_files(model_version, CONFIG_POLICY)
    if not entries:
        # A manifest that exists but describes no executed files is not a pass. The
        # previous version returned 0 here and the caller carried on, so an entry
        # stripped of its `config` section silently disabled the check it exists for.
        raise ModelIntegrityError(
            f"{LOG_PREFIX} Refusing to load: {MANIFEST_NAME} lists no configuration "
            f"files for {model_version!r}, so the file HyperPyYAML executes "
            f"(cosyvoice3.yaml) cannot be verified. Regenerate the manifest with "
            f"tools/generate_model_manifest.py."
        )

    missing = [
        name
        for name in entries
        if not os.path.isfile(os.path.join(model_root, name.replace("/", os.sep)))
    ]
    if missing:
        raise ModelIntegrityError(
            f"{LOG_PREFIX} Refusing to load: these files are required and verified, "
            f"but are not present in {model_root}:\n"
            + "".join(f"    {name}\n" for name in sorted(missing))
            + "  A partial model directory can result from an interrupted download or "
            "from two download sources being mixed together. Remove the model folder "
            "so it is fetched again at the pinned revision."
        )

    bad = _mismatches(model_root, model_version, CONFIG_POLICY)
    if bad:
        details = "\n".join(
            f"    {name}\n      expected sha256 {expected}\n      found    sha256 {actual}"
            for name, expected, actual in bad
        )
        raise ModelIntegrityError(
            f"{LOG_PREFIX} Refusing to load: model configuration does not match the "
            f"manifest.\n"
            f"{details}\n"
            f"  These files are executed, not just read: cosyvoice3.yaml is resolved by "
            f"HyperPyYAML, whose !new: and !apply: tags import modules and construct "
            f"objects.\n"
            f"  Nothing has been deleted. Either:\n"
            f"    - remove the model folder so it is re-downloaded at the pinned "
            f"revision, or\n"
            f"    - if you are intentionally moving to a newer model release, "
            f"regenerate the manifest with tools/generate_model_manifest.py and review "
            f"the diff."
        )

    present = sum(
        1
        for name in entries
        if os.path.isfile(os.path.join(model_root, name.replace("/", os.sep)))
    )
    return present


# PyTorch made `weights_only=True` the default for torch.load in 2.6. Before that the
# default ran the pickle machinery, which can execute arbitrary code while
# deserialising. The vendored loader calls torch.load with no weights_only argument
# (cosyvoice/cli/model.py, cosyvoice/cli/frontend.py), and those files are downloaded,
# so on an older torch a tampered checkpoint would be code execution rather than a
# corrupt tensor. Patching the vendored tree is avoided elsewhere in this pack, so
# this is enforced as a precondition instead.
SAFE_TORCH_LOAD_VERSION = (2, 6)


def _torch_version_tuple() -> tuple[int, int]:
    import torch

    parts = str(torch.__version__).split("+")[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):  # pragma: no cover - unparseable dev build
        return (0, 0)


def require_safe_torch_load() -> None:
    """
    Refuse to load remote checkpoints on a PyTorch that unpickles them by default.
    """
    import torch

    if _torch_version_tuple() >= SAFE_TORCH_LOAD_VERSION:
        return
    raise ModelIntegrityError(
        f"{LOG_PREFIX} Refusing to load: PyTorch {torch.__version__} deserialises "
        f"torch.load with the pickle machinery unless weights_only=True is passed, and "
        f"the vendored CosyVoice loader does not pass it. These checkpoints are "
        f"downloaded, so on this version a tampered file could execute code inside "
        f"ComfyUI rather than merely fail.\n"
        f"  PyTorch {SAFE_TORCH_LOAD_VERSION[0]}.{SAFE_TORCH_LOAD_VERSION[1]} or newer "
        f"defaults to weights_only=True, which closes this. Upgrade the torch in the "
        f"Python that runs ComfyUI."
    )


def validate_tensors_only(path: str, description: str) -> None:
    """
    Prove a ``.pt`` file contains nothing but data, by loading it safely first.

    For files the manifest cannot cover. ``spk2info.pt`` is the case that matters: the
    upstream repository does not ship it, this pack creates an empty one when it is
    absent, and the vendored frontend ``torch.load``s whatever is there. That leaves a
    file nobody vetted on the load path, so it is parsed here with
    ``weights_only=True`` — which refuses anything that would need pickle — before the
    vendored code touches it.
    """
    import torch

    if not os.path.isfile(path):
        return
    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelIntegrityError(
            f"{LOG_PREFIX} Refusing to load: {description} ({os.path.basename(path)}) "
            f"is not a plain data file.\n"
            f"  Loading it with weights_only=True failed: {exc}\n"
            f"  This file is not covered by the manifest and is read directly by the "
            f"CosyVoice frontend, so anything that needs pickle to deserialise is "
            f"refused here. Delete it; the pack recreates an empty one."
        ) from exc


def verify_weight_files(model_root: str, model_version: str) -> list[str]:
    """
    Verify tensor and graph payloads. Warns; never raises.

    Expensive (several gigabytes), so callers should record the result and not
    repeat it while the files are unchanged.
    """
    bad = _mismatches(model_root, model_version, WEIGHTS_POLICY)
    for name, expected, actual in bad:
        LOGGER.warning(
            "%s %s does not match the manifest (expected sha256 %s, found %s). This is "
            "not the pinned release; it may be a newer model or an incomplete "
            "download. Loading anyway.",
            LOG_PREFIX,
            name,
            expected[:16],
            actual[:16],
        )
    return [name for name, _expected, _actual in bad]
