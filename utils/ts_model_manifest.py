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
def load_manifest() -> dict[str, Any]:
    """
    Read the manifest, or return an empty one if it is missing or malformed.

    A pack without a readable manifest keeps working — verification is a guarantee
    added on top, not a new way for the pack to fail to start. The absence is
    logged once, because silently skipping an integrity check is exactly the kind
    of thing that should be visible.
    """
    path = manifest_path()
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except OSError:
        LOGGER.warning(
            "%s %s is missing; model files cannot be verified against known hashes.",
            LOG_PREFIX,
            MANIFEST_NAME,
        )
        return {}
    except ValueError as exc:
        LOGGER.warning(
            "%s %s is not valid JSON (%s); model files cannot be verified.",
            LOG_PREFIX,
            MANIFEST_NAME,
            exc,
        )
        return {}
    return manifest if isinstance(manifest, dict) else {}


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
    entries = manifest_files(model_version, CONFIG_POLICY)
    if not entries:
        return 0

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
