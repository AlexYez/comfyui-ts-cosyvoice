"""
Regenerate ``model_manifest.json`` from the upstream model repositories.

The manifest is what makes a downloaded model trustworthy. The pack feeds
``cosyvoice3.yaml`` to HyperPyYAML, which resolves ``!new:`` and ``!apply:`` tags by
importing modules and constructing objects — so that file is *executed*, not merely
parsed. Checking sizes, ZIP CRCs and "does ONNX open" says a download finished; it
says nothing about whether the download is the one we vetted.

Run this only when deliberately moving to a newer model release, and commit the
result as its own reviewable change:

    python tools/generate_model_manifest.py

Where the hashes come from, and why both mirrors are queried:

- ModelScope's file listing returns sha256 for *every* file, including the small
  non-LFS ones. That is the only source for ``cosyvoice3.yaml`` — the file that
  matters most here.
- HuggingFace exposes sha256 only through LFS metadata, so it covers the weights
  and ONNX graphs but not the small text files.

Every file present in both listings is cross-checked. A mismatch means the two
mirrors disagree about content and the run aborts rather than picking a winner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MANIFEST_PATH = os.path.join(REPO_ROOT, "model_manifest.json")

# Files whose contents are executed, imported or parsed as configuration before any
# weight is touched. A mismatch here is refused: this is the code-execution path.
CONFIG_FILES = (
    "cosyvoice3.yaml",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
    "CosyVoice-BlankEN/merges.txt",
)

# Tensor and graph payloads. Large, so they are hashed once and remembered; a
# mismatch warns rather than refuses, because a user whose copy predates the pinned
# revision has a working install and should not be locked out of it.
WEIGHT_FILES = (
    "llm.pt",
    "llm.rl.pt",
    "flow.pt",
    "hift.pt",
    "CosyVoice-BlankEN/model.safetensors",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "speech_tokenizer_v3.batch.onnx",
    "flow.decoder.estimator.fp32.onnx",
)

# Deliberately absent from the manifest:
#   README.md, .gitattributes, asset/       - never read by the pack, and the two
#                                             mirrors legitimately differ on them
#   config.json, configuration.json (root)  - registry metadata, not consumed here
#   spk2info.pt                             - created locally by the pack itself
MODELS = {
    "Fun-CosyVoice3-0.5B": {
        "huggingface_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "modelscope_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    }
}


def modelscope_hashes(repo_id: str) -> dict[str, tuple[str, int]]:
    from modelscope.hub.api import HubApi

    listing = HubApi().get_model_files(repo_id, revision="master", recursive=True)
    out: dict[str, tuple[str, int]] = {}
    for entry in listing:
        if entry.get("Type") != "blob":
            continue
        digest = entry.get("Sha256")
        if digest:
            out[entry["Path"]] = (digest, int(entry.get("Size") or 0))
    return out


def huggingface_hashes(repo_id: str) -> tuple[str, dict[str, tuple[str, int]]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    out: dict[str, tuple[str, int]] = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        if lfs is None:
            continue
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if digest:
            out[sibling.rfilename] = (digest, int(getattr(sibling, "size", 0) or 0))
    return info.sha, out


def build(model_version: str, ids: dict[str, str]) -> dict:
    modelscope = modelscope_hashes(ids["modelscope_id"])
    hf_commit, huggingface = huggingface_hashes(ids["huggingface_id"])

    conflicts = [
        path
        for path in set(modelscope) & set(huggingface)
        if modelscope[path][0] != huggingface[path][0]
    ]
    if conflicts:
        raise SystemExit(
            "ModelScope and HuggingFace disagree about the contents of: "
            + ", ".join(sorted(conflicts))
            + "\nRefusing to guess which mirror is authoritative."
        )
    print(f"  cross-checked {len(set(modelscope) & set(huggingface))} files across both mirrors")

    files: dict[str, dict] = {}
    missing: list[str] = []
    for policy, names in (("config", CONFIG_FILES), ("weights", WEIGHT_FILES)):
        for name in names:
            found = modelscope.get(name) or huggingface.get(name)
            if not found:
                missing.append(name)
                continue
            digest, size = found
            files[name] = {"sha256": digest, "size": size, "policy": policy}
    if missing:
        raise SystemExit(
            "No sha256 available for: " + ", ".join(missing) + "\nUpstream layout may have changed."
        )

    return {
        "huggingface": {"repo_id": ids["huggingface_id"], "revision": hf_commit},
        "modelscope": {
            "repo_id": ids["modelscope_id"],
            "revision": "master",
            "revision_note": (
                "ModelScope publishes no immutable ref for this repository -- its API "
                "reports ModelRevisions: None and the only branch is 'master'. The "
                "hashes below, not the revision, are what make a ModelScope download "
                "trustworthy."
            ),
        },
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the manifest is stale")
    args = parser.parse_args()

    models = {}
    for version, ids in MODELS.items():
        print(f"querying {version}...")
        models[version] = build(version, ids)
        print(f"  pinned HuggingFace commit: {models[version]['huggingface']['revision']}")
        print(f"  {len(models[version]['files'])} files recorded")

    manifest = {
        "_comment": (
            "sha256 for every model file the pack reads, verified before the model is "
            "loaded. cosyvoice3.yaml is executed by HyperPyYAML (!new:/!apply: import "
            "modules and construct objects), so 'config' files are refused on mismatch. "
            "'weights' files warn instead, so an install that predates the pinned "
            "revision keeps working. Regenerate with tools/generate_model_manifest.py "
            "and review the diff -- this file is the trust anchor."
        ),
        "manifest_version": 1,
        "models": models,
    }

    serialised = json.dumps(manifest, indent=2, sort_keys=False) + "\n"

    if args.check:
        existing = open(MANIFEST_PATH, encoding="utf-8").read() if os.path.isfile(MANIFEST_PATH) else ""
        if existing != serialised:
            print("\nmodel_manifest.json is out of date with upstream.", file=sys.stderr)
            return 1
        print("\nmanifest is current")
        return 0

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialised)
    print(f"\nwrote {os.path.relpath(MANIFEST_PATH, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
