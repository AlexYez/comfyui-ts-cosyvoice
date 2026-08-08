# Third-party notices

This package redistributes source code from other projects. `LICENSE` at the root
covers only the code written for this package (the `nodes/`, `utils/` and `tools/`
directories, `__init__.py`, and the documentation). It does **not** cover the
vendored trees described below, which remain under their own licences and belong to
their own copyright holders.

Full licence texts are in [`licenses/`](licenses/).

| Vendored path | Upstream project | Licence | Text |
|---|---|---|---|
| `cosyvoice/` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Apache License 2.0 | [`licenses/LICENSE.CosyVoice-Apache-2.0.txt`](licenses/LICENSE.CosyVoice-Apache-2.0.txt) |
| `matcha/` | [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | MIT | [`licenses/LICENSE.Matcha-TTS-MIT.txt`](licenses/LICENSE.Matcha-TTS-MIT.txt) |
| `configs/` | mixed — see below | — | — |

## `cosyvoice/` — Apache License 2.0

Vendored from https://github.com/FunAudioLLM/CosyVoice.

This tree is not the work of a single party. It carries Apache-2.0 headers in 37
source files, and the copyright notices found inside it are:

- Copyright (c) 2019 Shigeki Karita
- Copyright (c) 2020 Johns Hopkins University (Shinji Watanabe)
- Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
- Copyright (c) 2020 Mobvoi Inc (Di Wu)
- Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
- Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
- Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
- Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
- Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
- Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
- Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
- Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)

The Karita / Johns Hopkins / Mobvoi notices are the ESPnet and WeNet lineage that
CosyVoice's transformer and encoder code descends from; upstream carries them
forward, and so does this copy.

Apache-2.0 §4 requires that redistribution retain the licence, the copyright
notices, and any NOTICE file. The licence text is included, the per-file headers are
preserved verbatim in the vendored files, and upstream CosyVoice ships no NOTICE
file.

## `matcha/` — MIT

Vendored from https://github.com/shivammehta25/Matcha-TTS.

Copyright (c) 2023 Shivam Mehta.

The vendored `matcha/` files carry no per-file licence headers, which is why the
full MIT text with its copyright line is included in `licenses/` — the MIT licence
requires the notice to travel with the code, and here there is nowhere else for it
to live.

## `configs/`

Local configuration written for this package, plus configuration consumed by the
vendored trees. Nothing here is a third-party source file.

## Modifications

The vendored trees are kept as close to upstream as possible so that upstream fixes
can be pulled in cheaply. Any local change is required to carry a comment of the
form:

```python
# TS-PATCH: <why>; revert if upstream fixes <reference>.
```

so that `grep -rn "TS-PATCH" cosyvoice/ matcha/` lists every deviation. As of this
release that search returns nothing: both trees are unmodified.

Notably, this package does **not** patch the vendored code to remove its
module-level `whisper` and `librosa.filters` imports. It installs replacements in
`sys.modules` from `utils/ts_runtime_shims.py` instead, precisely so that the
vendored files stay byte-identical to upstream.

## Provenance, stated honestly

The exact upstream commit each tree was copied from **is not recorded in this
repository**. Both trees arrived in a single commit (`dbd070b`, "Initial import of TS
CosyVoice custom nodes") with no upstream revision noted, and neither tree contains a
version marker. What can be said from the code itself is that `cosyvoice/` includes
CosyVoice 3 support (`cosyvoice/cli/cosyvoice.py` defines `CosyVoice3`) and targets
the `Fun-CosyVoice3-0.5B-2512` release.

Rather than print a commit hash that has not been verified, this is left as a known
gap. It will be recorded properly the next time either tree is re-synced from
upstream.

## Model weights

Model weights are **not** redistributed here. They are downloaded at runtime from
the upstream repository and are covered by that repository's own terms:

- https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

The revision fetched and the sha256 of every file this package reads are recorded in
[`model_manifest.json`](model_manifest.json).
