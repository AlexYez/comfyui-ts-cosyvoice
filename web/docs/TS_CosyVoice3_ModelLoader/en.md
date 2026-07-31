# TS CosyVoice Model Loader

Downloads (once) and loads a CosyVoice model, then hands it to every other node in the pack through the COSYVOICE_MODEL output. Put one loader in the graph and fan its output out to all synthesis nodes — the model is cached in memory, so a second loader with the same settings costs nothing extra.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model_version` | COMBO | yes | `Fun-CosyVoice3-0.5B` | Which CosyVoice model to load. Options: `Fun-CosyVoice3-0.5B`. |
| `download_source` | COMBO | yes | `HuggingFace` | Where to download the weights from on first use. If the chosen source fails, the other one is tried automatically. Options: `HuggingFace`, `ModelScope`. |
| `device` | COMBO | yes | `auto` | Preferred compute device, best effort. The CosyVoice runtime places the model itself and uses the GPU when one is available, so this request may not be honoured literally. Options: `auto`, `cuda`, `cpu`, `mps`. |
| `fp16` | BOOLEAN | no | `false` | Load in half precision on supported accelerators to cut VRAM use. Ignored without CUDA. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `model` | COSYVOICE_MODEL | The loaded model, to be wired into the pack's synthesis nodes. |

## Notes

- Weights land in `ComfyUI/models/cosyvoice/<model_version>/` (about 1.5 GB). The first run downloads them; later runs reuse them.
- A downloaded model is checked for truncated or corrupted files. The result of that check is recorded next to the weights, so the expensive part runs once rather than on every ComfyUI start.
- The device selector is a request, not a guarantee — read the node's log line to see where the model actually ended up.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
