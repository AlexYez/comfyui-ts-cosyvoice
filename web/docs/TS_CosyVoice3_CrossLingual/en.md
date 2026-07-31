# TS CosyVoice Cross-Language

Speaks text in a different language while keeping the timbre of the reference voice. The reference may be in one language and the text in another — the voice carries over, the accent follows the target language.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | COSYVOICE_MODEL | yes | — | The loaded CosyVoice model, from the TS CosyVoice Model Loader node. |
| `text` | STRING | yes | `Hello, this is cross-lingual speech synthesis.` | The text to speak, written in the target language. |
| `reference_audio` | AUDIO | yes | — | Voice to clone. Trimmed to 30 seconds and converted to mono 24 kHz; 3-10 seconds of clean speech gives the best timbre. |
| `speed` | FLOAT | yes | `1.0` | Playback speed multiplier applied to the generated speech. Range: `0.5`–`2.0`, step `0.05`. |
| `target_language` | COMBO | no | `auto` | Target language of the text. auto lets the model infer it, which is usually right for unambiguous text. Options: `auto`, `zh`, `en`, `ja`, `ko`, `de`, and 4 more. |
| `seed` | INT | no | `42` | Random seed. Set it to -1 for a fresh take on every run: the node then re-runs instead of returning its cached audio. Any value from 0 up is deterministic — re-running with everything unchanged returns the previous result, so change the seed (or use control_after_generate: randomize) when you want a different take from a fixed seed. Range: `-1`–`2147483647`. |
| `text_normalize` | BOOLEAN | no | `true` | Run CosyVoice's text frontend (number and abbreviation expansion). Turn it off when you feed CMU phonemes or special tags that must survive verbatim. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `audio` | AUDIO | The generated speech, at the model's own sample rate (24 kHz). |

## Notes

- Reference audio is trimmed to the first 30 seconds and downmixed to mono 24 kHz before it reaches the model. Clean speech without music or background noise clones far better than a long noisy sample.
- Write the text in the target language itself — this node does not translate. Pair it with a translation node if you need that.
- On CosyVoice3 the language tag is not sent explicitly; the model infers the language from the text, so target_language mostly matters for older CosyVoice checkpoints.
- Two ways to get a different take. seed = -1 re-runs the node on every execution, so each run is a new take. With a fixed seed the result is reproducible, and ComfyUI returns the cached audio until something changes — set control_after_generate to randomize if you want the seed itself to move between runs.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
