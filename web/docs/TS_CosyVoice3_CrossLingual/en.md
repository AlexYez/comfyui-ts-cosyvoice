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
| `seed` | INT | no | `42` | Random seed. Note that -1 does not give you a new take on re-run: ComfyUI caches a node whose inputs did not change. Use the seed widget's control_after_generate: randomize to actually get a different take. Range: `-1`–`2147483647`. |
| `text_normalize` | BOOLEAN | no | `true` | Run CosyVoice's text frontend (number and abbreviation expansion). Turn it off when you feed CMU phonemes or special tags that must survive verbatim. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `audio` | AUDIO | The generated speech, at the model's own sample rate (24 kHz). |

## Notes

- Reference audio is trimmed to the first 30 seconds and downmixed to mono 24 kHz before it reaches the model. Clean speech without music or background noise clones far better than a long noisy sample.
- Write the text in the target language itself — this node does not translate. Pair it with a translation node if you need that.
- On CosyVoice3 the language tag is not sent explicitly; the model infers the language from the text, so target_language mostly matters for older CosyVoice checkpoints.
- To get a different take, set the seed widget's control_after_generate to randomize. Leaving seed at -1 does not help on its own: with unchanged inputs ComfyUI returns the cached result instead of running the node again.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
