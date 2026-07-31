# TS CosyVoice Text to Voice

Zero-shot voice cloning: give it a reference recording and the text to read, and it speaks that text in the reference voice. The delivery — warm, dramatic, tired — is steered by an instruction, either a ready-made preset or your own sentence.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | COSYVOICE_MODEL | yes | — | The loaded CosyVoice model, from the TS CosyVoice Model Loader node. |
| `text` | STRING | yes | `Hello, this is my cloned voice speaking.` | The text to speak in the reference voice. |
| `instruct_text` | STRING | yes | `Speak in a warm and friendly tone.` | Free-form instruction describing emotion and delivery. Only used when emotion_preset is set to the custom entry. |
| `reference_audio` | AUDIO | yes | — | Voice to clone. Trimmed to 30 seconds and converted to mono 24 kHz; 3-10 seconds of clean speech gives the best timbre. |
| `speed` | FLOAT | yes | `1.0` | Playback speed multiplier applied to the generated speech. Range: `0.5`–`2.0`, step `0.05`. |
| `seed` | INT | no | `42` | Random seed. Note that -1 does not give you a new take on re-run: ComfyUI caches a node whose inputs did not change. Use the seed widget's control_after_generate: randomize to actually get a different take. Range: `-1`–`2147483647`. |
| `text_normalize` | BOOLEAN | no | `true` | Run CosyVoice's text frontend (number and abbreviation expansion). Turn it off when you feed CMU phonemes or special tags that must survive verbatim. |
| `emotion_preset` | COMBO | no | `Ваша инструкция` | Ready-made delivery preset, read from configs/ts_emotion_presets.json. Pick the custom entry to use the instruct_text field instead. Options: `Ваша инструкция`, `Теплый дружелюбный`, `Спокойный мягкий`, `Энергичный выразительный`, `Кинематографичный драматичный`, `Интимный тихий`, and 7 more. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `audio` | AUDIO | The generated speech, at the model's own sample rate (24 kHz). |

## Notes

- Reference audio is trimmed to the first 30 seconds and downmixed to mono 24 kHz before it reaches the model. Clean speech without music or background noise clones far better than a long noisy sample.
- emotion_preset wins over instruct_text. instruct_text is read only when the preset is set to the custom entry.
- Presets are read from `configs/ts_emotion_presets.json` on every run, so you can edit that file and use the change without restarting ComfyUI. New preset *names* appear in the dropdown after a node-definition refresh.
- To get a different take, set the seed widget's control_after_generate to randomize. Leaving seed at -1 does not help on its own: with unchanged inputs ComfyUI returns the cached result instead of running the node again.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
