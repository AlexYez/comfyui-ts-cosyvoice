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
| `seed` | INT | no | `42` | Random seed. Set it to -1 for a fresh take on every run: the node then re-runs instead of returning its cached audio. Any value from 0 up is deterministic — re-running with everything unchanged returns the previous result, so change the seed (or use control_after_generate: randomize) when you want a different take from a fixed seed. Range: `-1`–`2147483647`. |
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
- Two ways to get a different take. seed = -1 re-runs the node on every execution, so each run is a new take. With a fixed seed the result is reproducible, and ComfyUI returns the cached audio until something changes — set control_after_generate to randomize if you want the seed itself to move between runs.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
