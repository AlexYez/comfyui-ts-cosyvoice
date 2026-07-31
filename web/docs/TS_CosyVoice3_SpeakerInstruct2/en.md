# TS CosyVoice Speaker Text To Voice

The same instruction-guided synthesis as TS CosyVoice Text to Voice, but the voice comes from a preset you saved earlier with TS CosyVoice Save Speaker instead of from a reference clip on the canvas. Use it when you reuse one voice across many generations.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | COSYVOICE_MODEL | yes | — | The loaded CosyVoice model, from the TS CosyVoice Model Loader node. |
| `text` | STRING | yes | `Hello, this is my cloned voice speaking.` | The text to speak in the saved voice. |
| `instruct_text` | STRING | yes | `Please say this in a warm and soft voice.` | Free-form instruction describing emotion and delivery. Only used when emotion_preset is set to the custom entry. |
| `speaker_preset` | COMBO | yes | — | A voice preset saved earlier by TS CosyVoice Save Speaker, read from ComfyUI/models/cosyvoice/speaker/. Options: `Nastya`, `my_speaker`, `my_speaker_v2`. |
| `speed` | FLOAT | yes | `1.0` | Playback speed multiplier applied to the generated speech. Range: `0.5`–`2.0`, step `0.05`. |
| `seed` | INT | no | `42` | Random seed. Set it to -1 for a fresh take on every run: the node then re-runs instead of returning its cached audio. Any value from 0 up is deterministic — re-running with everything unchanged returns the previous result, so change the seed (or use control_after_generate: randomize) when you want a different take from a fixed seed. Range: `-1`–`2147483647`. |
| `text_normalize` | BOOLEAN | no | `true` | Run CosyVoice's text frontend (number and abbreviation expansion). Turn it off when you feed CMU phonemes or special tags that must survive verbatim. |
| `emotion_preset` | COMBO | no | `Ваша инструкция` | Ready-made delivery preset, read from configs/ts_emotion_presets.json. Pick the custom entry to use the instruct_text field instead. Options: `Ваша инструкция`, `Теплый дружелюбный`, `Спокойный мягкий`, `Энергичный выразительный`, `Кинематографичный драматичный`, `Интимный тихий`, and 7 more. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `audio` | AUDIO | The generated speech, at the model's own sample rate (24 kHz). |

## Notes

- The dropdown lists the `.pt` files in `ComfyUI/models/cosyvoice/speaker/`. With none saved yet it shows a single placeholder entry and the node asks you to run Save Speaker first.
- A preset re-saved under the same name is picked up: the node keys its cache on the file's timestamp, so it re-runs instead of returning the previous voice.
- A workflow shared with someone who does not have that preset file will fail validation on this node — ship the `.pt`, or have them recreate it with Save Speaker.
- Two ways to get a different take. seed = -1 re-runs the node on every execution, so each run is a new take. With a fixed seed the result is reproducible, and ComfyUI returns the cached audio until something changes — set control_after_generate to randomize if you want the seed itself to move between runs.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
