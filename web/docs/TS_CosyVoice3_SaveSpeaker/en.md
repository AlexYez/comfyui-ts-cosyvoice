# TS CosyVoice Save Speaker

Extracts the speaker features from a reference recording and saves them as a reusable `.pt` preset, so you do not have to keep the reference audio in every graph. Presets are picked up by TS CosyVoice Speaker Text To Voice.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | COSYVOICE_MODEL | yes | — | The loaded CosyVoice model, from the TS CosyVoice Model Loader node. |
| `reference_audio` | AUDIO | yes | — | Voice to clone. Trimmed to 30 seconds and converted to mono 24 kHz; 3-10 seconds of clean speech gives the best timbre. |
| `reference_text` | STRING | yes | — | What is actually said in the reference audio. Leave it empty to have Whisper transcribe it automatically — an accurate transcript noticeably improves the clone. |
| `speaker_name` | STRING | yes | `my_speaker` | File name for the preset, without extension. Must be a plain name: no slashes, no '..', none of < > : " / \ \| ? *. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `saved_path` | STRING | Absolute path the preset was written to. |

## Notes

- Reference audio is trimmed to the first 30 seconds and downmixed to mono 24 kHz before it reaches the model. Clean speech without music or background noise clones far better than a long noisy sample.
- Presets are written to `ComfyUI/models/cosyvoice/speaker/`. Saving under an existing name overwrites it.
- Leaving reference_text empty pulls in Whisper. The first time, that downloads the Whisper 'base' model (about 140 MB) into `ComfyUI/models/whisper/`.
- If Whisper is missing or fails, the preset is still saved but with an empty transcript, and a warning naming the reason is written to the ComfyUI console log. Such a preset clones noticeably worse, and the node gives no on-canvas sign of it — check the log after saving a preset without a reference_text, then fill the text in by hand and save again.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
