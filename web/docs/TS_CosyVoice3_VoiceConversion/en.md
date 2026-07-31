# TS CosyVoice Voice To Voice

Re-voices an existing recording: keeps the timing, phrasing and delivery of the source performance, and replaces its timbre with the target voice. Nothing is re-read from text, so the original acting is preserved.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | COSYVOICE_MODEL | yes | — | The loaded CosyVoice model, from the TS CosyVoice Model Loader node. |
| `source_audio` | AUDIO | yes | — | The performance to re-voice. Timing and delivery are kept; only the timbre changes. Long audio is processed in chunks. |
| `target_audio` | AUDIO | yes | — | The voice to sound like. Trimmed to 30 seconds and converted to the model's format. |
| `speed` | FLOAT | yes | `1.0` | Playback speed multiplier applied to the generated speech. Range: `0.5`–`2.0`, step `0.05`. |
| `pitch_shift_semitones` | INT | yes | `0` | Shift the source pitch before conversion. Useful when source and target sit in different registers — try -12 or +12 when converting between a low and a high voice. Range: `-12`–`12`, step `1`. |
| `seed` | INT | no | `42` | Random seed. Note that -1 does not give you a new take on re-run: ComfyUI caches a node whose inputs did not change. Use the seed widget's control_after_generate: randomize to actually get a different take. Range: `-1`–`2147483647`. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `audio` | AUDIO | The re-voiced audio, at the model's own sample rate (24 kHz). |

## Notes

- Source audio longer than 24 seconds is split into chunks. The split prefers a pause so it does not land mid-word, and the converted chunks are rejoined with a short crossfade.
- This is the slowest node in the pack. On CPU a ten-minute file can take hours — run it on a GPU.
- Cancel stops it between chunks, so a long job aborts within roughly one chunk rather than running to the end.
- If silence detection is unavailable in your environment, the node logs a warning and falls back to fixed-size chunks. Conversion still works; a split may land mid-word.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
