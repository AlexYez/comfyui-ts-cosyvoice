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
| `seed` | INT | no | `42` | Random seed. Set it to -1 for a fresh take on every run: the node then re-runs instead of returning its cached audio. Any value from 0 up is deterministic — re-running with everything unchanged returns the previous result, so change the seed (or use control_after_generate: randomize) when you want a different take from a fixed seed. Range: `-1`–`2147483647`. |
| `diffusion_steps` | INT | no | `10` | Euler steps taken by the flow decoder. 10 is the model's own default and favours speed; 25-40 resolves more detail and costs proportionally more time. Raise this before anything else if the result sounds smeared. Range: `4`–`60`, step `1`. |
| `guidance_strength` | FLOAT | no | `0.7` | Classifier-free guidance strength. 0.7 is the shipped configuration; higher follows the target voice more closely but increases the risk of artefacts, lower is safer and blander. Range: `0.0`–`1.5`, step `0.05`. |
| `normalize_output` | BOOLEAN | no | `false` | Scale the result to a fixed level so different reference clips do not produce wildly different output volumes. |
| `target_rms_dbfs` | FLOAT | no | `-20.0` | Level to normalise to, when normalize_output is on. This is RMS, not LUFS — it is not an EBU R128 loudness measurement. Peaks are held below -1 dBFS, so a very high target is given up rather than clipped. Range: `-40.0`–`-6.0`, step `0.5`. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `audio` | AUDIO | The re-voiced audio, at the model's own sample rate (24 kHz). |

## Notes

- Source audio longer than 24 seconds is split into overlapping chunks. The split prefers a pause so it does not land mid-word, each chunk re-reads one second of its predecessor so the voice does not restart at the boundary, and the converted chunks are aligned on that overlap before being crossfaded together.
- Pitch shifting rescales the fundamental with the WORLD vocoder, which leaves the formants where they are — so a large shift does not also make the speaker sound smaller or larger. This needs the optional `pyworld` package, which is not installed by default because it publishes wheels for Windows only; without it the node falls back to a phase vocoder and logs which path it took. `pip install pyworld` enables it (macOS needs the Xcode Command Line Tools to build it).
- This is the slowest node in the pack. On CPU a ten-minute file can take hours — run it on a GPU.
- Cancel stops it between chunks, so a long job aborts within roughly one chunk rather than running to the end.
- If silence detection is unavailable in your environment, the node logs a warning and falls back to fixed-size chunks. Conversion still works; a split may land mid-word.
- One clip at a time. These nodes process a single AUDIO clip, so a batch of more than one is refused with an explicit error rather than silently reduced to the first clip. Split the batch upstream, or run the graph once per clip.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
