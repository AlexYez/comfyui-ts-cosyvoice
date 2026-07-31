# TS CosyVoice Dialog

Turns a script into a multi-voice conversation. Label each line with SPEAKER A:, SPEAKER B: and so on, give each speaker a reference clip, and the node renders the whole exchange — plus one isolated track per speaker for mixing.

## Inputs

| Input | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `model` | COSYVOICE_MODEL | yes | — | The loaded CosyVoice model, from the TS CosyVoice Model Loader node. |
| `dialog_text` | STRING | yes | `SPEAKER A: Hello, how are you?
SPEAKER B: I'm doing great, thanks for asking!` | The script, one line per turn, each prefixed with SPEAKER A:, SPEAKER B:, SPEAKER C: or SPEAKER D:. Lines without a recognised prefix, and lines for a speaker with no reference audio, are skipped. |
| `speaker_A_Audio` | AUDIO | yes | — | Reference voice for SPEAKER A. 3-10 seconds works best; 30 seconds is the hard limit. |
| `speaker_B_Audio` | AUDIO | yes | — | Reference voice for SPEAKER B. 3-10 seconds works best; 30 seconds is the hard limit. |
| `speed` | FLOAT | yes | `1.0` | Playback speed multiplier applied to every line of the dialog. Range: `0.5`–`2.0`, step `0.05`. |
| `speaker_C_Audio` | AUDIO | no | — | Optional reference voice for SPEAKER C. |
| `speaker_D_Audio` | AUDIO | no | — | Optional reference voice for SPEAKER D. |
| `seed` | INT | no | `42` | Random seed. Set it to -1 for a fresh take on every run: the node then re-runs instead of returning its cached audio. Any value from 0 up is deterministic — re-running with everything unchanged returns the previous result, so change the seed (or use control_after_generate: randomize) when you want a different take from a fixed seed. Range: `-1`–`2147483647`. |

## Outputs

| Output | Type | Description |
| --- | --- | --- |
| `dialog_audio` | AUDIO | The full conversation, lines concatenated in script order. |
| `speaker_a_audio` | AUDIO | SPEAKER A alone, silent where the others speak — same length as dialog_audio, so the tracks line up. |
| `speaker_b_audio` | AUDIO | SPEAKER B alone, silent where the others speak. |
| `speaker_c_audio` | AUDIO | SPEAKER C alone; all silence when C never speaks. |
| `speaker_d_audio` | AUDIO | SPEAKER D alone; all silence when D never speaks. |
| `message` | STRING | Short report: duration, number of lines rendered, and any lines that were skipped. |

## Notes

- The prefix match is case-insensitive, so `Speaker A:` works as well as `SPEAKER A:`.
- Every per-speaker track is the full length of the dialog, so you can drop all four straight onto a timeline without aligning them.
- Reference clips are transcribed with Whisper once per run and the result is reused for every line by that speaker. The first use downloads the Whisper 'base' model (about 140 MB).
- Cancel stops the node between lines rather than at the end of the whole dialog.
- Check the message output: it names the lines that were skipped for want of a matching speaker input.

---

<sub>This page is generated from the node schema and `tools/ts_node_docs_source.py`. Do not edit it by hand.</sub>
