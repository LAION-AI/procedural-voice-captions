# Humaneness Voice caption and transcript format

This document defines the **default output contract** of Procedural Voice Captions.
The format identifier is `humaneness-general-script-v2-20261011`.
The shared conventions come from the [Humaneness Voice Small model](https://huggingface.co/laion/Humaneness-Voice-Small), its [timed annotation engine](https://huggingface.co/laion/Humaneness-Voice-Small/blob/main/code/anno_timed.py), and the [original prompting guide](https://laion-humaneness-voice-small-acting-eval-space.static.hf.space/index.html). The local original is the August 25, 2026 prompting schema; sentence durations in that guide use **square**, not round, brackets.

## The two required sections

```text
GENERAL: A quiet, soft-spoken delivery with a slightly breathy timbre and audible sadness.
SCRIPT:
(soft, restrained, sad) [2.10 seconds duration] I thought you might stay.
[0.40 seconds pause]
(Chuckle, 0.30 seconds)
[0.20 seconds pause]
(quiet, hesitant) [1.20 seconds duration] Just for tonight.
```

The example spans 4.20 seconds. A production caption uses model predictions for its wording and measured ASR/detector endpoints for the numbers. This example is illustrative.

| Element | Exact syntax | Meaning |
|---|---|---|
| Global voice description | `GENERAL: ...` | Voice, delivery, affect and quality over the complete recording |
| Script section | `SCRIPT:` followed by lines | The spoken content, sentence delivery and vocal events |
| Delivery direction | `(soft, restrained, sad)` | How to perform the following spoken segment; no duration inside a direction |
| Spoken segment duration | `[2.10 seconds duration]` | The inclusive time budget of the following segment |
| Pause | `[0.40 seconds pause]` | A gap with no aligned speech or selected vocal event |
| Vocal burst | `(Chuckle, 0.30 seconds)` | An event label and its measured duration; never a bare label in the default timed output |

One duration is emitted for **every spoken segment**. A sentence can be split at a substantial pause, a vocal burst in a word gap, or a long segment boundary. Each ASR word appears exactly once, in its original order. This is not one clip-length duration prepended to an entire multi-sentence transcript.

## Event endpoints and overlapping events

Every event also retains `start` and `end`, or `start_s` and `end_s`, in the JSON, relative to the complete recording. The displayed event duration comes from `end − start`; it is not guessed from the event class.

The historical guide allowed speech-overlapping bursts to become bare labels. The current default instead retains **all selected event durations**, as requested for training. This requires an explicit extension to the old timing rules:

```text
(strained, hesitant) [1.00 seconds duration] I (Breath, 0.30 seconds) know.
```

Here the breath happens **inside** the one-second segment. It is an annotation of a simultaneous event and does not make the segment 1.30 seconds long. A burst on its **own line** consumes additional timeline time. A burst inside a line that has a segment duration is included in that enclosing duration. A group of overlapping standalone bursts similarly uses one enclosing interval budget with all event lengths retained inside it.

This distinction is recorded in `script_timing.timeline`, `duration_includes_inline_events` and `overlap_policy`. The independent verifier parses this distinction from the emitted lines. Do not reuse an older verifier that blindly adds every numeric burst token to every enclosing duration; it would double-count simultaneous audio. The original Humaneness Small linear parser predates this extension. Plain and untimed inline transcripts remain available when adapting existing training consumers; the accompanying new parser is the timing contract for the duration-rich output.

Raw detections, selected detections, class softmax probabilities, actual endpoints and all original acoustic scores remain available. Formatting does not establish that a candidate event is truly audible.

## Pauses, rounding and verification

Gaps of at least 0.20 seconds receive pause markers. Smaller gaps are included in adjacent speech budgets. If neither neighbour is speech, a small gap is printed rather than added to a burst's duration. A pause means no aligned foreground speech or selected vocal event; a recording can still contain background sound.

Numbers have two decimal places. Segment budgets are differences of rounded absolute endpoints, so rounding cannot accumulate drift. The independent parser checks the elapsed timeline against the recording length, the complete word sequence, and one numeric duration per selected burst. Strict default generation raises `TimingUnavailable` when the available alignment cannot establish a valid script; it does **not** silently substitute an untimed prompt. `timed=False` is an explicit opt-in to historical untimed output.

ASR unsupported for a language is different from silence. Listening pages show the absence of verified alignment explicitly and keep any original corpus text separate. They must not invent sentence timestamps for that text.

## Regenerate from saved predictions

```python
import json
from refresh_saved_caption import refresh_saved_caption
from humaneness_caption_settings import load_settings

settings = load_settings()
baseline = json.load(open(settings["normalization_file"]))
record = json.load(open("saved_annotation.json"))
updated = refresh_saved_caption(record, baseline, settings)
print(updated["procedural_caption"])
assert updated["timed_ok"]
```

This runs no ASR or neural model. It reuses the stored global and sentence predictions and measured word/event endpoints. The default settings and normalizer are versioned in this repository. Nine wording variants, when exported, share the same audio, timestamps, event selection and verified script; they are not nine independent acoustic annotations.

`render_timed(..., duration_style="round")` also supports `(2.10 seconds duration)` when a caller explicitly requests it. The default follows the source guide's `[2.10 seconds duration]` grammar.

## MOSS `CAPTION` / `TRANSCRIPT` training surfaces

The same dataset can provide the other Humaneness Voice Small prompt surface:

```text
CAPTION: A softly spoken, hesitant voice with a breathy delivery.
TRANSCRIPT: "[2.10 seconds duration] I thought you might stay.
[0.40 seconds pause]
(Chuckle, 0.30 seconds)
[0.20 seconds pause]
[1.20 seconds duration] Just for tonight."
```

For this surface, `CAPTION` starts with the **MOSS whole-audio caption**. A separate variant appends chronological MOSS burst descriptions with start/end times. `TRANSCRIPT` can be plain ASR text, text with `(Label, duration seconds)` event annotations, or the verified timed script. These variants are exported separately, so a training reader can choose a surface without reconstructing it or discarding the original annotations. Burst timestamps come from Humaneness Ears; the burst prose comes from the interval-conditioned MOSS adapter, which receives the complete audio and both endpoints.

Original voice identifiers, language, acting challenge, source revision, directions, scripts, API model identifiers and generation settings stay in the dataset metadata. Credentials are not dataset metadata and are excluded.

Original documentation: CC BY 4.0, LAION.
