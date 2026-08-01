# Procedural Voice Captions

**Turn a speech clip into a voice-acting caption, entirely from model predictions.**
A stack of listening models scores the clip — 57 VoiceNet voice dimensions, 40 EmoNet
emotions, genuineness, ASR with word timestamps, and vocal-burst detection — and this
repo turns those numbers into text: one **GENERAL** line describing how the voice
sounds, and a **SCRIPT** with a delivery cue per sentence, the vocal bursts written
in where they actually happen, and the silences marked. No hand-writing, no LLM.

---

## What the output looks like

Two kinds of brackets, and they never mean the same thing:

| | |
|---|---|
| **`(round brackets)`** | **What the voice does.** At the head of a sentence it is the **delivery cue** for that sentence; inside the sentence it is a **vocal burst** the locator found and the classifier named, written at the moment it happens. |
| **`[square brackets]`** | **Silence.** `[pause X.Xs]` is a real gap between words, measured from the Parakeet-v3 word timestamps (default: any gap ≥ 0.30 s). |

Three real outputs, produced by `burst_captions.py` at its current defaults
(the full set is on the [demo page](#demo-pages)):

**1 — a goblin character voice** ([listen ↗](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/#goblin5))

```
GENERAL: very flat-resonance, very warm, very decelerating, very mumbled, very soft-onset,
         middle-aged, low-register, slow, very pining, no absorption, very helplessness,
         genuine, with-bursts
SCRIPT:
(very flat-resonance, very smooth, very non-narrative, no deep focus, very fatigue, no interest)
    Huh. (Chuckle)
(very flat-resonance, very flat, very mumbled, very helplessness, very misery, no absorption)
    [pause 0.4s] What have we here?
(very flat-resonance, very meek, very warm, fondness, feeling turned on, altered perception)
    [pause 0.5s] A lost little traveller with shiny pockets?
```

The `(Chuckle)` is not in the transcript — Parakeet never heard a word there. The locator
found a burst at 0.22–0.70 s, the classifier named it, and it was written between the two
words nearest that moment.

**2 — an in-the-wild podcast clip** ([listen ↗](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/#EN_B00000_S03298_W000123))

```
GENERAL: oral-bright, full, halting, naturalistic, slightly non-narrative, adult, masculine,
         low-register, very mirth, existential void, short-temperedness, genuine, with-bursts
SCRIPT:
(very flat-resonance, very oral-bright, dialogic, no attention, very ribbing, very disdain)
    So that people go, What is that? (Breathy Giggle)
(very flat-resonance, very commanding, very oral-bright, no engrossment, no thoughtfulness, no fascination)
    I need to find out about that.
(very oral-bright, very blended-resonance, smooth, very bantering, very existential void, very suspicion)
    [pause 0.7s] So this is me making stuff up right in the moment, but I'm like, okay, I'm
    gonna use these oblique strategies as a way of
```

**3 — a distressed character voice** ([listen ↗](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/#zombie5))

```
GENERAL: very flat-resonance, very vulnerable, very meek, very explicit, very choppy, young,
         feminine, low-register, very slow, very lethargy, no engrossment, very feeling horny,
         genuine, with-bursts
SCRIPT:
(very flat-resonance, very vulnerable, very meek, very lethargy, very suffering, very submission)
    So hungry.
(very flat-resonance, very smooth, very flat, very burnout, very respite, no engrossment)
    [pause 0.5s] Hmm. (Wistful Sigh)
(very flat-resonance, very mumbled, very jerky, very desperation, very despair, very torment)
    [pause 1.3s] Help me.
(very flat-resonance, very choppy, very soft-onset, very burnout, very desperation, very dejection)
    [pause 0.3s] I can't stop.
```

Those examples use the terse **`tags`** surface form, which is the default. The same
content — same dimensions, same emotions, same gates — can be rendered as prose through
[11 interchangeable templates](#caption-templates--dimension-shuffling). Example 1's GENERAL
line under `template="default"`:

```
A voice that is extremely one-dimensional in resonance; extremely warm and enveloping;
extremely decelerating in pace; extremely blurry and mumbled; extremely soft and gentle in
onset; middle-aged to mature; low and bassy in register; notably slow and deliberate in
tempo; extremely carrying pining; notably free of absorption; extremely carrying
helplessness; genuine in delivery; interwoven with vocal bursts (laughs, gasps, sighs).
```

### Hear the clips

GitHub's README renderer **cannot embed an audio player** — the links above and below are
ordinary links out to the GitHub Pages demo, where each clip has a real `<audio>` element
next to its caption. Nothing plays inside this page.

- ▶ **[Current output — locator v2 + classifier v2](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/)**
  (14 clips, each with a player, the located spans, and the previous pipeline's output for comparison)
- ▶ **[100-clip multilingual caption grid](https://projects.laion.ai/procedural-voice-captions/)**
- ▶ **[Character voices: procedural vs LLM-assisted](https://projects.laion.ai/procedural-voice-captions/character-captions/)**

Mirror: [`laion-ai.github.io/procedural-voice-captions`](https://laion-ai.github.io/procedural-voice-captions/).

---

## The default path

### What goes into the GENERAL line

Every dimension is **z-scored against the average voice**: `z = (value − median) / spread`,
where the baseline medians and spreads come from the bundled `baseline_stats.json`
(99 dimensions: 57 VoiceNet + genuineness + vocal-burst blend + 40 EmoNet emotions).
Then, for the whole clip:

| | selected | of |
|---|---|---|
| VoiceNet dimensions | **top 5 by \|z\|** | 57 |
| EmoNet emotions | **top 3 by \|z\|** | 40 |
| Always included regardless of \|z\| | **Age `AGEV` · Gender `GEND` · Register `REGS` · Tempo `TEMP`** | — |
| Also appended | genuineness, and vocal-burst blend when \|z\| ≥ 0.5 | — |

(`k_voicenet=5`, `k_emonet=3`, `ALWAYS_ON = ["AGEV","GEND","REGS","TEMP"]` in
[`caption.py`](caption.py) — those are the defaults of `caption()` / `caption_detail()`.)

Two selection rules are worth knowing:

- **Categorical dims use absolute 0–6 bands, not z-scores.** `AGEV`, `GEND` and `REGS`
  are near-categorical, so deviation-from-baseline is the wrong lens: the population
  median for `GEND` leans masculine (3.71), and a ±0.5·spread "neutral" band swallowed
  most male voices and mislabeled them *gender-neutral*. These three are mapped through
  fixed thresholds centred on the scale midpoint 3.0 (`ABSOLUTE_BANDS`) — `GEND ≥ 3.5` →
  masculine, `≤ 2.5` → feminine, in between → androgynous. `TEMP` and everything else
  stays on the z-score path.
- **Intensity comes from |z|** — ≥ 2.0 Extremely, ≥ 1.5 Very, ≥ 1.0 Notably, ≥ 0.5
  Somewhat; below 0.5 the dimension is "about average" and, in the `tags` form, dropped
  entirely rather than padded with filler.

Emotion wording then rotates through each emotion's **synonym cluster** (so *Sadness*
surfaces as *dejection*, *heartache*, *misery*, … across clips), and the
[genuineness gate](#the-genuineness-gate) and [gender gate](#the-gender-gate-empathic-insight)
are applied.

### What goes into each SCRIPT line

Per sentence the same machinery runs on that sentence's audio only, with
**top-3 VoiceNet + top-3 EmoNet and no always-on identity dims** — age, gender, register
and tempo are properties of the speaker, so they belong in GENERAL and would only be noise
repeated on every line (`sentence_caption()` in [`burst_captions.py`](burst_captions.py)).
Sentences and word times come from **Parakeet-TDT v3**; gaps ≥ `BURST_PAUSE_THR` (0.30 s)
become `[pause X.Xs]`, both inside a sentence and between sentences.

---

## The vocal-burst path

`caption.py` describes the *voice*. It does not know that the speaker laughed. Getting
`(Chuckle)` into the script takes **two different models**, and they are easy to confuse:

| stage | model | question it answers |
|---|---|---|
| **1. LOCATOR** | [`laion/vocalburst-locator`](https://huggingface.co/laion/vocalburst-locator) → **`model_v2.pt`** | *When?* — a burst probability for every 20 ms frame |
| **2. CLASSIFIER** | [`laion/vocal-burst-detector-v2`](https://huggingface.co/laion/vocal-burst-detector-v2) | *What?* — names an already-cut span, or vetoes it |

The pipeline:

1. **Locate.** Run the locator over the clip → 50 fps burst probabilities.
2. **Extract events.** Threshold, merge near-adjacent runs, drop very short ones →
   `[(start, end, confidence), …]`.
3. **Cut.** Slice each span's audio out of the waveform.
4. **Name.** VoiceCLAP-commercial embedding → the v2 MLP → softmax over 83 classes.
   If **`P(no_burst) ≥ 0.5`** the span is discarded — this is the gate that kills false
   alarms. Otherwise the top-1 burst class is taken.
5. **Write it in.** The class name is inserted as `(Class Name)` inline in the sentence
   the burst falls in, positioned between the two ASR words nearest its midpoint.

### The locator — `model_v2.pt`, and why the defaults changed

`model_v2.pt` is a fine-tune of the original `model.pt` on real in-the-wild audio.
Measured by its [model card](https://huggingface.co/laion/vocalburst-locator#v2--fine-tuned-on-real-in-the-wild-audio-model_v2pt)
on **992 held-out real expressive-speech clips**, each checkpoint given its own best
operating point by a post-processing sweep:

| checkpoint | event F1 @ IoU 0.5 | precision | recall | best threshold |
|---|---|---|---|---|
| `model.pt` (v1, trained on synthetic soundscapes) | 0.152 | 0.469 | 0.207 | 0.80 |
| **`model_v2.pt` (default here)** | **0.607** | 0.678 | 0.669 | 0.50 |

**The post-processing defaults published with v1 are wrong for real audio**, and this repo
no longer uses them:

| parameter | v1 default | **current default** | env var |
|---|---|---|---|
| threshold | 0.65 | **0.50** | `BURST_LOCATOR_THR` |
| merge_gap | 0.30 s | **0.10 s** | `BURST_MERGE_GAP` |
| min_duration | 0.50 s | **0.10 s** | `BURST_MIN_DUR` |

`min_duration` is the one that matters. Real vocal bursts have a **median duration of
180 ms**, so a 0.5 s floor throws away ~96 % of them before anything else runs. On an
unchanged checkpoint, fixing the post-processing alone moved event F1 from **0.243 to
0.598** — a bigger effect than any training change.

Architecture (unchanged from v1, so the same loader works): whisper-small encoder with
rank-8 LoRA **merged into the base weights** → `Linear(768→384) + GELU + Dropout` →
`Conv1d(384, k=7) + GELU + Dropout` → `Linear(384→1)` → **1500 frame logits (50 fps × 30 s)**.
Plain state dict, 485 tensors, loads with `load_state_dict(..., strict=True)`.

### The 30-second window — how longer audio is handled

**The locator's input is a fixed 30 s window (1500 frames).** It physically cannot see
more. Anything longer must be split, and this repo does it in `BurstCaptioner.locator_probs()`:

- The clip is cut into **30 s windows with 5 s of overlap** (hop 25 s; `BURST_CHUNK_SEC`,
  `BURST_CHUNK_OVERLAP`). A final window is anchored to the end of the audio so the tail
  is always fully covered. Short audio is zero-padded to 30 s, exactly as at training time.
- Each window is scored on its own, and its 1500 frame probabilities are written back into
  **one global frame track at the window's offset** — this is where the chunk offset is
  added.
- **Merging happens at frame level, before events are extracted.** In an overlap region the
  two windows are cross-faded: each window's contribution ramps linearly from 0 at its own
  shared edge to 1 at the inner end of the overlap, and the track is the weight-normalised
  sum. A frame is therefore dominated by the window that sees it with the most context on
  both sides.
- Only then are events extracted, **once**, from the stitched track.

The consequence is the important part: **a burst that straddles a seam cannot be lost or
double-counted.** There are no per-chunk events to de-duplicate, because there is only ever
one event list, derived from one continuous probability curve. (Event-level merging with an
IoU rule is the usual approach; stitching first is strictly simpler and makes the
de-duplication exact rather than heuristic.)

Verified on this repo's own audio: a 91.4 s clip (four windows) placed the same eight
bursts that per-clip scoring finds, at the correct global times — e.g. an `Exhausted Groan`
detected at 0.22 s in a standalone 7 s clip appears at **84.54 s** in the concatenation,
where that clip starts at 84.32 s. Deliberately positioning a burst at 25.0 s and at 30.0 s
(both window boundaries) produced exactly **one** event each time, at the right offset.
You can hear the long clip on the
[demo page](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/#longform_concat_91s).

One honest caveat: windowing changes what the model *hears around* a burst, so a long clip
is not bit-identical to the sum of its parts. Re-scoring the same 91 s of audio as nine
separate clips found 7 events; the windowed pass found 8, sharing 6. Neither is
ground truth; the difference is acoustic context, not a bookkeeping error.

### The classifier — `laion/vocal-burst-detector-v2`

Frozen [`laion/voiceclap-commercial`](https://huggingface.co/laion/voiceclap-commercial)
768-d embedding → `Linear(768,256) → BatchNorm → GELU → Dropout(0.3) → Linear(256,83)`,
**219,220 parameters**. 83 classes = 82 taxonomy bursts + `no_burst` at index 82; the class
order is identical to the bundled `vocalburst_taxonomy.json` (verified against the repo's
`classes.json`). Its model card reports **58.1 % argmax accuracy** on the official
single-label validation set (1940 clips: 70.4 % on pure single-burst, 39.4 % on composite).

Five classes — `Blowing a Kiss`, `Finger Snaps`, `Hand Scratching Head`, `Hand Slaps`,
`Slap Face` — were folded into `no_burst` during v2 training because they are hand gestures,
not audible vocalizations. This repo reads `folded_classes.json` and **masks them to −∞
before the softmax**, so they can never be predicted. The effective label space is
**77 burst classes + `no_burst`**.

That masking is why the old *transient duration gate* is off by default. v1 needed a 0.60 s
floor for the smack/click/slap groups because the old multi-label classifier hallucinated
`Slap Face` on 0.1–0.2 s spans; those classes are now unreachable, and a 0.6 s floor
contradicts the 180 ms median burst duration. `BURST_TRANSIENT_MIN_DUR` still exists if you
want it back.

The span is handed to the classifier **exactly as the locator found it** (`BURST_CLF_CONTEXT=0`,
which reproduces the classifier model card's own inference). Widening the cut does move
labels — on the demo clips, 0.25 s of extra context flips a 0.38 s span from `Contented Sigh`
to `Surprised Gasp` — but there is no held-out measurement saying which is better, so the
faithful cut is the default.

### Two burst-insertion variants

- **Variant A — locator, precise position (the default).** Everything described above:
  each kept span becomes its own `(Class Name)` at its own time, inside the sentence it
  falls in. Reads like a script.
- **Variant B — sentence-level (fallback).** The classifier is run on a whole sentence
  segment; if `P(no_burst) ≥ 0.5` nothing is attached, otherwise the top-1 class is woven
  into that sentence's *cue* with a small procedural phrase (*"… punctuated by a (Gasp)."*).
  Use it when word-level timestamps are unavailable (non-speech screams, failed ASR) or when
  one label per sentence is enough — Variant A has nothing to anchor to there.

Both are computed on every clip; `result["script_lines"]` is Variant A, and
`result["sentences"][i]["variant_b_caption"]` is Variant B. They are never mixed in one caption.

---

## Quickstart

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # the LAION model repos
export HF_HOME=/path/to/hf_cache

# full pipeline on audio -> per-clip JSON (+ mp3s for a demo page)
python burst_captions.py 'my_clips/*.wav' --out out.json --mp3-dir mp3/

# caption from already-computed predictions, no audio, no models, stdlib only
python caption.py examples/worker_0_EN_tDPU-wXSB5y_W000085.json --kv 5 --ke 3
```

**Every model defaults to a public HuggingFace repo** and is downloaded on first use —
a clean checkout needs no local files. Each location is still overridable by env var if you
have them on disk (`VOICENET_REPO`, `GENU_PT`, `BLEND_PT`, `BURST_LOCATOR_PT`,
`BURST_MLP_PT`, …); see the ENV block at the top of [`burst_captions.py`](burst_captions.py).

From Python:

```python
from burst_captions import BurstCaptioner
cap = BurstCaptioner()                       # loads every model once
r = cap.process("clip.wav")
print(cap.procedural_caption(r))             # the GENERAL + SCRIPT block shown above
r["variant_a_bursts"]                        # every located span, kept or dropped, with reasons
r["n_locator_windows"]                       # how many 30 s windows the clip needed
```

Or caption alone, from stored predictions (no torch, no audio):

```python
from caption import caption, caption_detail, load_baseline

base  = load_baseline()                      # bundled baseline_stats.json
preds = { ... }                              # see "Prediction format"
print(caption(preds, base, k_voicenet=5, k_emonet=3))
detail = caption_detail(preds, base)         # structured: per-dim {dim, z, direction, phrase, ...}
```

### Prediction format

`caption()` accepts either layout:

- **nested** — `{"dims": {DIM: value, ...}, "emo": {Emotion: value, ...}, "genu": x, "blend": y}`
- **flat** — `{DIM: value, ..., Emotion: value, ..., "genuineness": x, "blend": y}`

Values may be plain numbers or `{"value"/"reg_score"/"reg": x}`. VoiceNet dims and
genuineness are on a 0–6 scale, blend on 0–10, EmoNet emotions are raw head outputs
(roughly 0–3, mostly ≈ 0).

```bash
python caption.py preds.json --template identity_first     # pick a surface form
python caption.py preds.json --template random --shuffle   # seed-random form + shuffled order
python caption.py - --json < preds.json                    # stdin -> structured detail
```

---

## Caption templates & dimension shuffling

A single fixed sentence shape (`"A voice that is …; …; …."`) means **every** caption looks
the same. When these captions are used as **fine-tuning targets** for a voice-acting model,
that rigid surface form is an easy thing to overfit to — the model learns the template, not
the voice. So the same phrase content can be rendered through **11 interchangeable
templates** plus optional **dimension shuffling**.

Crucially, **only the arrangement changes.** Every template passes through the same pipeline
— z-scores, top-k selection, always-on identity dims, absolute-band categoricals, the
genuineness gate, synonym rotation. The phrase *content* is identical; templates reorder the
four phrase groups (identity = Age/Gender/Register/Tempo · timbre = the other VoiceNet dims ·
emotion = EmoNet · quality = genuineness/vocal-burst) and vary the connectives. Captions stay
information-preserving and comparable.

| template            | example surface form (same clip) |
|---------------------|----------------------------------|
| `default`           | *A voice that is very bright in oral resonance; …; young and youthful; masculine and deep-pitched; …; deeply and genuinely felt.* |
| `identity_first`    | *Young and youthful, masculine and deep-pitched, low and bassy in register, somewhat fast in tempo. It sounds very bright in oral resonance; …. Emotionally, it is …* |
| `emotion_first`     | *Emotionally extremely carrying amusement, …. The voice itself is very bright in oral resonance; …; masculine and deep-pitched; ….* |
| `telegraphic`       | *Very bright in oral resonance, notably forward in mask resonance, …, masculine and deep-pitched, ….* (bare comma list) |
| `two_sentence`      | *How it sounds: … timbre + identity …. What it conveys: … emotions + genuineness ….* |
| `sounds_like`       | *Sounds like a young and youthful masculine and deep-pitched voice, low and bassy in register, fast in tempo. It is …. Emotionally, ….* |
| `bulleted`          | *Identity: … · Timbre: … · Emotion: … · Delivery: …* |
| `varied_connectors` | *A voice that is very bright in oral resonance, with notably forward in mask resonance and …, carrying ….* |
| `quality_led`       | *Deeply and genuinely felt; interwoven with vocal bursts. The voice is …. Emotionally, ….* |
| `minimal_identity`  | *A young and youthful masculine and deep-pitched voice extremely carrying amusement, …. Also low and bassy in register, ….* |
| **`tags`**          | *very warm, very breathy, very meek, elderly, masculine, low-register, slightly slow, very lust, very heartache, genuine, with-bursts* |

The **`tags`** template is special: one short tag per salient dimension, about-average dims
dropped entirely, comma-joined, **no filler**. It is the **default for the burst pipeline**
(`PROC_TEMPLATE=tags`); set `PROC_TEMPLATE` to any prose template if you prefer sentences.

```python
from caption import caption, caption_detail, TEMPLATE_NAMES

caption(preds, base, template="identity_first")
caption(preds, base, template="random", synonym_seed=1234)                       # deterministic pick
caption(preds, base, template="random", synonym_seed=1234, shuffle_dims=True)    # + permuted order
caption_detail(preds, base, template="random", synonym_seed=1234)["template"]    # -> chosen name
```

- **`template`** — one of `TEMPLATE_NAMES`, or `"random"` to pick one deterministically from
  the clip's `synonym_seed`. `caption()` defaults to `"default"`, so existing callers are
  byte-for-byte unchanged.
- **`shuffle_dims`** — deterministically permutes the non-identity timbre dims and the
  emotions within their groups. Display order only; never *which* dims were selected. The
  always-on identity dims keep their slots.

Everything is deterministic: the same `synonym_seed` always yields the same template,
shuffle and wording — **including across processes**. (It did not, until recently: the seed
was mixed with Python's builtin `hash()` of the emotion name, which is salted per process by
`PYTHONHASHSEED`, so the same seed produced different synonyms in different runs. Everything
that mixes a name into a seed now goes through `caption.stable_hash()`.)

---

## The gates

### The genuineness gate

Emotion wording is only as strong as the delivery is believable. Let `zg` be the genuineness
z-score:

| condition | gate | effect on emotions | genuineness descriptor |
|---|---|---|---|
| `value ≥ genuineness median` | `open` | full intensity allowed (up to Extremely) | "deeply and genuinely felt" / "genuine …" |
| below median but `zg > −1` | `capped` | intensity capped at **Notably** | "only slightly genuine, somewhat performed" |
| `zg ≤ −1` | `dropped` | emotion phrases **dropped entirely** | "measured and performed rather than genuine" |

This prevents an over-acted, low-genuineness clip from being captioned as "extremely
enraged" when the anger is performed rather than real.

### The gender gate (Empathic-Insight)

The VoiceNet `GEND` dimension always emits *some* gender phrase, which is risky on voices
whose gender is not clearly perceptible (soft whispers, ASMR, children, stylized characters).
A second, dedicated signal — the **Empathic-Insight Gender expert** (`model_Gender_best.pth`,
bipolar **−2 = very masculine … +2 = very feminine**) — is used purely as a **confidence gate**:

- **|EI gender| ≥ `EI_GENDER_GATE`** (default 0.5) → gender is confidently perceptible →
  keep the well-calibrated **VoiceNet `GEND`** label.
- **|EI gender| < gate** → **omit the gender clause entirely** rather than assert a
  possibly-wrong one.

Why gate-only and not label: an A/B study on ASMR clips
([live grid →](https://projects.laion.ai/procedural-voice-captions/gender-ab/), Gemini-3.5-Flash
as ground truth) found the EI expert tracks perceived gender about as well as VoiceNet in
*direction* (sign-agree 30–31 / 32) but **compresses clearly-masculine voices toward neutral**
(raw ≈ −0.2 where Gemini says *very masculine*). So EI is trustworthy at the extremes it was
trained to separate — ideal as a gate — while VoiceNet gives the better graded label above it.
The study also confirmed the "soft male ASMR sounds feminine" captions are *correct*: Gemini
hears those generations as feminine too. Set `EI_GENDER_GATE=0` to disable.

### The `no_burst` gate

Described [above](#the-vocal-burst-path): `P(no_burst) ≥ 0.5` (`BURST_NOBURST_GATE`) discards
a located span outright. Every discarded span is still returned in `variant_a_bursts` with
`kept: false` and the reason, so the gate is auditable rather than silent.

---

## On-the-fly caption augmentation — *score once, caption many times*

A caption is a pure function of the stored **scores** + a template + a seed, so you don't have
to freeze one caption per clip. Run the scoring models **once**, store only the raw scores
(per sentence and global) with the transcript and confirmed bursts, then regenerate a
*different* caption every epoch — different template, synonym rotation, shuffle, tags vs prose
— with no audio and no model calls at train time (microseconds of string formatting).

This is a cheap, strong **text augmentation** for TTS / caption models: the target *wording and
structure* vary across epochs while the *content* (which dims/emotions are salient, the bursts,
the words) stays fixed, so the model learns the score→text mapping instead of memorizing one
phrasing.

```python
from burst_captions import BurstCaptioner
from augment import score_record, augment_script
import json

# 1) score ONCE, store the compact record (then you can discard the audio)
cap = BurstCaptioner()
rec = score_record(cap.process("clip.wav"))          # global + per-sentence raw scores + bursts
json.dump(rec, open("clip.scores.json", "w"))

# 2) at train time, per example per epoch — fresh phrasing each call
text = augment_script(rec, seed=epoch * 1_000_003 + example_id)   # random template each epoch
text = augment_script(rec, seed=epoch, template="tags")           # or force the terse tags form
```

`process()` already attaches the raw scores it computes — `result["scores"]` (global) and
`result["sentences"][i]["scores"]` — so `score_record()` is just a distillation, no extra
inference. The gender gate is applied automatically from the stored `ei_gender`.
See [`augment.py`](augment.py) (`python augment.py` prints four varied captions from one record).

---

## The scoring models

All run on top of the **VoiceCLAP-commercial** 768-d embedding, except EmoNet, which uses
BUD-E-Whisper.

| group | what | model |
|---|---|---|
| embedder | 768-d speech embedding | [`laion/voiceclap-commercial`](https://huggingface.co/laion/voiceclap-commercial) |
| VoiceNet | 57 voice dimensions (0–6), incl. Valence `VALN` & Arousal `AROU` | [`laion/voicenet-dimension-predictors-commercial`](https://huggingface.co/laion/voicenet-dimension-predictors-commercial) |
| quality | genuineness (0–6) | [`laion/voiceclap-commercial-genuineness`](https://huggingface.co/laion/voiceclap-commercial-genuineness) |
| quality | vocal-burst blend (0–10) | [`laion/voiceclap-commercial-vocalburst-blend`](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend) |
| EmoNet | 40 emotions | [`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) (BUD-E-Whisper encoder + per-emotion heads) |
| ASR | word + sentence timestamps | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| burst locator | 50 fps burst probability, 30 s window | [`laion/vocalburst-locator`](https://huggingface.co/laion/vocalburst-locator) · `model_v2.pt` |
| burst classifier | names a cut span, 82 classes + `no_burst` | [`laion/vocal-burst-detector-v2`](https://huggingface.co/laion/vocal-burst-detector-v2) |

**Valence / Arousal rule:** Valence and Arousal are taken from VoiceNet's `VALN` / `AROU`
heads. EmoNet is used for its **40 emotions only** — EmoNet's own valence/arousal axes are
never loaded and never double-counted.

The dimension names, level rubrics and emotion **synonym clusters** come from
[**LAION-AI/voice-taxonomies**](https://github.com/LAION-AI/voice-taxonomies).
EmoNet is optional: `BC_EMONET=0` skips it and captions fall back to VoiceNet + genuineness.

### Upstream: sentence cutting + non-verbal tags

The captioner consumes per-clip predictions; producing well-segmented clips with timestamps
and non-verbal event tags is an upstream step. Recommended models:

- **ASR + timestamps:** [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
  [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR),
  [`mrfakename/vibevoice-asr-soundscapes-events-promptfix-vb4x-12k-fft-20260603`](https://huggingface.co/mrfakename/vibevoice-asr-soundscapes-events-promptfix-vb4x-12k-fft-20260603),
  [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- **Non-verbal / soundscape tags:** [`fluxions-ai/whisperdrz`](https://huggingface.co/fluxions-ai/whisperdrz)
- **Taxonomy source of truth:** [`LAION-AI/voice-taxonomies`](https://github.com/LAION-AI/voice-taxonomies)

The fine-tuning prompt format that consumes these captions is described here:
[MOSS-Local 1.5 Voice-Acting — fine-tuning prompt format](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/finetuning_prompt_format.html).

---

## How the baseline was built

`baseline_stats.json` describes "the average voice" so deviations are meaningful. It was
built (`compute_baseline.py`) from two sources:

**1. Emolia — ~1000 random clips per language** (`en, de, zh, fr, ko, ja`):
- VoiceNet / genuineness / blend are computed by running the heads directly on
  **precomputed VoiceCLAP-commercial embeddings** (no audio needed).
  Per-language n: **en 1000, de 1000, zh 1000, fr 687, ko 562, ja 454** → **4703** clips.
- The 40 EmoNet emotions require audio, so they come from a **bounded random subset**:
  ~80 random clips per shard across a few shards per language, streaming each shard tar and
  deleting it after use → **722** clips (en 240 / de 160 / zh 82 / fr 80 / ko 80 / ja 80).
  This reduced-n choice is recorded in the JSON `_meta`.

**2. ~1000 random takes from
[`laion/moss-character-voices-bestof64`](https://huggingface.co/datasets/laion/moss-character-voices-bestof64)**
— extreme, best-of-64 character voices (dragon, fairy, goblin, ASMR, …). Their precomputed
`dims_json` / `genu` / `blend` are folded into the VoiceNet / quality baselines to
**deliberately widen the spread**. Their per-character `emo_json` is **not** used for the
40-emotion baseline (Emolia audio is).

**Per-dimension statistics.** For each dimension we store mean, median, std, MAD, p10, p90,
n, and a robust **`spread`** for z-scoring:

> `spread = 1.4826 · MAD`, falling back to `std` when `1.4826·MAD < 0.5·std`.

The MAD fallback matters for the EmoNet emotions, which are strongly zero-inflated (most
clips ≈ 0, so the raw MAD collapses and would blow up z-scores).

| code | name | group | mean | median | std | spread | n |
|------|------|-------|------|--------|-----|--------|---|
| AGEV | Voice Age | voicenet | 2.87 | 2.68 | 1.26 | 1.19 | 5101 |
| GEND | Perceived Gender | voicenet | 3.12 | 3.71 | 1.74 | 1.88 | 5004 |
| REGS | Register | voicenet | 1.73 | 1.31 | 1.30 | 1.18 | 5101 |
| TEMP | Tempo | voicenet | 2.19 | 2.14 | 0.97 | 1.05 | 4703 |
| AROU | Arousal | voicenet | 2.53 | 2.45 | 1.24 | 1.22 | 5004 |
| VALN | Valence | voicenet | 2.36 | 2.33 | 1.00 | 1.02 | 4703 |
| genuineness | Genuineness | quality | 2.05 | 1.73 | 1.41 | 1.45 | 5703 |
| blend | Vocal-burst blend | quality | 2.72 | 2.21 | 2.32 | 2.33 | 5703 |
| Anger | Anger | emonet | 0.33 | 0.03 | 0.45 | 0.45 | 722 |
| Amusement | Amusement | emonet | 0.20 | 0.00 | 0.46 | 0.46 | 722 |
| Interest | Interest | emonet | 1.83 | 1.90 | 0.53 | 0.58 | 722 |

(VoiceNet / quality dims get `n ≈ 4703 + bestof64` where bestof64 covers that dim.)

### Rebuild it

```bash
export HF_HOME=/path/to/hf_cache HF_TOKEN=...     # needs access to the gated repos above
python compute_baseline.py --stage emolia_vn      # CPU: heads on precomputed embeddings
python compute_baseline.py --stage bestof64       # CPU: read precomputed scores
python compute_baseline.py --stage emonet --gpu 0 # 1 GPU, small batch: BUD-E-Whisper
python compute_baseline.py --stage merge          # write baseline_stats.json
```

The paths to the Emolia index, embeddings and heads are constants at the top of
`compute_baseline.py`; adjust them for your environment.

---

## Evaluation notes

### Procedural vs LLM-assisted — throughput & quality

Measured on 1×A100, n=32, Gemini-3.5-Flash scoring 1–10. This repo produces the
**procedural** caption directly; the
[Comprehensive Voice-Acting Annotation Pipeline](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline)
feeds the same annotations to a small LLM (Gemma-4-E4B, text-only).

| path | s/clip | clips/s | overall | emotion | burst&nbsp;acc | gender | wins |
|------|--------|---------|---------|---------|----------------|--------|------|
| **procedural** (this repo) | **0.66** | **1.52** | 7.28 | 7.55 | 7.97 | 9.19 | 3 |
| **LLM-assisted** ([VAP repo](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline)) | 2.05 | 0.49 | **8.25** | **8.36** | **7.97** | **9.91** | **29** |

The LLM path **rewords the procedural draft** but **burst positions are forced to the
procedural ones** (the LLM only rewrites the GENERAL line and the delivery cues; each
sentence's text + inline bursts are copied verbatim). That is why **burst_accuracy is
identical (7.97)** for both, while overall/emotion improve. An A/B of four placement
strategies confirmed this hybrid beats letting the LLM re-place bursts from a list, which
tended to *drop* real bursts.

- **Procedural** — ~3× faster, fully deterministic, no GPU LLM, trivially re-augmentable.
  Terser, more mechanical wording. Best for **large-scale labeling, training targets, and
  on-the-fly augmentation**.
- **LLM-assisted** — same burst placement, more fluent GENERAL + cues, can be coloured by a
  (non-gendered) archetype; wins the head-to-head 29–3, at ~1.4 s/clip extra. Best when
  **fluency and per-clip polish matter**.

Both keep gender **correct or correctly unstated** (9.19 / 9.91) thanks to the gender gate.

### Historical: the locator-v1 threshold grid

A Gemini-3.5-Flash sweep over the character voices settled `BURST_LOCATOR_THR=0.75` for the
**v1** locator + multi-label classifier:

| thr | precision | recall | kept | hallucinated |
|-----|-----------|--------|------|--------------|
| 0.70 | 8.09 | 8.84 | 42 | 18 |
| **0.75** | **8.66** | **8.78** | 38 | 12 |
| 0.80 | 8.59 | 8.75 | 37 | 9 |
| 0.85 | 8.72 | 8.38 | 33 | 6 |
| 0.90 | 8.75 | 8.50 | 28 | 7 |

**This grid no longer sets the default.** It was measured on `model.pt`, whose probabilities
are calibrated differently, and it was a *relative* judge study rather than an F1 measurement
against burst timestamps. The current 0.50 comes from the v2 model card's sweep on 992 real
held-out clips. The table is kept because the shape of the trade-off (buying precision costs
recall fast) still holds, and because the earlier
[config study](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline/blob/main/CAPTION_CONFIG_EVAL.md)
is what established that **inline placement (Variant A) beats sentence-level (Variant B)** as
the default — a conclusion unaffected by the model swap. The v1-era classifiers
([single](https://huggingface.co/laion/vocalburst-classifier-single),
[multi-label](https://huggingface.co/laion/vocalburst-classifier-multilabel)) are superseded
by `laion/vocal-burst-detector-v2`.

---

## Known issues

- **`R_MIXD` ("mixed resonance") dominates almost every caption.** Its baseline `spread` is
  0.20, an order of magnitude tighter than most dimensions, while the head's output on real
  audio sits well below the baseline median — so its z-score lands around **−7** and it wins
  the top-5 selection on 13 of the 14 demo clips (mean |z| 7.39, against a median of 0.96
  across the other 56 dims). That is why nearly every example above starts with
  *"very flat-resonance"*. `R_MIXD` is also the weakest-correlating regression head in the
  VoiceNet release (val Pearson r = 0.392). This is a **baseline/head scale mismatch that
  predates the burst work** and is not fixed here — fixing it means recomputing that
  dimension's baseline against the same embedding path used at inference.
- The demo page's long-form example is a **concatenation of shorter clips**, not a natural
  long recording — nothing longer than ~18 s ships in this repo. It exercises the 30 s
  windowing honestly, but it is not a substitute for evaluation on real long-form audio.

---

## Files

```
caption.py               # caption() / caption_detail() + 11 templates + gates  (stdlib only)
baseline_stats.json      # the baselines (99 dimensions + _meta)
burst_captions.py        # full pipeline: score -> timestamps -> locate -> classify -> insert
asr_words.py             # bundled Parakeet token -> word -> sentence helpers (stdlib only)
augment.py               # score once, caption many times (train-time text augmentation)
compute_baseline.py      # rebuild baseline_stats.json (staged)
build_burst_demo_v2.py   # renders docs/burst-captions-v2/ from audio already in this repo
build_burst_demo.py      # renders the older docs/burst-captions/ (needs out-of-repo wavs)
vocalburst_taxonomy.json # 82 vocal-burst classes (class order for the classifier)
examples/                # real complete predictions (dims + emo + genu + blend)
docs/                    # the published GitHub Pages demos
requirements.txt
```

---

## Demo pages

Published at [`projects.laion.ai/procedural-voice-captions/`](https://projects.laion.ai/procedural-voice-captions/)
and [`laion-ai.github.io/procedural-voice-captions/`](https://laion-ai.github.io/procedural-voice-captions/).
Each has real audio players — GitHub READMEs cannot embed audio, so these are links, not embeds.

| page | what |
|---|---|
| [**burst-captions-v2**](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/) | **Current defaults.** 14 clips re-annotated with locator `model_v2.pt` + `vocal-burst-detector-v2`, including a 91 s clip that needs 4 locator windows. Each card shows the located spans and, where available, the previous pipeline's output for comparison. |
| [caption grid](https://projects.laion.ai/procedural-voice-captions/) | 100 real multilingual clips, captioned, with the 11 templates assigned round-robin. |
| [character-captions](https://projects.laion.ai/procedural-voice-captions/character-captions/) | LAION character voices: procedural vs LLM-assisted, side by side. |
| [burst-captions](https://projects.laion.ai/procedural-voice-captions/burst-captions/) | The previous burst pipeline (locator v1 + multi-label classifier). Kept for comparison. |
| [gender-ab](https://projects.laion.ai/procedural-voice-captions/gender-ab/) | The VoiceNet-vs-Empathic-Insight gender study behind the gender gate. |
| [moss-thinking](https://projects.laion.ai/procedural-voice-captions/moss-thinking/) | The same clips through the MOSS-Audio-Thinking reasoning models (4B & 8B), which listen to the audio and fuse the procedural caption + transcript + taxonomies into the final voice-acting format. |
| [vocalburst-*](https://projects.laion.ai/procedural-voice-captions/vocalburst-predictions/) | Earlier burst-model studies: predictions, single-burst, detect→classify combo, validation. |

---

## License / attribution

Built for the LAION voice-acting effort. Models and taxonomies are the respective
LAION / third-party releases linked above.

## Changelog

- **Locator v2 + classifier v2 are the defaults.** `laion/vocalburst-locator` now loads
  **`model_v2.pt`** (event F1 0.607 vs 0.152 on 992 real clips) and the naming stage is
  **`laion/vocal-burst-detector-v2`**, replacing `vocalburst-classifier-single` /
  `-multilabel`. Post-processing moved to **0.50 / 0.10 / 0.10** (was 0.65 / 0.30 / 0.50);
  the transient duration floor is off by default. Five hand-gesture classes are masked and
  can never be predicted.
- **Audio longer than 30 s is handled properly.** The locator's fixed 30 s window is scanned
  in overlapping windows whose frame probabilities are cross-faded onto one global timeline
  before events are extracted, so seam-straddling bursts are neither lost nor duplicated.
- **The repo runs from a clean checkout.** Every model location now defaults to a public HF
  repo instead of a hard-coded `/run/user/1001/...` path, the Parakeet word/sentence helpers
  are bundled (`asr_words.py`) instead of imported from an out-of-repo directory, and audio
  decoding falls back to `soundfile` when no system `ffmpeg` is present.
- **Captions are deterministic across processes.** Name-derived seeds used Python's salted
  builtin `hash()`; they now use `caption.stable_hash()`.
- **Pause markers** — `[pause X.Xs]` is inserted into the SCRIPT from the ASR word timestamps
  at any silent gap ≥ `BURST_PAUSE_THR` (default 0.30 s), within and between sentences; `0`
  disables. The markers are copied verbatim into the LLM-reword output.
