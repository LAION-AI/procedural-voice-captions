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
GENERAL: very warm, very soft-onset, very flat, very mumbled, very organic, middle-aged,
         low-register, very slow, pining, helplessness, unease, semi-genuine, with-bursts
SCRIPT:
(very flat, very smooth, very non-narrative, no interest, very fatigue, no deep focus, genuine)
    Huh. (Chuckle)
(very flat, very warm, very smooth, helplessness, concern, no interest, semi-genuine, with-bursts)
    [pause 0.4s] What have we here?
(very whispered, very flat, very warm, feeling turned on, nostalgia, fondness, semi-genuine, with-bursts)
    [pause 0.5s] A lost little traveller with shiny pockets?
```

The `(Chuckle)` is not in the transcript — Parakeet never heard a word there. The locator
found a burst at 0.22–0.70 s, the classifier named it, and it was written between the two
words nearest that moment.

**2 — an in-the-wild podcast clip** ([listen ↗](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/#EN_B00000_S03298_W000123))

```
GENERAL: very oral-bright, very chesty, very full, very naturalistic, very throaty, adult,
         masculine, low-register, very existential void, very meditation, no ribbing,
         genuine, with-bursts
SCRIPT:
(very oral-bright, very dialogic, very throaty, no interest, no brooding, disdain, genuine, burst-free)
    So that people go, What is that? (Breathy Giggle)
(very oral-bright, very throaty, very commanding, no fascination, no thoughtfulness, no relaxation, genuine)
    I need to find out about that.
(very oral-bright, very throaty, very chesty, very existential void, very pensiveness, no fascination, genuine)
    [pause 0.7s] So this is me making stuff up right in the moment, but I'm like, okay, I'm
    gonna use these oblique strategies as a way of
```

**3 — a distressed character voice** ([listen ↗](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/#zombie5))

```
GENERAL: very vulnerable, very meek, very scattered, very soft-onset, very flat, young,
         feminine, low-register, very slow, lethargy, feeling horny, warmth, semi-genuine
SCRIPT:
(very soft-onset, very scattered, very flat, lethargy, submission, insensitivity, semi-genuine, with-bursts)
    So hungry.
(very flat, very scattered, very smooth, burnout, no fascination, sullenness, semi-genuine, with-bursts)
    [pause 0.5s] Hmm. (Wistful Sigh)
(very scattered, very flat, very mumbled, desperation, pessimism, carnal desire, semi-genuine, with-bursts)
    [pause 1.3s] Help me.
(very scattered, very soft-onset, very flat, very burnout, very desperation, very unease, genuine, with-bursts)
    [pause 0.3s] I can't stop.
```

Those examples use the terse **`tags`** surface form, which is the default. The same
content — same dimensions, same emotions, same gates — can be rendered as prose through
[11 interchangeable templates](#caption-templates--dimension-shuffling). Example 1's GENERAL
line under `template="default"`:

```
A voice that is extremely warm and enveloping; extremely soft and gentle in onset; extremely
flat and unemphasized; extremely blurry and mumbled; extremely organic and soft; middle-aged
to mature; low and bassy in register; very slow and deliberate in tempo; notably carrying
pining; notably carrying helplessness; notably carrying unease; only slightly genuine,
somewhat performed; interwoven with vocal bursts (laughs, gasps, sighs).
```

> `very flat` above is **Emphasis** (`EMPH`) below baseline, not resonance. Captions produced
> before the [baseline swap](#the-baseline) opened almost every line with *"very
> flat-resonance"* (`R_MIXD`); that was a scaling defect and is fixed.

### Hear the clips

GitHub's README renderer **cannot embed an audio player** — the links above and below are
ordinary links out to the GitHub Pages demo, where each clip has a real `<audio>` element
next to its caption. Nothing plays inside this page.

- ▶ **[25 clips through the current stack](https://projects.laion.ai/procedural-voice-captions/burst-captions-25/)**
  — locator v2 + classifier v2 + Empathic-Insight + the in-domain baseline with reliability
  weighting. Each card shows the z-score, the reliability `r` and the effective rank `|z|·r`
  of every selected dimension, and expands to the caption the *same* scores produced under
  the previous captioner.
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
Dimensions are then ranked by an **effective score `|z| · r`**, where `r` is the
dimension's measured predictor reliability — see [reliability
weighting](#reliability-weighting--z--r). Then, for the whole clip:

| | selected | of |
|---|---|---|
| VoiceNet dimensions | **top 5 by \|z\|·r** | 57 |
| EmoNet emotions | **top 3 by \|z\|** (no published per-emotion reliability) | 40 |
| Always included regardless of rank | **Age `AGEV` · Gender `GEND` · Register `REGS` · Tempo `TEMP`** | — |
| Also appended | genuineness, and vocal-burst blend when \|z\| ≥ 0.5 | — |

(`k_voicenet=5`, `k_emonet=3`, `ALWAYS_ON = ["AGEV","GEND","REGS","TEMP"]` in
[`caption.py`](caption.py) — those are the defaults of `caption()` / `caption_detail()`.)

Three selection rules are worth knowing:

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
- **Reliability decides ranking, never wording.** `|z|·r` picks *which* dimensions appear;
  the reported `z`, the direction and the intensity adverb all stay on the raw `|z|`,
  because the deviation is a fact about the clip while `r` is a fact about the predictor.

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
become `[pause X.Xs]`, both inside a sentence and between sentences — but the gap has to be
measured correctly first, which it was not. See [Pauses](#pauses--why-a-tdt-word-span-is-not-a-word).

---

## Pauses — why a TDT word span is not a word

`[pause X.Xs]` is computed from one subtraction: `next_word.start − this_word.end`. That is
only a silence if `this_word.end` is where the word stops. **It is not.**

A Parakeet-TDT token carries a predicted *duration*, and that duration is the encoder advance
**to the next token**, not the extent of the token's own sound. Consecutive spans therefore come
out exactly contiguous — `('My', 1.52, 1.84) ('cousin', 1.84, 2.72)` — and the silence inside a
pause is billed to the word *before* it. Measured on 96 of this repo's own demo clips
(1,763 word pairs, `docs/audio/*.mp3`, Parakeet-TDT-0.6b-v3):

| | median gap | gap exactly 0.000 s | gaps ≥ 0.30 s |
|---|---:|---:|---:|
| raw token pairs | 0.000 s | **82.2 %** | 5.9 % |
| raw word pairs | 0.000 s | **65.5 %** | 11.4 % |

So the markers were not absent — TDT does emit real gaps when its duration head saturates — but
they were **systematically incomplete and systematically short**. Against the audio itself
(a word's speech end = the last 25 ms frame whose RMS clears 10 % of the clip's 90th-percentile
frame RMS):

* **313** pauses ≥ 0.30 s are audible; the shipped code printed **233**. Every one of those 233
  was real — **zero false positives** — but **80 (25.6 %) were missing entirely**.
* The ones it did print were **too short**: understated by a median 0.02 s, a mean 0.134 s, a
  90th percentile of 0.418 s and a worst case of **1.22 s**. Only 47.6 % were essentially exact.
* Nothing hangs on the silence floor: sweeping it over 0.03 … 0.30 moves the true count over
  273 … 348 and the old recall over 0.853 … 0.670.

### The fix is energy, not a cap

The obvious repair — cap every word extent at some fixed duration — was measured here and
**rejected**, because most long spans are simply long words:

| `WORD_CAP` | markers | true | invented | missed | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 s | 527 | 287 | 240 | 26 | 0.545 | 0.917 |
| **0.30 s** | 431 | 277 | **154** | 36 | **0.643** | 0.885 |
| 0.50 s | 299 | 245 | 54 | 68 | 0.819 | 0.783 |
| 1.00 s | 238 | 234 | 4 | 79 | 0.983 | 0.748 |
| none (shipped) | 233 | 233 | 0 | 80 | 1.000 | 0.744 |

Of the 910 words whose span exceeds 0.30 s, **681 are genuinely long** — German compounds,
drawn-out vowels — and a blind cap turns each of them into a pause that nobody said. There is no
fixed cap that buys recall without buying invented silence with it.

So `burst_captions.trim_word_ends()` uses the waveform, which is already in hand:
each word's `end` is moved back to the last frame above the silence floor. The rule is
one-directional and cannot invent a pause — an end only ever moves **earlier**, never before the
word's own start, and only across frames that are actually quiet; a word with no frame above the
floor is left exactly as it was. The raw TDT span is kept as `end_span`, so both readings are
available and nothing that wants the old number has to guess.

On the same 96 clips this takes the marker count from **233 to 291 (+25 %)** and the share of
word pairs reporting a gap of exactly 0.000 s from **65.5 % to 1.9 %**. It is a small edit to
each timestamp — median 0.005 s, mean 0.033 s, 3.6 % of words untouched — and an independent
level check says the 58 new pause windows peak at a median of **−28.1 dBFS**, against
−19.4 dBFS for the pauses the code already accepted and −6.3 dBFS for speech. The new markers
sit in *quieter* audio than the old ones.

| env | default | |
|---|---|---|
| `BURST_WORD_END` | `energy` | `raw` restores the previous, contiguous-span behaviour |
| `BURST_SILENCE_FLOOR` | `0.10` | silence floor as a fraction of the clip's p90 frame RMS |
| `BURST_PAUSE_THR` | `0.30` | unchanged — and note it no longer shares a value with any word-duration cap, because there is no cap |

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

### A second, narrower classifier — `laion/vocal-burst-detector-x2`

There is a newer naming head, and it is **better and smaller-mouthed at the same time**. It is
wired in as an *addition*; the default output of this repo does not change until you ask it to.

`vocal-burst-detector-x2` names a span over **17 classes = 16 bursts + `no_burst`**, on a frozen
`laion/voiceclap-large-v2` embedding (3584-d) through a five-member ensemble of
`3584 → 256 → 17` MLPs averaged in probability space. On held-out audio, with chance at 0.0588:

| head | encoder | dim | val acc | real, all | real, family | load |
|---|---|---:|---:|---:|---:|---:|
| **`large-v2`** (default) | `laion/voiceclap-large-v2` | 3584 | **0.7056** | **0.7536** | **0.8480** | ~101 s |
| `commercial` | `laion/voiceclap-commercial` | 768 | 0.6258 | 0.7039 | 0.7954 | ~0 s |

`large-v2` wins every cell, so it is the default. `commercial` is the selectable alternative and
it is nearly free *here specifically*: `BurstCaptioner` already holds the
`laion/voiceclap-commercial` encoder for the VoiceNet heads and the v2 classifier, so the
commercial x2 head costs five small MLPs — about 1 MB — and no second tower. (A third head on
`voiceclap-small-v2` scores between them and is **not** wired in: it is CC BY-NC 4.0.)

#### The catch, stated first: the new vocabulary is *narrower*, not different

**All 16 x2 classes already exist in this repo's 82-class `vocalburst_taxonomy.json`.** The new
head adds no vocabulary at all — it removes 66 classes, and with them seven of the taxonomy's
sixteen groups entirely:

| group | classes the x2 head can still say | of |
|---|---:|---:|
| sighs | 3 | 4 |
| breathing | 3 | 7 |
| groans_and_moans · humming · laughter | 2 each | 4 · 4 · 8 |
| gasps_and_inhales · grunts · screams_and_shrieks · coughing_and_sneezing | 1 each | 3 · 3 · 2 · 7 |
| **crying_and_distress · eating_and_drinking · hand_and_body_sounds · mouth_and_lip_sounds · throat_and_vocal_sounds · tongue_clicks · whistling** | **0** | 5 · 6 · 4 · 9 · 7 · 4 · 5 |

On 61,579 rows / **12,894 burst events** of the annotated LAION-TTS corpus, **83.8 % carry a
label the x2 head cannot emit**: `Low Mumble` 26.3 %, `Ahem` 26.2 %, `Contented Sigh` 20.5 %,
`Surprised Gasp` 7.2 % — the four commonest labels in the corpus, none of them expressible.
Swapping the classifier would delete exactly the sounds that occur most.

So the choice of vocabulary is explicit, per `BURST_LABEL_SOURCE`:

| value | who names a span | what it costs |
|---|---|---|
| **`v2`** (default) | the 83-class head, exactly as before | nothing changes; x2 is recorded beside every span and captions are byte-identical |
| `x2` | the 17-class head, on every span | honest, and much poorer coverage — 83.8 % of events lose their name |
| `union` | the 17-class head **only where the v2 label is one of its 16 classes**; otherwise the v2 label stands, flagged `out_of_x2_vocab` | the accuracy of the new head exactly where it applies, the coverage of the old one everywhere else |

#### Certainty is written to match the measured recall

`per_class_recall.json` ships with the detector and is bundled here as
`vocalburst_x2_recall.json` / `vocalburst_x2_recall_commercial.json`. Its own README puts the
rule plainly: *"a class recalled at 20 % cannot show a hit rate meaningfully above 20 %."*
Before this change the repo asserted every class with the same certainty — `(Sharp Inhale)`
(recalled 0.897 on real audio) and `(Relief Sigh)` (recalled 0.062) were written the same way.
They no longer are:

| tier | condition (recall on real audio) | written | `large-v2` classes |
|---|---|---|---|
| `named` | strict ≥ `BURST_X2_STRICT_MIN` (0.50) | `(Chuckle)` | Affirmative Grunt, Chuckle, Frustrated Groan, Panting, Scream, Sharp Inhale |
| `hedged` | strict ≥ `BURST_X2_HEDGE_MIN` (0.25) | `(Breathy Giggle?)` | Breathy Giggle, Exhausted Groan, Humming, Soft Hum, Yawn |
| `family` | family recall ≥ `BURST_X2_FAMILY_MIN` (0.50) | `(Breath)` | Deep Breath, Heavy Breathing |
| `generic` | nothing clears | `(Vocal Burst)` | Exasperated Sigh, Relief Sigh, Wistful Sigh |

The `family` tier is the one that earns its keep: the head confuses `Deep Breath` with
`Heavy Breathing` far more than it confuses either with a laugh — strict recall 0.186, family
recall 0.712 — so naming the *family* is a true statement where naming the class is a coin flip.
The `generic` tier falls back to what the binary burst/no-burst discriminator supports on its
own, which is 98.6 % accurate for `large-v2`. In Variant B the `hedged` tier also gets its own
phrasing (*"with what sounds like a (Yawn?)"* rather than *"with an audible (Yawn)"*); the other
tiers keep the six original templates, because their uncertainty is already in the label.

`BURST_X2_RECALL_SOURCE` picks which measurement the thresholds read — `real` (the in-the-wild
held-out set, default and conservative), `dramabox` (in-domain), or `min` (the worse of the two).

#### What lands in the output

Every span in `variant_a_bursts` carries `x2_label`, `x2_prob`, `x2_top`, `x2_family`,
`x2_p_noburst`, `x2_tier` and `x2_recall` whether or not x2 wrote the caption, plus
**`written`** (the text that went into the script), **`vocab`** (`v2-83` / `x2-17` /
`x2-17-family` / `generic`) and `x2_agrees`. `result["x2"]` records the head, the encoder, the
thresholds and the recall source, so a stored result can always be read back against the
configuration that produced it.

The x2 head **never removes a burst**. The v2 `no_burst` gate alone decides whether a span is a
burst; a 16-class head has no vocabulary for most of what it would be vetoing. `BURST_X2_GATE`
enables a second gate for anyone who wants it, and records the rejection rather than hiding it.

```bash
# name spans with both heads, let the wider vocabulary keep what the narrower one cannot say
python burst_captions.py clip.wav --x2 --label-source union

# the cheap head — no second encoder, because voiceclap-commercial is already loaded
BURST_X2=1 BURST_X2_HEAD=commercial python burst_captions.py clip.wav

# what the thresholds do to each head's vocabulary, no models needed
python burst_x2.py
```

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
python -m unittest discover        # 60 tests, stdlib only, no models and no network
```

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # the LAION model repos
export HF_HOME=/path/to/hf_cache

# full pipeline on audio -> per-clip JSON (+ mp3s for a demo page)
python burst_captions.py 'my_clips/*.wav' --out out.json --mp3-dir mp3/

# caption from already-computed predictions, no audio, no models, stdlib only
python caption.py examples/worker_0_EN_tDPU-wXSB5y_W000085.json --kv 5 --ke 3

# same clip against the previously published baseline, ranked by raw |z|
python caption.py examples/worker_0_EN_tDPU-wXSB5y_W000085.json --baseline emolia --no-reliability
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

### Emotions come from Empathic-Insight, and only from there

The 40 emotions and the gender gate are scored by
[`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)
(BUD-E-Whisper encoder + one head per emotion). There is **no distilled-head path in this
repo and none is being added.**

The alternative would be the distilled MLP heads in
[`laion/voiceclap-commercial-attribute-heads`](https://huggingface.co/laion/voiceclap-commercial-attribute-heads)
— 61 tiny heads on the same frozen VoiceCLAP-commercial embedding the VoiceNet dims already
use, so they are nearly free once the clip is embedded, where Empathic-Insight needs a
second encoder pass. They were compared against Empathic-Insight on **72,809 matched pairs
of identical DramaBox audio** (20,000 whole utterances and 52,809 sentences of ≥ 8 words):

| level | dims compared | median Spearman ρ | median MAE | ρ ≥ 0.7 | ρ < 0.5 |
|---|---|---|---|---|---|
| utterance | 44 | **0.381** | 0.438 | 1 | 34 |
| sentence | 44 | **0.315** | 0.476 | 3 | 35 |

**Spearman is the number that decides this**, because a caption only ever uses the **top-k
dimensions by |z|** — rank agreement is exactly the question of whether the two scorers pick
the same dimensions at all. At a median ρ of 0.32–0.38 they largely do not.

The heads are genuinely good at what they were distilled for, and the split is sharp:

| well distilled (sentence level) | ρ | poorly distilled (sentence level) | ρ | bias |
|---|---|---|---|---|
| `score_overall_quality` | 0.869 | `Awe` | 0.052 | +0.58 |
| `score_speech_quality` | 0.744 | `Thankfulness_Gratitude` | 0.083 | +0.99 |
| `Interest` | 0.707 | `Jealousy_&_Envy` | 0.132 | +0.42 |
| `score_background_quality` | 0.694 | `Affection` | 0.145 | +0.50 |

The rare emotions also carry a **large positive bias** — classic regression to the mean on
emotions that are rare in the distillation training data, which would systematically inflate
exactly the emotions a voice-acting caption most wants to be right about. Short segments make
it worse: the heads were distilled on whole clips, and `Jealousy_&_Envy` loses 0.33 ρ,
`Affection` 0.21 and `Contentment` 0.20 when applied to a single sentence — and this repo
scores **every sentence** separately.

So the distilled heads remain a reasonable **fast approximation for the four quality
scores**, and a poor one for emotions. Wiring them in as an optional path is not a matter of
keeping an existing code path alive — there has never been one here — so it is left out
rather than added as a second, weaker scorer with a caveat attached.

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

## The baseline

`baseline_stats.json` describes "the average voice" so deviations are meaningful. Two
baselines ship, and either can be selected — earlier published results stay reproducible:

| file | measured on | default |
|---|---|---|
| **`baseline_stats.json`** | **256,000 DramaBox edge-case clips, in-domain** | ✅ |
| `baseline_stats_emolia.json` | 4,703 Emolia clips + 1,000 character takes (722 clips for the emotions) | — |

```python
base = load_baseline()             # the default, in-domain
base = load_baseline("emolia")     # the previously published baseline
base = load_baseline("/path/to/my_baseline.json")
```

```bash
python caption.py preds.json --baseline emolia    # CLI
export PVC_BASELINE=emolia                        # or process-wide
```

### Why it was swapped — `R_MIXD`

Measured on 256,000 clips, **`R_MIXD` (Mixed Resonance) was the only one of the 57
VoiceNet dimensions whose median sat more than 2 baseline-spreads off the published
baseline**, at `z = −4.64`; the next worst were `Teasing` at +2.25 and `VFLX` at −2.06.

The problem was not the median but the **spread**. The emolia baseline gave `R_MIXD` a
spread of **0.202** where the in-domain measurement finds **0.785** — 3.9× too narrow,
and ~5× narrower than a typical VoiceNet dimension on the same 0–6 scale. Healthy
dimensions match closely across the two corpora (`DFLU` 1.280 vs 0.749, `COGL` 1.081 vs
0.802):

```
z = (1.902 − 2.822) / 0.202 = −4.55        # published spread
z = (1.902 − 2.822) / 0.785 = −0.87        # true spread — never reaches the top-5
```

So a single under-estimated denominator put one dimension at |z| ≈ 7 on real audio and let
it win a top-5 slot on almost every clip. **That is why nearly every caption in this repo
used to begin with *"very flat-resonance"*.** `R_MIXD` is also the weakest-correlating
regression head in the VoiceNet release (held-out `reg_pearson` = **0.392**) — it was both
the mis-scaled dimension and the least trustworthy one.

**Measured effect.** `eval_baseline_swap.py` re-captions the stored raw scores of the 14
demo clips and their 67 sentences (81 captions, no models, no audio) under four
configurations:

| configuration | `R_MIXD` takes a top-k slot | mean \|z\| `R_MIXD` | median \|z\| other 56 dims | mean `r` of selected dims |
|---|---|---|---|---|
| **A** emolia · no floor · no reliability *(previous default)* | **79/81** | 7.44 | 0.83 | 1.000 |
| **B** emolia · floor · reliability | 76/81 | 4.29 | 0.83 | 1.000 |
| **C** dramabox · floor · no reliability | 0/81 | 0.80 | 1.25 | 0.800 |
| **D** dramabox · floor · reliability *(new default)* | **0/81** | 0.80 | 1.25 | 0.837 |

On the 14 global captions alone, A gives `R_MIXD` **13/14** top-5 wins at mean |z| **7.39**
against a median of **0.94** across the other 56 dims — reproducing the figure this README
previously reported as a known issue — and D gives **0/14** at mean |z| 0.87.

Read row **B** honestly: **the spread floor and reliability weighting do not fix `R_MIXD` on
their own.** A 7.4σ artefact survives both. What fixes it is measuring the spread correctly;
the floor and the weighting are safety nets that bound the damage from the *next* such
defect. Independently confirmed on freshly scored audio: on the
[25-clip grid](https://projects.laion.ai/procedural-voice-captions/burst-captions-25/)
`R_MIXD` took a top-5 slot on **20 of 25** clips under the old configuration and **0 of 25**
now, and on the regenerated 100-clip grid it takes **0 of 100**.

One honest consequence of moving in-domain: the genuineness median rises from 1.73 to 2.27,
so clips that used to be described as *"genuine"* are now often *"semi-genuine"*. That is a
different reference population, not a change of opinion about the clips.

### The spread floor

`z = (value − median) / spread` has no defence against a spread that is simply too small,
so the captioner floors it. The floor is **relative**, because the failure was relative:

> **No dimension may have a `spread` narrower than ⅓ of the median `spread` of its own
> group** (`voicenet` / `emonet` / `quality`).

⅓ is taken from the data, not chosen for roundness. Ordering every dimension by
`spread ÷ group-median-spread`:

| baseline / group | lowest ratios | first healthy dim |
|---|---|---|
| emolia · voicenet | `R_MIXD` 0.19, `EXPL` 0.22 | `ROUG` 0.55 |
| emolia · emonet | — | `Awe` 0.48 |
| dramabox · voicenet | `EXPL` 0.35 | `ROUG` 0.57 |
| dramabox · emonet | `Awe` 0.00, `Shame` 0.02, `Pain` 0.03, `Infatuation` 0.05, `Distress` 0.10, `Affection` 0.29, `Helplessness` 0.33 | `Longing` 0.39 |

In all four distributions every well-estimated dimension sits above ~0.35 and every
pathological one far below, so ⅓ separates them cleanly. On the shipped default it floors
**0 of 57** VoiceNet dims, **0 of 2** quality dims, and **9 of 40** EmoNet emotions — the
zero-inflated ones whose raw IQR collapses towards zero (`Awe`'s is exactly `0.0`, which
without a floor is an unbounded z). On the emolia baseline it would have floored exactly the
two anomalies, `R_MIXD` (0.202 → 0.351) and `EXPL`.

`PVC_SPREAD_FLOOR_FRAC=0` disables it; `caption.spread_floors(baseline)` returns the
per-group floors actually in force.

### Reliability weighting — `|z| · r`

A dimension the model cannot predict should not lead the caption. The VoiceNet release
publishes a held-out **`reg_pearson` per dimension** (`metrics_best_per_dim.parquet`), and
the baseline carries it as `reliability_reg_pearson`. It spans **0.392 (`R_MIXD`)** to
**0.949 (`VOLT`)**; the next weakest are `EXPL` 0.415, `ARSH` 0.532, `VFLX` 0.558.

Dimensions are ranked by **`|z| · r`** instead of `|z|`, so a head at `r = 0.39` needs 2.4×
the deviation of one at `r = 0.95` to take the same slot. Deliberate limits:

- **Selection only.** The reported `z`, the direction and the intensity adverb come from the
  raw `|z|`. The deviation is a fact about the clip; `r` is a fact about the predictor, and
  folding one into the other would understate a measured deviation.
- **Unknown reliability ⇒ weight 1.0.** A baseline without the field — including
  `baseline_stats_emolia.json` — behaves exactly as before. This is verified in row A/B of
  the table above: with the emolia baseline, turning weighting on moves 4 of 271 selected
  slots, and those 4 come from the spread floor, not the weighting.
- **EmoNet emotions are not silently assumed perfect.** They have no published
  `reg_pearson`. They are also ranked in their **own** pool (top-3 EmoNet, separate from the
  top-5 VoiceNet), so any *uniform* weight there cannot reorder them — reliability weighting
  is a mathematical **no-op** on the EmoNet group today. It is still applied, so that
  published per-emotion reliabilities would be picked up with no code change. Genuineness and
  blend are never ranked (always-on / thresholded) and are unaffected.

Holding the baseline fixed, the weighting changes **56 of 271** selected slots (20.7 %) and
lifts the mean reliability of the dimensions that make it into a caption from **0.800 to
0.837**.

```bash
python caption.py preds.json --no-reliability        # rank by raw |z|
python caption.py preds.json --reliability-min 0.5   # also drop dims below r = 0.5
export PVC_RELIABILITY=0                             # process-wide off
```

`caption_detail()` returns `reliability` and `rank_score` per dimension, so a selection is
always auditable.

### How each baseline was measured

**The default — DramaBox, in-domain.** 256,000 clips sampled from the DramaBox edge-case
corpus (reward-Top-3 annotations), capped at 2,000 per shard. `spread` is **IQR / 1.349**, a
robust σ that, unlike `std`, is not dragged by these heads' long tails. Every dimension is
measured on the same embedding path used at inference. One emotion, `Jealousy & Envy`, is
carried over from the emolia baseline — the scoring run that produced the corpus was missing
that head — and says so in its `source` field. `import_baseline.py` normalises such a
measurement into this repo's schema (it reconciles Empathic-Insight file-stem keys such as
`Hope_Enthusiasm_Optimism` with the taxonomy names, carries the synonym clusters over, and
reports every rename).

**The legacy baseline — Emolia + character voices**, built by `compute_baseline.py`:

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

When a baseline supplies no precomputed `spread`, the captioner falls back to
`1.4826 · MAD`, and to `std` when `1.4826·MAD < 0.5·std`. That fallback matters for the
EmoNet emotions, which are strongly zero-inflated (most clips ≈ 0, so the raw MAD collapses).
Whatever the estimator, the [floor](#the-spread-floor) is applied on top.

**Per-dimension statistics** in the shipped default (mean, median, std, MAD, p10, p90, n,
`spread`, and `reliability_reg_pearson` for the VoiceNet dims):

| code | name | group | mean | median | std | spread | `r` | n |
|------|------|-------|------|--------|-----|--------|-----|---|
| AGEV | Voice Age | voicenet | 3.09 | 3.10 | 0.78 | 0.78 | 0.894 | 256,000 |
| GEND | Perceived Gender | voicenet | 4.04 | 4.84 | 1.60 | 1.87 | 0.887 | 256,000 |
| REGS | Register | voicenet | 1.54 | 1.22 | 1.04 | 1.00 | 0.884 | 256,000 |
| TEMP | Tempo | voicenet | 2.31 | 2.27 | 0.86 | 0.91 | 0.910 | 256,000 |
| AROU | Arousal | voicenet | 2.54 | 2.51 | 0.97 | 1.06 | 0.929 | 256,000 |
| VALN | Valence | voicenet | 2.49 | 2.57 | 0.81 | 0.70 | 0.785 | 256,000 |
| R_MIXD | Mixed Resonance | voicenet | 1.93 | 1.88 | 0.74 | **0.79** | **0.392** | 256,000 |
| genuineness | Genuineness | quality | 2.45 | 2.27 | 0.96 | 0.98 | — | 256,000 |
| blend | Vocal-burst blend | quality | 3.61 | 3.38 | 2.19 | 2.40 | — | 256,000 |
| Anger | Anger | emonet | 0.34 | 0.04 | 0.45 | 0.48 | — | 256,000 |
| Amusement | Amusement | emonet | 0.80 | 0.83 | 0.64 | 0.93 | — | 256,000 |
| Interest | Interest | emonet | 2.28 | 2.35 | 0.40 | 0.28 | — | 256,000 |

### Rebuild them

```bash
# the legacy emolia baseline, from scratch
export HF_HOME=/path/to/hf_cache HF_TOKEN=...     # needs access to the gated repos above
python compute_baseline.py --stage emolia_vn      # CPU: heads on precomputed embeddings
python compute_baseline.py --stage bestof64       # CPU: read precomputed scores
python compute_baseline.py --stage emonet --gpu 0 # 1 GPU, small batch: BUD-E-Whisper
python compute_baseline.py --stage merge          # write baseline_stats.json

# normalise a baseline measured on some other corpus into this repo's schema
python import_baseline.py measured.json --out baseline_stats.json \
       --reference baseline_stats_emolia.json

# what a swap does to real captions, from stored scores — no models, no audio
python eval_baseline_swap.py --sentences
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

- **The default baseline is in-domain for one domain.** It is measured on 256,000 DramaBox
  edge-case clips — expressive, acted, mostly dramatic speech. That is a far better match for
  voice-acting captions than the previous 4,703-clip Emolia baseline, and it is what fixed
  `R_MIXD`, but it is *not* a neutral cross-domain reference. Captioning calm read speech or
  broadcast audio against it will report larger deviations than a same-domain baseline would.
  `load_baseline("emolia")` and `import_baseline.py` exist for that reason.
- **`R_MIXD` is still the weakest head in the VoiceNet release** (`reg_pearson` 0.392). It no
  longer dominates captions — correct spread plus `|z|·r` ranking — but when it *does* reach
  the top-5 the underlying prediction is still the least trustworthy of the 57. Setting
  `--reliability-min 0.5` excludes it (and `EXPL`, 0.415) outright.
- **One emotion's baseline is not in-domain.** `Jealousy & Envy` is carried over from the
  Emolia baseline because the head was missing from the scoring run that produced the
  DramaBox corpus. Its `source` field says so, and its z-scores are not comparable with the
  other 39. It is the only such dimension.
- **Duplicate tags in the terse form.** Several dimensions share a tag word (`ROUG` low and
  `DFLU` low are both *smooth*; `S_FORM` high and `S_CASU` low are both *formal*), so a
  `tags` caption can read "very formal, very formal". Cosmetic, pre-existing, and more
  visible now that `R_MIXD` no longer occupies a slot on every clip.
- The demo page's long-form example is a **concatenation of shorter clips**, not a natural
  long recording — nothing longer than ~18 s ships in this repo. It exercises the 30 s
  windowing honestly, but it is not a substitute for evaluation on real long-form audio.

---

## Files

```
caption.py                  # caption() / caption_detail() + 11 templates + gates  (stdlib only)
baseline_stats.json         # DEFAULT baseline — 256k DramaBox clips, in-domain (99 dims + _meta)
baseline_stats_emolia.json  # the previously published baseline, kept for reproducibility
burst_captions.py           # full pipeline: score -> timestamps -> locate -> classify -> insert
asr_words.py                # bundled Parakeet token -> word -> sentence helpers (stdlib only)
augment.py                  # score once, caption many times (train-time text augmentation)
compute_baseline.py         # rebuild the emolia baseline (staged)
import_baseline.py          # normalise an out-of-repo baseline measurement into this schema
eval_baseline_swap.py       # measure what a baseline / weighting change does to real captions
build_demo25.py             # renders docs/burst-captions-25/ (25 clips, full stack)
build_caption_grid.py       # rebuilds docs/captions.json + the DATA block in docs/index.html
build_burst_demo_v2.py      # renders docs/burst-captions-v2/ (--recaption = no models needed)
build_burst_demo.py         # renders the older docs/burst-captions/ (needs out-of-repo wavs)
vocalburst_taxonomy.json    # 82 vocal-burst classes (class order for the classifier)
burst_x2.py                 # the 17-class second-opinion head: recall tiers + merge policy
vocalburst_x2_recall.json   # per-class recall floor, laion/voiceclap-large-v2 head
vocalburst_x2_recall_commercial.json  # ... and the voiceclap-commercial head
test_burst_x2.py            # tiers, vocabulary relation, merge policies (stdlib, no models)
test_burst_pipeline.py      # burst insertion, pause markers, word-end trimming
examples/                   # real complete predictions (dims + emo + genu + blend)
docs/                       # the published GitHub Pages demos
requirements.txt
```

---

## Demo pages

Published at [`projects.laion.ai/procedural-voice-captions/`](https://projects.laion.ai/procedural-voice-captions/)
and [`laion-ai.github.io/procedural-voice-captions/`](https://laion-ai.github.io/procedural-voice-captions/).
Each has real audio players — GitHub READMEs cannot embed audio, so these are links, not embeds.

| page | what |
|---|---|
| [**burst-captions-25**](https://projects.laion.ai/procedural-voice-captions/burst-captions-25/) | **Current stack, end to end.** 25 multilingual clips: locator `model_v2.pt` → `vocal-burst-detector-v2` → Empathic-Insight scoring → captions on the in-domain baseline with reliability weighting. Each card lists every selected dimension's `z`, reliability `r` and effective rank `|z|·r`, and expands to show what the *same* scores produced under the previous captioner. |
| [burst-captions-v2](https://projects.laion.ai/procedural-voice-captions/burst-captions-v2/) | 14 clips with locator `model_v2.pt` + `vocal-burst-detector-v2`, including a 91 s clip that needs 4 locator windows, and the locator-v1 output for comparison. Captions regenerated on the current baseline. |
| [caption grid](https://projects.laion.ai/procedural-voice-captions/) | 100 real multilingual clips, captioned, with the 11 templates assigned round-robin. Rescored and re-captioned on the current baseline. |
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

- **A second, narrower burst classifier — added, not swapped in.**
  `laion/vocal-burst-detector-x2` names a span over 17 classes (16 bursts + `no_burst`) on the
  3584-d `laion/voiceclap-large-v2` encoder, and beats the shipped head on every measured cell
  (val 0.7056, real 0.7536, family 0.8480 vs chance 0.0588); `BURST_X2_HEAD=commercial` selects a
  768-d alternative that reuses the encoder this repo already loads. But **all 16 of its classes
  are already in the 82-class taxonomy** — it removes 66 classes and seven whole taxonomy groups,
  and 83.8 % of the 12,894 burst events in the annotated LAION-TTS corpus carry a label it cannot
  emit (`Low Mumble` 26.3 %, `Ahem` 26.2 %, `Contented Sigh` 20.5 %, `Surprised Gasp` 7.2 %). So
  `BURST_LABEL_SOURCE` stays at `v2` and captions are unchanged by default; `union` gives the new
  head the spans it has jurisdiction over and leaves the rest to the old vocabulary. Each class is
  written only as strongly as its measured per-class recall allows — `(Chuckle)`,
  `(Breathy Giggle?)`, `(Breath)`, `(Vocal Burst)` — and every burst record says which vocabulary
  produced it.
- **`[pause X.Xs]` was missing a quarter of the pauses, and shortening the rest.** A Parakeet-TDT
  token's duration runs to the *next* token, so word spans are contiguous (65.5 % of word pairs on
  96 real clips report a gap of exactly 0.000 s) and each pause is billed to the word before it.
  Against the audio, 313 pauses ≥ 0.30 s are audible where the code printed 233 — all real, but 80
  missing, and those printed understated by a mean 0.134 s (max 1.22 s). `trim_word_ends()` now
  moves each word's end back to its last non-silent frame and keeps the raw span as `end_span`:
  233 → 291 markers, gap-exactly-zero 65.5 % → 1.9 %. A blind duration cap was measured and
  rejected — at 0.30 s it invents 154 pauses out of 431, because 681 of the 910 spans longer than
  0.30 s are genuinely long words. `BURST_WORD_END=raw` restores the old behaviour.
- **The repo has tests.** `python -m unittest discover` — 60 of them, stdlib only, no models and
  no network, covering the vocabulary relation, the recall tiers, the merge policies, burst
  insertion, pause markers and word-end trimming.

- **The default baseline is now measured in-domain**, on 256,000 DramaBox edge-case clips
  (`baseline_stats.json`); the previously published one is kept as
  `baseline_stats_emolia.json` and is selectable by name, path or `$PVC_BASELINE`. This
  fixes the `R_MIXD` scale defect that made almost every caption open with *"very
  flat-resonance"* — top-5 wins go from 79/81 to 0/81 on the stored demo captions, 20/25 to
  0/25 on freshly scored audio, and 0/100 on the regenerated caption grid.
- **Reliability weighting, on by default.** Dimensions are ranked by `|z| · reg_pearson`
  instead of `|z|`, so the weakest heads (`R_MIXD` 0.392, `EXPL` 0.415) no longer lead a
  caption on deviation alone. Selection only — the reported z and the intensity wording are
  unchanged. EmoNet emotions have no published reliability and are ranked in their own pool,
  where a uniform weight is a no-op; this is documented rather than assumed.
  `PVC_RELIABILITY=0` / `--no-reliability` restores the old ranking.
- **Spread floor.** No dimension's `spread` may be narrower than ⅓ of its group's median
  spread. On the default baseline this floors 0 of 57 VoiceNet dims and 9 of 40 zero-inflated
  emotions (`Awe`'s raw spread is exactly 0.0). `PVC_SPREAD_FLOOR_FRAC=0` disables it.
- **Empathic-Insight-Voice-Plus is the only emotion scorer**, and the README now cites the
  72,809-matched-pair comparison against the distilled VoiceCLAP attribute heads (median
  Spearman ρ 0.381 utterance / 0.315 sentence) that settles it. No distilled path is shipped.
- **New 25-clip demo grid** (`docs/burst-captions-25/`, `build_demo25.py`) showing the whole
  current stack with per-dimension `z` / `r` / `|z|·r` and a per-clip before/after. It reuses
  the mp3s already in `docs/audio/`, so it adds no audio bytes. The 100-clip grid and the
  14-clip page were regenerated; `build_burst_demo_v2.py --recaption` rebuilds the latter
  from stored scores with no GPU and no audio.
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

## Timed prompt scheme (prompting guide §3) — default ON

`BC_TIMED=1` (default) makes `process()` additionally emit **`script_timed`** and
`procedural_caption()` prefer it, matching the prompting guide:

```
(very smooth, very formal) [1.52 seconds duration] Es riecht nach Sonnencreme.
[0.66 seconds pause]
(very brisk, curious) [2.08 seconds duration] (Chuckle, 0.34 seconds) Why what's that?
```

* `[N.NN seconds duration]` — next segment's spoken time, measured **first word onset →
  last word offset**; sub-threshold gaps fold into the neighbouring sentence so printed
  numbers add up to the clip length; a bounded ≤ 0.30 s rounding residue is absorbed into
  the last numeric tag (`BC_TIMED_ABSORB`), anything larger is a loud `ValueError`.
* `[N.NN seconds pause]` — every silence ≥ `BURST_PAUSE_THR` (default 0.20 s), including
  before the first and after the last word.
* Bursts print `(Label, N.NN seconds)` at the nearest word gap; when speech overlap
  leaves < 50 ms free they collapse to a bare `(Label)`. Direction cues stay digit-free.
* `verify_timed_script(script, transcript, dur)` re-parses independently (length sum,
  transcript identity) — `process()` withholds the timed form (`timed_ok: false`) rather
  than emitting a broken prompt. `BC_TIMED=0` restores the legacy untimed lines exactly.

Tests: `python3 test_timed.py` (4 cases, no GPU). The 20k/50k/100k ladder packs still emit
the untimed arm via their own template; the timed repair of that packed data lives in
`m2_multitask/repair_prompts.py` (timed-v2) and predates this module.
