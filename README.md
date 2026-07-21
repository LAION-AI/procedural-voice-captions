# Procedural Voice Captions

Turn a speech clip's **model predictions** into a compact, human-readable
**caption** that describes the voice by how far it deviates from the *average
voice*.

> **🔊 [Live demo grid →](https://projects.laion.ai/procedural-voice-captions/)**
> 100 real multilingual speech clips, each with an audio player and the full
> caption written automatically by this module (also at
> [laion-ai.github.io/procedural-voice-captions](https://laion-ai.github.io/procedural-voice-captions/)).
>
> **🧠 [MOSS-Audio-Thinking experiment →](https://projects.laion.ai/procedural-voice-captions/moss-thinking/)**
> The same 100 clips passed through the **MOSS-Audio-Thinking** reasoning models
> (4B &amp; 8B): they *listen to the audio* and fuse the procedural caption + a
> Parakeet transcript with timestamps + the LAION taxonomies (VoiceNet / EmoNet /
> VocalBurst) into the final voice-acting format — a **GENERAL** "how it sounds"
> instruction plus a **SCRIPT** with a per-sentence `(delivery cue)`, inline vocal
> `(bursts)` the model actually hears, and `[pause X.Xs]` markers from the timestamps.
>
> **💥 [Vocal bursts inserted into captions →](https://projects.laion.ai/procedural-voice-captions/burst-captions/)**
> Detected **vocal bursts** (laughs, gasps, sighs, screams, sobs, grunts…) written
> straight into the captions on real LAION character voices + in-the-wild clips.
> Two selectable insertion variants shown side by side — **A: locator** (precise
> inline `(burst)` at the detected time) vs **B: sentence-level** (one burst woven
> into a sentence's caption). See [Vocal bursts in captions](#vocal-bursts-in-captions)
> below. Code: [`burst_captions.py`](burst_captions.py).

This repo ships:

- **`baseline_stats.json`** — baseline distribution statistics (mean / median /
  std / MAD / spread / p10 / p90 / n) for **99 dimensions** (57 VoiceNet
  dimensions + genuineness + vocal-burst blend + 40 EmoNet emotions), estimated
  over a broad, multilingual sample of speech.
- **`caption.py`** — a self-contained, configurable procedural caption module
  (stdlib only at runtime) + CLI.
- **`compute_baseline.py`** — the script that rebuilds `baseline_stats.json`
  from scratch.
- **`examples/`** — real, complete predictions for a few clips.

It is the offline captioning stage used by the LAION voice-acting stack: an
upstream ASR/segmentation step cuts sentences and tags non-verbal events, the
four scoring models below predict per-clip attributes, and this module converts
those numbers into words.

---

## What is a "caption" here?

Given a clip's predictions, the module:

1. **z-scores** every dimension against the baseline:
   `z = (value − median) / spread`.
2. selects the **top-k VoiceNet dimensions by |z|** (default `k=5`) and the
   **top-k EmoNet emotions by |z|** (default `k=3`) — both configurable.
3. **always includes** Age (`AGEV`), Gender (`GEND`), Register (`REGS`) and
   Tempo (`TEMP`), even when they sit at the baseline.
   - **Exception — categorical dims use absolute 0–6 bands, not z-scores.**
     `AGEV`, `GEND` and `REGS` are near-categorical on the 0–6 VoiceNet scale, so
     deviation-from-baseline is the wrong lens: the population median for `GEND`
     leans masculine (~3.7), so a ±0.5·spread "neutral" band swallowed most male
     voices and mislabeled them *gender-neutral*. These three dims are therefore
     mapped through fixed thresholds centered on the scale midpoint `3.0` (see
     `ABSOLUTE_BANDS` in `caption.py`) — e.g. `GEND ≥ 3.5` → masculine,
     `≤ 2.5` → feminine, in-between → androgynous. `TEMP` and all other
     dimensions still use z-scores.
4. rotates through each emotion's **synonym cluster** so captions aren't one
   rigid word per emotion.
5. applies the **genuineness gate** (see below).
6. renders each dimension with a **direction** (above / below baseline) and an
   **intensity** from `|z|` (Extremely / Very / Notably / Somewhat; below `0.5`
   → "about average").

```python
from caption import caption, caption_detail, load_baseline

base  = load_baseline()                 # bundled baseline_stats.json
preds = { ... }                         # see "Prediction format" below
print(caption(preds, base, k_voicenet=5, k_emonet=3))
# A voice that is extremely forward in mask resonance; very full and cinematic;
# ...; notably high in vocal register; moderate in tempo; extremely carrying
# playfulness; extremely carrying distrust; very carrying detestation; deeply
# and genuinely felt.

detail = caption_detail(preds, base)    # structured: per-dim {dim, z, direction, phrase, ...}
```

CLI:

```bash
python caption.py examples/worker_0_EN_tDPU-wXSB5y_W000085.json --kv 5 --ke 3
python caption.py - --json < preds.json           # read stdin, print structured detail
```

### Prediction format

`caption()` accepts either layout:

- **nested** — `{"dims": {DIM: value, ...}, "emo": {Emotion: value, ...},
  "genu": x, "blend": y}`
- **flat** — `{DIM: value, ..., Emotion: value, ..., "genuineness": x, "blend": y}`

Values may be plain numbers or `{"value"/"reg_score"/"reg": x}`. VoiceNet dims
and genuineness are on a 0–6 scale, blend on 0–10, EmoNet emotions are raw head
outputs (roughly 0–3, mostly ≈0).

```bash
python caption.py preds.json --template identity_first     # pick a surface form
python caption.py preds.json --template random --shuffle   # seed-random form + shuffled order
```

---

## Caption templates & dimension shuffling

A single fixed sentence shape (`"A voice that is …; …; …."`) means **every** caption
looks the same. When these captions are used as **fine-tuning targets** for a
voice-acting model, that rigid, repetitive surface form is an easy thing for the
model to overfit to — it learns the template, not the voice. To avoid that, the
same underlying phrase content can be rendered through **11 interchangeable
surface templates** — including a terse **`tags`** form — plus optional **dimension
shuffling**.

Crucially, **only the arrangement changes.** Every template passes through the
exact same pipeline — z-scores, top-k selection, always-on Age/Gender/Register/
Tempo, the absolute-band categorical dims, the genuineness gate, and EmoNet
synonym rotation. The phrase *content* is identical; templates only reorder the
four phrase groups (identity = Age/Gender/Register/Tempo · timbre = the other
VoiceNet dims · emotion = EmoNet · quality = genuineness/vocal-burst) and vary the
connective style. So captions stay information-preserving and comparable.

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
| **`tags`**          | *very warm, very breathy, very meek, elderly, masculine, low-register, slightly slow, very lust, very heartache, genuine, with-bursts* (terse tags, **no filler** — intensity + dimension as short words; the **default** for the burst pipeline) |

The **`tags`** template is special: instead of the verbose phrases it emits one short tag per salient
dimension (`very warm`, `breathy`, `masculine`, `elderly`, `low-register`, …), drops about-average
dims entirely, and comma-joins them. It is the recommended, on-the-point surface form and is the
**default** for the vocal-burst caption pipeline (`PROC_TEMPLATE=tags`); set `PROC_TEMPLATE` to any
prose template if you prefer full sentences.

API:

```python
from caption import caption, caption_detail, TEMPLATE_NAMES

caption(preds, base, template="identity_first")     # one of the 10 names
caption(preds, base, template="random",             # deterministic pick from synonym_seed
        synonym_seed=1234)
caption(preds, base, template="random",             # + permute non-identity dims / emotions
        synonym_seed=1234, shuffle_dims=True)        # (order only; selection unchanged)

caption_detail(preds, base, template="random", synonym_seed=1234)["template"]  # -> chosen name
```

- **`template`** — one of `TEMPLATE_NAMES` (the 11 above), or `"random"` to pick one
  deterministically from the clip's `synonym_seed`. `caption()` defaults to `"default"`,
  so existing callers are **byte-for-byte unchanged**; the burst pipeline defaults to
  `"tags"` via `PROC_TEMPLATE`.
- **`shuffle_dims`** — deterministically permutes the order of the non-identity
  timbre dims and the emotions within their groups (seeded by `synonym_seed`). It
  changes only *display order*, never *which* dims/emotions were selected. The
  always-on identity dims keep their slots.
- **`caption_detail(...)`** reports the resolved template under the `"template"`
  key (useful for showing a badge in a UI).

Everything is deterministic: given the same `synonym_seed`, a clip always renders
the same template, shuffle and wording. The [**live demo grid →**](https://projects.laion.ai/procedural-voice-captions/)
assigns the 10 templates round-robin across its 100 clips (with shuffling on) and
shows a few clips rendered under several templates side by side.

---

## The genuineness gate

Emotion wording is only as strong as the delivery is believable. Let `zg` be the
genuineness z-score:

| condition                          | gate      | effect on emotions                       | genuineness descriptor                     |
|------------------------------------|-----------|------------------------------------------|--------------------------------------------|
| `value ≥ genuineness median`       | `open`    | full intensity allowed (up to Extremely) | "deeply and genuinely felt" / "genuine …"  |
| below median but `zg > −1`         | `capped`  | intensity capped at **Notably**          | "only slightly genuine, somewhat performed"|
| `zg ≤ −1` (well below)             | `dropped` | emotion phrases **dropped entirely**     | "measured and performed rather than genuine" |

This prevents an over-acted, low-genuineness clip from being captioned as
"extremely enraged" when the anger is performed rather than real.

---

## The gender gate (Empathic-Insight)

The VoiceNet `GEND` dimension always emits *some* gender phrase, which is risky on voices whose
gender is not clearly perceptible (soft whispers, ASMR, children, stylized characters). We therefore
add a second, dedicated gender signal — the **Empathic-Insight Gender expert** (`model_Gender_best.pth`,
bipolar **−2 = very masculine … +2 = very feminine**, 0 = neutral) — and use it purely as a
**confidence gate**:

- **|EI gender| ≥ `EI_GENDER_GATE`** (default 0.5) → the voice's gender is confidently perceptible →
  keep the well-calibrated **VoiceNet `GEND`** label.
- **|EI gender| < gate** (near zero) → **omit the gender clause entirely** rather than assert a
  possibly-wrong one.

Why gate-only (not label): a small A/B study on ASMR clips ([**live grid →**](https://projects.laion.ai/procedural-voice-captions/gender-ab/),
Gemini-3.5-Flash as ground truth) found the EI expert tracks perceived gender about as well as
VoiceNet in *direction* (sign-agree 30–31 / 32) but **compresses clearly-masculine voices toward
neutral** (raw ≈ −0.2 where Gemini says *very masculine*). So EI is trustworthy at the extremes it was
trained to separate — ideal as a gate — while VoiceNet gives the better graded label above the gate.
It also confirmed the "soft male ASMR sounds feminine" captions are *correct*: Gemini hears those
generations as feminine too. Tunable via `EI_GENDER_GATE` (env); set to `0` to disable.

---

## On-the-fly caption augmentation — *score once, caption many times*

Because a caption is a pure function of the stored **scores** + a template + a seed, you don't have to
freeze one caption per clip. Run the four scoring models **once**, store only the raw scores (per
sentence and global) with the transcript and confirmed bursts, then regenerate a **different**
procedural caption every epoch — different template, synonym rotation, dim/emotion shuffle, tags vs.
prose — with no audio and no model calls at train time (microseconds of string formatting).

This is a cheap, strong **text augmentation** for TTS / caption models: the target *wording and
structure* vary across epochs while the *content* (which dims/emotions are salient, the bursts, the
words) stays fixed, so the model learns the score→text mapping instead of memorizing one phrasing.

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

`process()` already attaches the raw scores it computes: `result["scores"]` (global) and
`result["sentences"][i]["scores"]` (per sentence) hold `{dims, emo, genu, blend}`, so `score_record()`
is just a distillation — no extra inference. The gender gate is applied automatically from the stored
`ei_gender`. See [`augment.py`](augment.py) (`python augment.py` prints four varied captions from one
record).

---

## The four scoring models

All four run on top of the **VoiceCLAP-commercial** 768-d embedding (except
EmoNet, which uses BUD-E-Whisper):

| group      | what                              | model |
|------------|-----------------------------------|-------|
| VoiceNet   | 57 voice dimensions (0–6), incl. Valence `VALN` & Arousal `AROU` | [`laion/voicenet-dimension-predictors-commercial`](https://huggingface.co/laion/voicenet-dimension-predictors-commercial) |
| quality    | genuineness (0–6)                 | [`laion/voiceclap-commercial-genuineness`](https://huggingface.co/laion/voiceclap-commercial-genuineness) |
| quality    | vocal-burst blend (0–10)          | [`laion/voiceclap-commercial-vocalburst-blend`](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend) |
| EmoNet     | 40 emotions                       | [`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) (BUD-E-Whisper encoder + per-emotion heads) |
| embedder   | 768-d speech embedding            | [`laion/voiceclap-commercial`](https://huggingface.co/laion/voiceclap-commercial) |

**Valence / Arousal rule:** Valence and Arousal are taken from VoiceNet's
`VALN` / `AROU` heads. EmoNet is used for its **40 emotions only** — EmoNet's own
valence/arousal axes are never loaded and never double-counted.

The dimension names, level rubrics and emotion **synonym clusters** come from
[**LAION-AI/voice-taxonomies**](https://github.com/LAION-AI/voice-taxonomies)
(VoiceNet dimension taxonomy, vocalburst taxonomy, EmoNet emotion taxonomy).

---

## How the baseline was built

`baseline_stats.json` describes "the average voice" so deviations are
meaningful. It was built (`compute_baseline.py`) from two sources:

**1. Emolia — ~1000 random clips per language** (`en, de, zh, fr, ko, ja`):
- VoiceNet / genuineness / blend are computed by running the heads directly on
  **precomputed VoiceCLAP-commercial embeddings** (no audio needed).
  Per-language n: **en 1000, de 1000, zh 1000, fr 687, ko 562, ja 454**
  (fewer where the embedding set had fewer clips) → **4703** clips.
- The 40 EmoNet emotions require audio, so they are computed from a **bounded
  random subset**: ~80 random clips per shard across a few shards per language,
  streaming each shard tar and deleting it after use → **722** clips
  (en 240 / de 160 / zh 82 / fr 80 / ko 80 / ja 80). This reduced-n choice is
  recorded in the JSON `_meta`.

**2. ~1000 random takes from
[`laion/moss-character-voices-bestof64`](https://huggingface.co/datasets/laion/moss-character-voices-bestof64)**
— extreme, best-of-64 character voices (dragon, fairy, goblin, ASMR, …). Their
precomputed `dims_json` / `genu` / `blend` are folded into the VoiceNet / quality
baselines to **deliberately widen the spread** (these are intentionally extreme
and only cover the dimensions each character exercises). Their per-character
`emo_json` is **not** used for the 40-emotion baseline (Emolia audio is).

**Per-dimension statistics.** For each dimension we store mean, median, std,
MAD, p10, p90, n, and a robust **`spread`** for z-scoring:

> `spread = 1.4826 · MAD`, falling back to `std` when `1.4826·MAD < 0.5·std`.

The MAD fallback matters for the EmoNet emotions, which are strongly
zero-inflated (most clips ≈ 0, so the raw MAD collapses and would blow up
z-scores). Captions use `z = (value − median) / spread`.

### Example baseline values

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
export HF_HOME=/path/to/hf_cache HF_TOKEN=...           # needs access to the gated repos above
python compute_baseline.py --stage emolia_vn            # CPU: heads on precomputed embeddings
python compute_baseline.py --stage bestof64             # CPU: read precomputed scores
python compute_baseline.py --stage emonet  --gpu 0      # 1 GPU, small batch: BUD-E-Whisper
python compute_baseline.py --stage merge                # write baseline_stats.json
```

The paths to the Emolia index, embeddings and heads are constants at the top of
`compute_baseline.py`; adjust them for your environment.

---

## Upstream: sentence cutting + non-verbal tags (recommended models)

The captioner consumes per-clip predictions; producing well-segmented clips with
timestamps and non-verbal event tags is an upstream step. Recommended models:

- **ASR + timestamps:** [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
  [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR),
  [`mrfakename/vibevoice-asr-soundscapes-events-promptfix-vb4x-12k-fft-20260603`](https://huggingface.co/mrfakename/vibevoice-asr-soundscapes-events-promptfix-vb4x-12k-fft-20260603),
  [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- **Non-verbal / soundscape tags:** [`fluxions-ai/whisperdrz`](https://huggingface.co/fluxions-ai/whisperdrz)
- **Taxonomy source of truth:** [`LAION-AI/voice-taxonomies`](https://github.com/LAION-AI/voice-taxonomies)

The intended fine-tuning prompt format that consumes these captions is described
here:
[MOSS-Local 1.5 Voice-Acting — fine-tuning prompt format](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/finetuning_prompt_format.html).

---

## Vocal bursts in captions

`caption.py` describes the *voice*; it does not place the non-verbal **vocal
bursts** (laughs, gasps, sighs, screams, sobs, grunts, slaps…) that a clip
actually contains. [`burst_captions.py`](burst_captions.py) adds that: it runs
the full scoring stack on raw audio, **detects and names the bursts**, and writes
them into the captions. **[Live demo →](https://projects.laion.ai/procedural-voice-captions/burst-captions/)**

**Pipeline (all env-driven, everything is a model prediction):**

1. **Score the audio** — VoiceCLAP-commercial embedding → the 57 VoiceNet dim
   heads + genuineness + vocal-burst-blend; EmoNet-40
   (`laion/BUD-E-Whisper` → `laion/Empathic-Insight-Voice-Plus`; set
   `BC_EMONET=0` to skip and fall back to VoiceNet + genuineness only);
   **Parakeet-TDT** for word + sentence timestamps.
2. **Detect bursts** — the `laion/vocalburst-locator` (a 50 fps per-frame
   burst-probability model) is scanned over the clip; a VoiceCLAP→MLP multi-label
   classifier (83 outputs = 82 taxonomy classes + `no_burst`, sigmoid, **top-1**)
   names each candidate. The `no_burst` gate (`P(no_burst) ≥ 0.5` ⇒ nothing) kills
   false alarms.

**Caption composition (as on the demo page):**

- **Global caption** = Top-5 VoiceNet dims + Top-3 EmoNet emotions + genuineness +
  Age + Gender + Tempo. Bursts are **not** placed at global level.
- **Per-sentence caption** = Top-3 VoiceNet dims + Top-3 emotions (no
  Age/Gender/Tempo — those are global only).

**Two burst-insertion variants (selectable, never both at once):**

- **Variant A — locator (precise position).** Locator over the whole clip at
  threshold **0.7**; each detected span → classifier → if `P(no_burst) < 0.5` the
  **top-1** class is inserted as its own `(Class Name)` **inline at the burst's
  time**, between the two ASR words nearest that moment. Reads like a script:
  *“(Surprised Gasp) No, no, (Surprised Gasp) oh, it hurts.”*
- **Variant B — sentence-level.** Classifier on each whole sentence segment; if
  `P(no_burst) ≥ 0.5` attach nothing, else the top-1 class is woven into that
  sentence's caption via a small procedural phrase (*“… punctuated by a (Gasp).”*).

**Recommendation — prefer Variant A (locator), keep B as a fallback.** Variant A
anchors each burst to *when it happens*, which is exactly the per-event,
time-aligned signal the voice-acting SCRIPT format wants, and its per-span
`P(no_burst) < 0.5` gate suppresses false positives event-by-event. Variant B is
coarser: it can only say a sentence *contains* a burst, collapses multiple bursts
per sentence into one label, and running the classifier over a whole sentence
dilutes a short burst among seconds of speech (lower recall on brief events). Use
Variant B when word-level timestamps are unavailable (non-speech screams, failed
ASR) or when a compact one-label-per-sentence summary is enough — there Variant A
has nowhere to anchor and B still adds value.

```bash
# score arbitrary audio -> per-clip JSON with both variants
export HF_HOME=/tmp/hf_cache HF_TOKEN=...            # for the gated LAION models
python burst_captions.py 'my_clips/*.wav' --out out.json --mp3-dir mp3/
python build_burst_demo.py                          # rebuild the demo page
```

The model locations (VoiceNet head repo, genu/blend heads, burst-locator,
classifier `.pt`) are all overridable via env vars documented at the top of
`burst_captions.py`; the burst taxonomy is bundled (`vocalburst_taxonomy.json`).

---


### Evaluation & recommended default

A Gemini-3.5-Flash study (see the [config study](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline/blob/main/CAPTION_CONFIG_EVAL.md)) over caption configurations found:

- **Sentence-level burst insertion (Variant B) is the recommended default** — it beats locator-inline (Variant A), 8.50 vs 7.63, because it keeps the transcript clean (better ASR readability + burst score). The kept bursts are placed **inline** into the per-sentence script (mapped to the sentence they occur in by word timestamps).
- The **detector→confirm** stage (`vocalburst-locator` → confirm with the multi-label classifier, `P(no_burst)≥0.5` → discard, else top-1) is the base for both variants and gates hallucinations.
- **Locator threshold `0.75` (grid-searched, default).** A sweep over the character voices (Gemini-3.5-Flash rating burst precision vs recall) settled `BURST_LOCATOR_THR`:

  | thr | precision | recall | kept | hallucinated |
  |-----|-----------|--------|------|--------------|
  | 0.70 | 8.09 | 8.84 | 42 | 18 |
  | **0.75 (default)** | **8.66** | **8.78** | 38 | 12 |
  | 0.80 | 8.59 | 8.75 | 37 | 9 |
  | 0.85 | 8.72 | 8.38 | 33 | 6 |
  | 0.90 | 8.75 | 8.50 | 28 | 7 |

  `0.70` is clearly too loose (18 hallucinations). `0.75` is the precision/recall optimum — it nearly halves hallucinations with almost no recall loss. Bump to **`0.80`** if you want still fewer hallucinations at a small recall cost.
- **Duration gate** — locator spans below a minimum duration are rejected (default 0.30 s global, 0.60 s for the smack/click/slap groups). Short spans dominate transient false positives (Slap Face / Lip Smack firing at high `p` on 0.10–0.16 s spans); every discarded span in the character-voice study was < 0.6 s. Tunable via `BURST_MIN_DUR` / `BURST_TRANSIENT_MIN_DUR`.
- **Gender gate** — the Empathic-Insight Gender expert gates whether a gender phrase is emitted at all (see above); near-zero → gender omitted.
- **Surface form** — the terse **`tags`** template is the default (`PROC_TEMPLATE`).
- Burst classifiers are the **multilingual-retrained v2** ([single](https://huggingface.co/laion/vocalburst-classifier-single) mAP 0.70/0.87, [multi-label](https://huggingface.co/laion/vocalburst-classifier-multilabel)) — the multi-label model has fewer false positives at the confirm stage.

### Procedural vs. LLM-assisted — throughput & quality (1×A100, n=32, Gemini-3.5-Flash 1–10)

There are two ways to turn the scores into a final caption. This repo produces the **procedural** one
directly; the [**Comprehensive Voice-Acting Annotation Pipeline**](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline)
feeds the same annotations to a small LLM (Gemma-4-E4B, text-only).

| path | s/clip | clips/s | overall | emotion | burst&nbsp;acc | gender | wins |
|------|--------|---------|---------|---------|----------------|--------|------|
| **procedural** (this repo) | **0.62** | **1.62** | 6.95 | 7.56 | 6.83 | 9.47 | 12 |
| **LLM-assisted** ([VAP repo](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline)) | 1.64 | 0.61 | **7.78** | **8.39** | **8.12** | **9.94** | **20** |

**Pros/cons — pick by need:**
- **Procedural** — **~2.6× faster** (no LLM), fully deterministic, no GPU LLM, trivially re-augmentable (see [augment.py](augment.py)). Terser and a bit more mechanical; slightly lower emotion/burst polish. Best for **large-scale labeling, training targets, and on-the-fly augmentation**.
- **LLM-assisted** — higher **overall / emotion / burst** quality and can be **coloured by a character archetype**; wins the head-to-head 20–12. Costs a batched LLM pass (~1 s/clip) and is non-deterministic. Best when **fluency and per-clip polish matter** (demos, showcase captions). See the [VAP repo](https://github.com/LAION-AI/Comprehensive-Voice-Acting-Annotation-Pipeline) for that path.

Both keep gender **correct or correctly unstated** (9.47 / 9.94) thanks to the gender gate.

Live examples on real character voices (procedural tags vs LLM-assisted, both with duration-gated inline bursts + gender gate):
**https://projects.laion.ai/procedural-voice-captions/character-captions/**

## Files

```
baseline_stats.json      # the baselines (99 dimensions + _meta)
caption.py               # caption() / caption_detail() + 11 templates (incl. tags) + gender gate  (stdlib only)
augment.py               # on-the-fly caption augmentation from stored scores (score once, caption many times)
compute_baseline.py      # rebuild baseline_stats.json (staged)
burst_captions.py        # scoring -> timestamps -> detect -> classify (duration+no_burst gate) -> insert bursts inline
build_burst_demo.py      # renders docs/burst-captions/ from real audio
vocalburst_taxonomy.json # 82 vocal-burst classes (class order for the classifier)
examples/                # real complete predictions (dims + emo + genu + blend)
requirements.txt
```

## License / attribution

Built for the LAION voice-acting effort. Models and taxonomies are the
respective LAION / third-party releases linked above.
