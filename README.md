# Procedural Voice Captions

Turn a speech clip's **model predictions** into a compact, human-readable
**caption** that describes the voice by how far it deviates from the *average
voice*.

> **🔊 [Live demo grid →](https://projects.laion.ai/procedural-voice-captions/)**
> 100 real multilingual speech clips, each with an audio player and the full
> caption written automatically by this module (also at
> [laion-ai.github.io/procedural-voice-captions](https://laion-ai.github.io/procedural-voice-captions/)).

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

## Files

```
baseline_stats.json     # the baselines (99 dimensions + _meta)
caption.py              # caption() / caption_detail() + CLI  (stdlib only)
compute_baseline.py     # rebuild baseline_stats.json (staged)
examples/               # real complete predictions (dims + emo + genu + blend)
requirements.txt
```

## License / attribution

Built for the LAION voice-acting effort. Models and taxonomies are the
respective LAION / third-party releases linked above.
