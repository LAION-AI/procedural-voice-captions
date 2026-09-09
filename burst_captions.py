#!/usr/bin/env python3
r"""Procedural voice captions **with detected vocal bursts inserted**.

This extends the procedural captioner in ``caption.py`` so that non-verbal
**vocal bursts** (laughs, gasps, sighs, screams, sobs, grunts, ...) that the
audio actually contains are located, named, and written into the captions.

For each clip the pipeline produces:

* a **global caption** — Top-5 VoiceNet dims + Top-3 EmoNet emotions +
  genuineness + Age + Gender + Tempo (bursts are *not* placed at global level);
* **per-sentence captions** — Top-3 VoiceNet dims + Top-3 emotions (no
  Age/Gender/Tempo — those are global only);
* **two selectable variants** of burst insertion (never both in one caption):

  - **Variant A — "locator" (precise position).** Run the burst *locator*
    (``laion/vocalburst-locator``, checkpoint ``model_v2.pt``, a 50 fps per-frame
    burst-probability model with a fixed 30 s / 1500-frame receptive field) over
    the whole clip in **overlapping 30 s windows**, stitch the per-frame
    probabilities back onto one global timeline, then extract events. Every
    detected span's audio is fed to the burst classifier
    (``laion/vocal-burst-detector-v2``: VoiceCLAP ``encode_waveform`` -> MLP,
    83 softmax outputs = 82 taxonomy classes + ``no_burst``). If
    ``P(no_burst) < 0.5`` the span's **top-1** class is inserted as its own
    ``(Class Name)`` inline at the span's time, positioned between the two ASR
    words nearest that moment.
  - **Variant B — "sentence-level".** Each sentence segment's audio is run
    through the same classifier; if ``P(no_burst) >= 0.5`` no burst is attached,
    otherwise the **top-1** class is woven into that sentence's caption with a
    small procedurally-generated phrase (``... punctuated by a (Gasp).``).

A **second naming head** can be switched on next to the first: ``BURST_X2=1`` adds
``laion/vocal-burst-detector-x2``, which names a span over **17 classes = 16 bursts +
``no_burst``** instead of 83, and writes each class only as strongly as its measured
per-class recall allows (``(Chuckle)`` / ``(Breathy Giggle?)`` / ``(Breath)`` /
``(Vocal Burst)``). It is an **addition, never a replacement**: its 16 classes are a
strict subset of this repo's 82-class taxonomy, and on the corpus this repo captions
83.8 % of burst events carry a label it cannot emit at all. So ``BURST_LABEL_SOURCE``
stays at ``v2`` by default — the x2 head is recorded beside every span and changes no
caption until you ask it to. See :mod:`burst_x2` for the whole argument, the tiers and
the ``union`` policy that gives each vocabulary the spans it actually covers.

Scoring stack (all reused from the LAION voice stack):

* VoiceNet 57 dims + genuineness + vocal-burst blend — VoiceCLAP-commercial
  embedding -> best-per-dim MLP heads (``VOICENET_REPO``).
* EmoNet-40 — ``laion/BUD-E-Whisper`` encoder -> ``laion/Empathic-Insight-Voice-Plus``
  per-emotion heads. Optional (set ``BC_EMONET=0`` to skip; captions then fall
  back to VoiceNet + genuineness only).
* Parakeet TDT (``nvidia/parakeet-tdt-0.6b-v3``) — word + sentence timestamps.
* Burst locator ``laion/vocalburst-locator`` (``model_v2.pt``) + burst classifier
  ``laion/vocal-burst-detector-v2`` (VoiceCLAP -> 219k-param MLP, softmax over 83).

Everything is env-driven (see the ``ENV`` block) and **every model defaults to a
public HuggingFace repo**, so a clean checkout runs without any local files.
Import ``BurstCaptioner`` and call :meth:`BurstCaptioner.process`, or run this
file as a CLI over a set of audio files to dump a per-clip results JSON.
"""
import os, sys, io, json, glob, math, subprocess, argparse

# --- threading discipline: this box thrashes without it (set BEFORE torch) ----
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/tmp/hf_cache"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from caption import (caption_detail, load_baseline, render_caption, _groups,  # noqa: E402
                     EI_GENDER_GATE, stable_hash)
import burst_x2                                                              # noqa: E402

# --------------------------------------------------------------------------- #
# ENV — every model location is overridable; the defaults are public HF repos, so
# an unset environment downloads everything and just works.
DEVICE          = os.environ.get("BC_DEVICE", "cuda:0")
# Local directory laid out like `laion/voicenet-dimension-predictors-commercial`
# (regression/*.pt); empty -> snapshot_download that repo.
VOICENET_REPO   = os.environ.get("VOICENET_REPO", "")
VOICENET_HF     = os.environ.get("VOICENET_HF", "laion/voicenet-dimension-predictors-commercial")
VOICECLAP_HF    = os.environ.get("VOICECLAP_REPO", "laion/voiceclap-commercial")
GENU_PT         = os.environ.get("GENU_PT", "")     # empty -> HF
GENU_HF         = (os.environ.get("GENU_HF_REPO", "laion/voiceclap-commercial-genuineness"),
                   os.environ.get("GENU_HF_FILE", "genuineness_head.pt"))
BLEND_PT        = os.environ.get("BLEND_PT", "")    # empty -> HF
BLEND_HF        = (os.environ.get("BLEND_HF_REPO", "laion/voiceclap-commercial-vocalburst-blend"),
                   os.environ.get("BLEND_HF_FILE", "blend_head_commercial.pt"))
# --- the two vocal-burst models (see README: LOCATOR finds, CLASSIFIER names) --
LOCATOR_REPO    = os.environ.get("BURST_LOCATOR_REPO", "laion/vocalburst-locator")
LOCATOR_FILE    = os.environ.get("BURST_LOCATOR_FILE", "model_v2.pt")   # v2 — 4.0x v1's F1 on real audio
LOCATOR_PT      = os.environ.get("BURST_LOCATOR_PT", "")               # local .pt override
BURST_CLF_REPO  = os.environ.get("BURST_CLF_REPO", "laion/vocal-burst-detector-v2")
BURST_CLF_FILE  = os.environ.get("BURST_CLF_FILE", "vocal_burst_mlp_v2.pt")
BURST_MLP_PT    = os.environ.get("BURST_MLP_PT", "")                   # local .pt override
TAXONOMY_JSON   = os.environ.get("VOCALBURST_TAXONOMY", os.path.join(HERE, "vocalburst_taxonomy.json"))
PARAKEET_MODEL  = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
ASR_HELPERS     = os.environ.get("ASR_HELPERS_DIR", "")                # optional out-of-repo helpers
USE_EMONET      = os.environ.get("BC_EMONET", "1") != "0"

# --- locator post-processing --------------------------------------------------
# These are the operating point measured for `model_v2.pt` on 992 held-out real,
# in-the-wild clips (event F1 0.607 @ IoU 0.5, P 0.678 / R 0.669). They are NOT the
# defaults that shipped with locator v1 (0.65 / 0.30 / 0.50): real ground-truth
# bursts have a median duration of ~180 ms, so a 0.5 s minimum discards ~96 % of
# them. On an unchanged checkpoint, moving to these three values took event F1 from
# 0.243 to 0.598 — a larger effect than any training change.
LOCATOR_THR     = float(os.environ.get("BURST_LOCATOR_THR", "0.50"))
MERGE_GAP       = float(os.environ.get("BURST_MERGE_GAP", "0.10"))
MIN_BURST_DUR   = float(os.environ.get("BURST_MIN_DUR", "0.10"))
NOBURST_GATE    = float(os.environ.get("BURST_NOBURST_GATE", "0.5"))
# The locator's receptive field is a fixed 30 s / 1500-frame window. Longer audio is
# scanned in overlapping windows whose per-frame probabilities are stitched back onto
# one global timeline before events are extracted (see `locator_probs`).
CHUNK_SEC       = float(os.environ.get("BURST_CHUNK_SEC", "30.0"))
CHUNK_OVERLAP   = float(os.environ.get("BURST_CHUNK_OVERLAP", "5.0"))
# v1 used a stricter duration floor for the transient smack/click/slap groups because the
# old multi-label classifier hallucinated `Slap Face` / `Lip Smack` on 0.1-0.2 s spans.
# The v2 classifier *masks* the four hand/body classes and `Blowing a Kiss` (they can never
# be predicted), and a 0.6 s floor contradicts the 180 ms median burst duration, so the
# extra floor is disabled by default. Set BURST_TRANSIENT_MIN_DUR to re-enable it.
TRANSIENT_MIN_DUR = float(os.environ.get("BURST_TRANSIENT_MIN_DUR", str(MIN_BURST_DUR)))
TRANSIENT_GROUPS  = {"mouth_and_lip_sounds", "tongue_clicks", "hand_and_body_sounds"}
# Extra audio taken on each side of a located span before it is handed to the classifier.
# Default 0 = classify exactly the span the locator found, which reproduces the classifier
# model card's own inference. Widening the cut does move labels (on the demo clips 0.25 s
# flips a 0.38 s span from `Contented Sigh` to `Surprised Gasp`), but we have no held-out
# measurement saying which is better, so the default stays at the faithful 0.
CLF_CONTEXT     = float(os.environ.get("BURST_CLF_CONTEXT", "0.0"))

# --- the x2 second-opinion head (see burst_x2.py) ------------------------------
# `BURST_LABEL_SOURCE` is the switch that matters. It stays at `v2`, so loading the x2
# head is observable but never rewrites a caption behind your back; `x2` / `union` opt in.
LABEL_SOURCE    = burst_x2.LABEL_SOURCE                 # v2 (default) | x2 | union
X2_HEAD         = burst_x2.X2_HEAD_NAME                 # large-v2 (default) | commercial
_X2_ENV         = os.environ.get("BURST_X2", "").strip()
# Asking for x2 labels implies loading the x2 head; BURST_X2=0 still wins, in which case
# the label source falls back to `v2` rather than half-configuring the pipeline.
USE_X2          = (_X2_ENV not in ("", "0")) or (LABEL_SOURCE != "v2" and _X2_ENV != "0")
if not USE_X2:
    LABEL_SOURCE = "v2"
# Score every located span with x2, not just the ones the v2 gate kept. Costs one extra
# forward pass per rejected span and makes the two heads' disagreement auditable.
X2_ALL_SPANS    = os.environ.get("BURST_X2_ALL_SPANS", "1") != "0"
# Optional second no_burst gate from the x2 head — OFF by default. The v2 gate alone
# decides *whether* a span is a burst; letting a 16-class head veto spans whose sound it
# has no word for would delete bursts rather than rename them.
X2_GATE         = burst_x2.X2_GATE

# Surface form for procedural captions. Default "tags" = terse, on-the-point tags (intensity +
# dimension as short words, no filler); set PROC_TEMPLATE to any TEMPLATE_NAMES value for prose.
PROC_TEMPLATE     = os.environ.get("PROC_TEMPLATE", "tags")
# Insert [pause X.Xs] markers from ASR word timestamps when the gap between consecutive words (or
# between sentences) is at least this many seconds. Set BURST_PAUSE_THR=0 to disable pauses.
PAUSE_THR         = float(os.environ.get("BURST_PAUSE_THR", "0.30"))
# A Parakeet-TDT token's duration is the encoder advance to the NEXT token, not the extent of the
# word. Consecutive spans therefore come out exactly contiguous (measured: 65.5 % of word pairs on
# 96 real clips have a gap of exactly 0.000 s) and the silence inside a pause is attributed to the
# word before it. So `end` alone under-reports pauses: on those clips the shipped code printed 233
# `[pause]` markers where 313 are audible — every marker real, but a quarter of the pauses missing,
# and the ones it prints understated by a median 0.02 s / mean 0.134 s / max 1.22 s.
#
# The fix is to trim each word's end back to where its speech actually stops, using the waveform
# that is already in hand. A blind duration cap — the correction used elsewhere in this project for
# a different question — was measured and rejected here: at 0.30 s it produces 431 markers of which
# 154 are invented (precision 0.643), because 681 of the 910 words longer than 0.30 s are simply
# long words. No fixed cap beats the energy rule; see README, "Pauses".
#
# `end_span` keeps the raw span to the next token, `end` becomes the speech end. Set
# BURST_WORD_END=raw for the old behaviour.
WORD_END          = os.environ.get("BURST_WORD_END", "energy")      # energy | raw
# Silence floor as a fraction of the clip's 90th-percentile frame RMS. The true-pause count moves
# 273 -> 348 across 0.03 ... 0.30, so nothing here hangs on the exact value; 0.10 is the middle.
SILENCE_FLOOR     = float(os.environ.get("BURST_SILENCE_FLOOR", "0.10"))
RMS_HOP, RMS_WIN  = 0.010, 0.025

SR = 16000
VN_TARGET = 480000          # 30 s @ 16 kHz for the VoiceNet embedding
LOC_FPS = 50                # locator frame rate (1500 frames / 30 s)


# --------------------------------------------------------------------------- #
# Model resolution — local path if given, otherwise the public HF repo.
def _hf_file(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename)


def resolve_voicenet_repo():
    """Directory containing `regression/*.pt` (57 VoiceNet dimension heads)."""
    if VOICENET_REPO:
        return VOICENET_REPO
    from huggingface_hub import snapshot_download
    return snapshot_download(VOICENET_HF, allow_patterns=["regression/*"])


def _ffmpeg():
    """Path to an ffmpeg binary: system ffmpeg, else the one imageio-ffmpeg ships."""
    from shutil import which
    exe = os.environ.get("FFMPEG_BIN") or which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Audio
def decode_16k(path):
    """Decode any audio file to a 1-D float32 16 kHz mono torch tensor.

    Uses ffmpeg when available (widest format coverage) and falls back to
    soundfile + torchaudio, which covers wav/flac/ogg/mp3 on modern libsndfile.
    """
    import numpy as np, torch, soundfile as sf, torchaudio
    data = sr = None
    exe = _ffmpeg()
    if exe:
        p = subprocess.run([exe, "-y", "-loglevel", "error", "-i", path,
                            "-ac", "1", "-ar", str(SR), "-f", "wav", "pipe:1"],
                           capture_output=True)
        if p.returncode == 0 and p.stdout:
            data, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=False)
    if data is None:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    w = torch.from_numpy(np.ascontiguousarray(data))
    if w.dim() == 2:
        w = w.mean(1)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.float()


def frame_rms(wav, sr=SR, hop_s=RMS_HOP, win_s=RMS_WIN):
    """Short-time RMS envelope at `hop_s`, computed in one pass from a cumulative sum."""
    import numpy as np
    x = np.asarray(wav.numpy() if hasattr(wav, "numpy") else wav, dtype=np.float64).reshape(-1)
    h, w = max(1, int(hop_s * sr)), max(1, int(win_s * sr))
    if x.size < w:
        return np.asarray([np.sqrt(np.mean(x ** 2)) if x.size else 0.0], dtype=np.float32)
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(1 + (x.size - w) // h) * h
    return np.sqrt((c[idx + w] - c[idx]) / w).astype(np.float32)


def trim_word_ends(wav, words, floor_frac=None, mode=None, sr=SR):
    """Move each word's `end` back to where its speech stops; keep the raw span in `end_span`.

    A Parakeet-TDT token's duration runs to the *next* token, so a word span swallows the
    silence that follows it and the gap between consecutive words comes out as 0.000 s
    (measured on 96 real clips: 65.5 % of word pairs). Every `[pause X.Xs]` marker in this
    repo is derived from that gap, so without this step a quarter of the audible pauses are
    never printed and the printed ones are too short.

    The trim is deliberately conservative and cannot invent a pause:

    * an end only ever moves **earlier**, never later, and never before the word's own start;
    * it moves only across frames whose RMS is below `floor_frac` of the clip's 90th-percentile
      frame RMS — i.e. across audible silence, not across quiet speech;
    * a word with no frame above the floor at all is left exactly as it was, because that is a
      threshold artefact rather than a measurement.

    Returns new dicts; the input list is not mutated. `mode="raw"` returns the words unchanged
    apart from `end_span`, which is always present so downstream code can rely on it.
    """
    import numpy as np
    out = []
    for w in words or []:
        w = dict(w)
        w.setdefault("end_span", w.get("end"))
        out.append(w)
    if (mode or WORD_END).lower() == "raw" or not out or wav is None:
        return out
    frac = SILENCE_FLOOR if floor_frac is None else float(floor_frac)
    if frac <= 0:
        return out
    rms = frame_rms(wav, sr=sr)
    thr = frac * float(np.percentile(rms, 90))
    if not (thr > 0):                       # digital silence, or a constant-level clip
        return out
    voiced = rms >= thr
    for w in out:
        s, e = w.get("start"), w.get("end_span")
        if s is None or e is None or e <= s:
            continue
        # Only frames whose whole window lies inside [s, e) count — the frame sitting exactly
        # on `e` already contains the next word's onset and would defeat the trim entirely.
        i0 = max(0, int(s / RMS_HOP))
        i1 = min(len(rms) - 1, int((e - RMS_WIN) / RMS_HOP))
        if i1 < i0:
            continue
        nz = np.nonzero(voiced[i0:i1 + 1])[0]
        if nz.size == 0:
            continue                        # no voiced frame in the span: leave it alone
        last = i0 + int(nz[-1])
        w["end"] = round(min(e, max(s, last * RMS_HOP + RMS_WIN)), 3)
    return out


def write_mp3(src, dst, bitrate="128k"):
    """Transcode `src` to a mono `bitrate` mp3 at `dst` (used by the demo builders)."""
    exe = _ffmpeg()
    if not exe:
        raise RuntimeError("no ffmpeg binary found (set FFMPEG_BIN or pip install imageio-ffmpeg)")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", src,
                    "-ac", "1", "-b:a", bitrate, dst], capture_output=True, check=False)
    return dst


# --------------------------------------------------------------------------- #
# VoiceNet heads (copied from the v3 infer_lib so this file is self-contained)
def _mlp_head(D, H, out):
    import torch.nn as nn
    class MLPHead(nn.Module):
        def __init__(s):
            super().__init__()
            s.f1 = nn.Linear(D, H); s.act = nn.GELU(); s.dp = nn.Dropout(0.0); s.f2 = nn.Linear(H, out)
        def forward(s, x):
            return s.f2(s.dp(s.act(s.f1(x))))
    return MLPHead()


def _load_head(path, dev):
    import numpy as np, torch
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck["state_dict"]
    D, H, out = sd["f1.weight"].shape[1], sd["f1.weight"].shape[0], sd["f2.weight"].shape[0]
    net = _mlp_head(D, H, out).to(dev).eval(); net.load_state_dict(sd)
    ck["net"] = net
    ck["mu_t"] = torch.tensor(np.asarray(ck["mu"], np.float32).reshape(-1), device=dev)
    ck["sd_t"] = torch.tensor(np.asarray(ck["sd"], np.float32).reshape(-1), device=dev)
    ck["out"] = out
    ck["k"] = len(ck.get("levels") or {}) or 7          # ordinal levels (0..k-1), not head out
    return ck


# --------------------------------------------------------------------------- #
# LOCATOR — `laion/vocalburst-locator`, whisper-small encoder (LoRA merged) with a
# per-frame head. Input is a fixed 30 s window; output is 1500 logits at 50 fps.
def _whisper_seg():
    import torch.nn as nn
    from transformers import WhisperModel
    class WhisperSeg(nn.Module):
        def __init__(s):
            super().__init__(); s.whisper = WhisperModel.from_pretrained("openai/whisper-small")
            d = s.whisper.config.d_model; h = max(256, d // 2)          # 768 -> 384
            s.proj = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1))
            s.temporal = nn.Sequential(nn.Conv1d(h, h, 7, padding=3), nn.GELU(), nn.Dropout(0.1))
            s.out = nn.Linear(h, 1)
        def forward(s, x):
            e = s.whisper.encoder(input_features=x).last_hidden_state
            h = s.proj(e).permute(0, 2, 1); h = s.temporal(h).permute(0, 2, 1)
            return s.out(h).squeeze(-1)
    return WhisperSeg()


def extract_events(probs, threshold=None, merge_gap=None, min_dur=None, fps=LOC_FPS):
    """Frame probabilities -> [(start_s, end_s, mean_confidence), ...].

    Identical post-processing to the model card's `extract_events`: threshold ->
    contiguous runs -> merge runs separated by less than `merge_gap` -> drop
    anything shorter than `min_dur`. Works on a timeline of any length, so a
    stitched multi-window probability track can be processed in one go.
    """
    import numpy as np
    thr = LOCATOR_THR if threshold is None else threshold
    gap = MERGE_GAP if merge_gap is None else merge_gap
    mind = MIN_BURST_DUR if min_dur is None else min_dur
    b = np.asarray(probs) > thr
    runs, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j + 1 < n and b[j + 1]:
                j += 1
            runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if (s - merged[-1][1]) / fps <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if (e - s) / fps >= mind:
            out.append((round(s / fps, 3), round(e / fps, 3), round(float(np.asarray(probs)[s:e].max()), 3)))
    return out


# --------------------------------------------------------------------------- #
# CLASSIFIER — `laion/vocal-burst-detector-v2`: frozen VoiceCLAP 768-d embedding ->
# Linear(768,256) -> BatchNorm -> GELU -> Dropout(0.3) -> Linear(256,83), softmax.
# 83 = 82 taxonomy classes + `no_burst` at index 82.
def _asr_helpers():
    """(tokens_to_words, sentences_from_words, split_sentences).

    Prefers the out-of-repo helper package when ``ASR_HELPERS_DIR`` points at one
    (historical behaviour on the LAION boxes); otherwise uses the bundled,
    dependency-free implementations in ``asr_words.py``.
    """
    if ASR_HELPERS and os.path.isdir(ASR_HELPERS):
        try:
            sys.path.insert(0, ASR_HELPERS)
            from run_parakeet import _tokens_to_words, _sentences_from_words   # noqa
            from asr_common import split_sentences                            # noqa
            return _tokens_to_words, _sentences_from_words, split_sentences
        except Exception:
            pass
    from asr_words import tokens_to_words, sentences_from_words, split_sentences
    return tokens_to_words, sentences_from_words, split_sentences


def _burst_mlp_v2(D=768, H=256, C=83, p=0.3):
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(D, H), nn.BatchNorm1d(H), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(H, C))
        def forward(s, x):
            return s.net(x)
    return MLP()


# --------------------------------------------------------------------------- #
# EmoNet-40 head (BUD-E-Whisper encoder embedding -> FullEmbeddingMLP)
EMOS = ['Affection', 'Amusement', 'Anger', 'Astonishment/Surprise', 'Awe', 'Bitterness', 'Concentration',
        'Confusion', 'Contemplation', 'Contempt', 'Contentment', 'Disappointment', 'Disgust', 'Distress', 'Doubt',
        'Elation', 'Embarrassment', 'Emotional Numbness', 'Fatigue/Exhaustion', 'Fear', 'Helplessness', 'Hope/Optimism',
        'Impatience and Irritability', 'Infatuation', 'Interest', 'Intoxication/Altered States', 'Jealousy & Envy',
        'Longing', 'Malevolence/Malice', 'Pain', 'Pleasure/Ecstasy', 'Pride', 'Relief', 'Sadness', 'Sexual Lust',
        'Shame', 'Sourness', 'Teasing', 'Thankfulness/Gratitude', 'Triumph']
_EMO_OVR = {"Hope/Optimism": "model_Hope_Enthusiasm_Optimism_best.pth",
            "Intoxication/Altered States": "model_Intoxication_Altered_States_of_Consciousness_best.pth"}


def _emo_file(e):
    return _EMO_OVR.get(e, f"model_{e.replace('/', '_').replace(' ', '_')}_best.pth")


def _full_embedding_mlp():
    import torch.nn as nn
    SEQ, EMB, PROJ, HID, DR = 1500, 768, 64, [64, 32, 16], [0.0, 0.1, 0.1, 0.1]
    class FullEmbeddingMLP(nn.Module):
        def __init__(s):
            super().__init__(); s.flatten = nn.Flatten(); s.proj = nn.Linear(SEQ * EMB, PROJ)
            L = [nn.ReLU(), nn.Dropout(DR[0])]; cur = PROJ
            for i, h in enumerate(HID):
                L += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(DR[i + 1])]; cur = h
            L.append(nn.Linear(cur, 1)); s.mlp = nn.Sequential(*L)
        def forward(s, x):
            return s.mlp(s.proj(s.flatten(x)))
    return FullEmbeddingMLP()


# --------------------------------------------------------------------------- #
class BurstCaptioner:
    """Loads every scoring model once and captions clips with inserted bursts."""

    def __init__(self, device=DEVICE, use_emonet=USE_EMONET, verbose=True, baseline=None,
                 use_x2=None, x2_head=None, label_source=None):
        import torch
        torch.set_num_threads(2)                      # critical: box thrashes otherwise
        self.dev = device
        self.verbose = verbose
        # `baseline` is a name from caption.BASELINES ("default"/"dramabox"/"emolia") or a
        # path; None -> $PVC_BASELINE -> the bundled in-domain default.
        self.baseline = load_baseline(baseline)
        self._log("loading VoiceCLAP-commercial embedder + VoiceNet heads ...")
        from transformers import AutoModel
        vn_dir = resolve_voicenet_repo()
        vc_src = (os.path.join(VOICENET_REPO, "voiceclap_commercial")
                  if VOICENET_REPO and os.path.isdir(os.path.join(VOICENET_REPO, "voiceclap_commercial"))
                  else VOICECLAP_HF)
        self.vc = AutoModel.from_pretrained(vc_src, trust_remote_code=True).to(device).eval()
        self.reg = {}
        for p in sorted(glob.glob(os.path.join(vn_dir, "regression", "*.pt"))):
            self.reg[os.path.basename(p)[:-3]] = _load_head(p, device)
        if not self.reg:
            raise RuntimeError(f"no VoiceNet regression heads under {vn_dir}/regression")
        self.dims = sorted(self.reg)
        self.genu = _load_head(GENU_PT or _hf_file(*GENU_HF), device)
        self.blend = _load_head(BLEND_PT or _hf_file(*BLEND_HF), device)
        self._log(f"VoiceNet: {len(self.dims)} dimension heads + genuineness + blend")

        # burst taxonomy / classes (bundled; identical order to the classifier's classes.json)
        tax = json.load(open(TAXONOMY_JSON))["categories"]
        self.classes = [n for g, d in tax.items() for n in d.get("items", {})] + ["no_burst"]
        self.label_group = {n: g for g, d in tax.items() for n in d.get("items", {})}
        self.NB = len(self.classes) - 1

        self._log(f"loading LOCATOR {LOCATOR_REPO}/{LOCATOR_FILE} ...")
        from transformers import WhisperFeatureExtractor
        self.seg = _whisper_seg().to(device).eval()
        self.seg.load_state_dict(
            torch.load(LOCATOR_PT or _hf_file(LOCATOR_REPO, LOCATOR_FILE), map_location="cpu"),
            strict=True)
        self.wfe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

        self._log(f"loading CLASSIFIER {BURST_CLF_REPO}/{BURST_CLF_FILE} ...")
        ck = torch.load(BURST_MLP_PT or _hf_file(BURST_CLF_REPO, BURST_CLF_FILE),
                        map_location="cpu", weights_only=False)
        arch = ck.get("arch", {})
        self.clf = _burst_mlp_v2(arch.get("D", 768), arch.get("H", 256),
                                 arch.get("C", len(self.classes)), arch.get("dropout", 0.3))
        self.clf.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        self.clf = self.clf.to(device).eval()
        if ck.get("classes"):                        # trust the checkpoint's own class order
            self.classes = list(ck["classes"])
            self.NB = self.classes.index("no_burst") if "no_burst" in self.classes else len(self.classes) - 1
        # Classes folded into `no_burst` during v2 training (hand/body impacts + a blown kiss are
        # not audible vocal bursts). They are masked so they can never be predicted.
        self.folded = self._folded_indices()
        self._log(f"burst classes: {self.NB} + no_burst, {len(self.folded)} folded/masked")

        # --- the x2 second opinion: 16 burst classes + no_burst, recall-tiered ------
        # Built AFTER `self.vc`, on purpose: with `x2_head="commercial"` the x2 ensemble
        # runs on the encoder that is already in memory, so it costs five small MLPs and
        # no second tower. `large-v2` is the better head and brings its own 3584-d encoder.
        self.x2 = None
        self.x2_table = None
        self.x2_head = x2_head or X2_HEAD
        self.label_source = (label_source or LABEL_SOURCE).lower()
        want_x2 = USE_X2 if use_x2 is None else bool(use_x2)
        if not want_x2:
            self.label_source = "v2"
        else:
            self._log(f"loading x2 CLASSIFIER {burst_x2.X2_REPO} (head {self.x2_head}) ...")
            embed_fn = None
            if self.x2_head == "commercial":
                embed_fn = lambda w: burst_x2.embed_commercial(self.vc, w, self.dev)  # noqa: E731
            self.x2 = burst_x2.X2Classifier(head=self.x2_head, device=device,
                                            embed_fn=embed_fn, verbose=verbose)
            self.x2_table = self.x2.table
            tiers = self.x2_table.summary()
            self._log(f"x2: {len(self.x2.classes) - 1} burst classes + no_burst, tiers "
                      + ", ".join(f"{k} {len(v)}" for k, v in tiers.items())
                      + f"; recall source '{self.x2_table.source}'"
                      + f"; label source '{self.label_source}'")

        self._log(f"loading Parakeet ({PARAKEET_MODEL}) ...")
        from transformers import AutoProcessor, ParakeetForTDT
        self._tokens_to_words, self._sentences_from_words, self._split_sentences = _asr_helpers()
        self.pk_proc = AutoProcessor.from_pretrained(PARAKEET_MODEL)
        self.pk = ParakeetForTDT.from_pretrained(PARAKEET_MODEL).to(device).eval()

        self.use_emonet = use_emonet
        if use_emonet:
            self._log("loading EmoNet (BUD-E-Whisper encoder + Empathic-Insight heads) ...")
            from transformers import WhisperForConditionalGeneration
            from huggingface_hub import snapshot_download
            self.emo_proc = AutoProcessor.from_pretrained("laion/BUD-E-Whisper")
            wm = WhisperForConditionalGeneration.from_pretrained(
                "laion/BUD-E-Whisper", torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
            self.emo_enc = wm.get_encoder()
            md = snapshot_download("laion/Empathic-Insight-Voice-Plus",
                                   ignore_patterns=["*.mp3", "*.md", ".gitattributes"])
            avail = {os.path.basename(p): p for p in glob.glob(md + "/**/*.pth", recursive=True)}
            self.emo_heads = {}
            from collections import OrderedDict
            for emo in EMOS:
                m = _full_embedding_mlp()
                sd = torch.load(avail[_emo_file(emo)], map_location="cpu")
                if any(k.startswith("_orig_mod.") for k in sd):
                    sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in sd.items())
                m.load_state_dict(sd)
                self.emo_heads[emo] = m.to(device).eval().half()
            # Empathic-Insight Gender expert (same FullEmbeddingMLP) -> bipolar -2 masc .. +2 fem.
            # Used only as a confidence GATE on the VoiceNet gender phrase (see caption.py).
            self.gender_head = None
            gpath = avail.get("model_Gender_best.pth")
            if gpath:
                gm = _full_embedding_mlp()
                gsd = torch.load(gpath, map_location="cpu")
                if any(k.startswith("_orig_mod.") for k in gsd):
                    gsd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in gsd.items())
                gm.load_state_dict(gsd)
                self.gender_head = gm.to(device).eval().half()
        self._log("all models ready.")

    def _log(self, *a):
        if self.verbose:
            print("[burst_captions]", *a, flush=True)

    # -- low-level scorers ---------------------------------------------------- #
    def _embed_vn(self, wav):
        """VoiceCLAP embedding for VoiceNet (30 s pad/trunc)."""
        import torch
        T = wav.shape[0]
        w = wav[:VN_TARGET] if T >= VN_TARGET else torch.nn.functional.pad(wav, (0, VN_TARGET - T))
        with torch.no_grad():
            return self.vc.encode_waveform(w.unsqueeze(0).to(self.dev)).float()

    def _apply_head(self, head, E, clamp=None):
        import torch
        with torch.no_grad():
            z = (E - head["mu_t"]) / head["sd_t"]
            y = head["net"](z).squeeze(-1)
            if clamp is not None:
                y = y.clamp(*clamp)
            return y.cpu().numpy()

    def voicenet(self, wav):
        """Return {'dims': {CODE: r}, 'genu': g, 'blend': b} for a waveform."""
        E = self._embed_vn(wav)
        dims = {}
        for d in self.dims:
            kk = self.reg[d]["k"]                        # ordinal level count (e.g. 7 -> clamp 0..6)
            dims[d] = float(self._apply_head(self.reg[d], E, clamp=(0, kk - 1))[0])
        return {"dims": dims,
                "genu": float(self._apply_head(self.genu, E, clamp=(0, 6))[0]),
                "blend": float(self._apply_head(self.blend, E, clamp=(0, 10))[0])}

    def emonet_embed(self, wav):
        """BUD-E-Whisper encoder embedding [1500,768] fp16 on CPU (or None)."""
        if not self.use_emonet:
            return None
        import torch
        inp = self.emo_proc(wav.numpy(), sampling_rate=SR, return_tensors="pt",
                            padding="max_length", truncation=True)
        feats = inp.input_features.to(self.dev).to(next(self.emo_enc.parameters()).dtype)
        with torch.no_grad():
            e = self.emo_enc(feats, return_dict=True).last_hidden_state.half()
        return e.squeeze(0).cpu()

    def gender_ei(self, embed):
        """Empathic-Insight Gender expert on a BUD-E-Whisper embedding: -2 (masculine) .. +2
        (feminine), or None if the encoder/head is unavailable."""
        import torch
        if not self.use_emonet or getattr(self, "gender_head", None) is None or embed is None:
            return None
        with torch.no_grad():
            return float(self.gender_head(embed.unsqueeze(0).to(self.dev)).squeeze().float().item())

    def emonet_score(self, embeds):
        """Run the 40 heads once over a stack of embeddings -> list of {emo: score}."""
        import torch
        if not self.use_emonet or not embeds:
            return [{} for _ in embeds]
        E = torch.stack(embeds)                         # [N,1500,768] cpu fp16
        out = [{} for _ in range(len(embeds))]
        with torch.no_grad():
            for emo, m in self.emo_heads.items():
                for j in range(0, len(embeds), 16):
                    x = E[j:j + 16].to(self.dev)
                    y = m(x).squeeze(-1).float().cpu().tolist()
                    for k, v in enumerate(y):
                        out[j + k][emo] = round(float(v), 4)
        return out

    def parakeet(self, wav):
        """Return (transcript, words, sentences) with second-level timestamps."""
        import torch
        dur = wav.shape[0] / SR
        feats = self.pk_proc(wav.numpy(), sampling_rate=SR, return_tensors="pt")
        feats = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in feats.items()}
        with torch.no_grad():
            o = self.pk.generate(**feats, max_new_tokens=int(dur * 20) + 64)
        text = (self.pk_proc.batch_decode(o.sequences, skip_special_tokens=True)[0] or "").strip()
        durs = o.durations
        if not torch.is_tensor(durs):
            durs = torch.as_tensor(durs)
        toks = []
        try:
            dec = self.pk_proc.decode(o.sequences, durations=durs)
            if isinstance(dec, (list, tuple)) and len(dec) >= 2:
                ts = dec[1]
                if isinstance(ts, list) and ts and isinstance(ts[0], list):
                    toks = ts[0]
        except Exception:
            pass
        words = self._tokens_to_words(toks, 1.0) if toks else []
        # TDT spans run to the next token; trim each word's end back to its speech so that the
        # gap between two words is the silence a listener actually hears (see `trim_word_ends`).
        # Done here, before sentence assembly, so sentence ends are speech ends too.
        words = trim_word_ends(wav, words)
        sents = self._sentences_from_words(text, words) if words else \
            [dict(text=s, start=None, end=None) for s in self._split_sentences(text)]
        return text, words, sents

    # -- LOCATOR: 30 s windowing ---------------------------------------------- #
    def locator_probs(self, wav):
        """Per-frame burst probability at 50 fps over the WHOLE clip.

        The locator has a hard 30 s / 1500-frame input. Audio longer than that is cut
        into windows of ``CHUNK_SEC`` with ``CHUNK_OVERLAP`` seconds of overlap
        (hop = 30 - overlap). Each window is scored independently and its 1500 frame
        probabilities are written back into one global frame track at the window's
        offset. Inside an overlap region the two windows are **cross-faded**: each
        window's contribution is linearly ramped from 0 at its shared edge to 1 at the
        inner end of the overlap, and the track is the weight-normalised sum. So a
        frame is always dominated by the window that sees it with the most context,
        transitions are continuous, and — because events are extracted **after**
        stitching — a burst that straddles a seam stays one event and can never be
        reported twice.
        """
        import numpy as np, torch
        n_samp = int(wav.shape[0])
        dur = n_samp / SR
        win_samp = int(CHUNK_SEC * SR)
        ov = max(0.0, min(CHUNK_OVERLAP, CHUNK_SEC / 2))
        hop_samp = max(1, int((CHUNK_SEC - ov) * SR))
        starts = [0] if n_samp <= win_samp else list(range(0, max(1, n_samp - win_samp + hop_samp), hop_samp))
        if starts[-1] + win_samp < n_samp:                  # make sure the tail is covered
            starts.append(max(0, n_samp - win_samp))

        n_frames = max(1, int(math.ceil(dur * LOC_FPS)))
        acc = np.zeros(n_frames, np.float64)
        wgt = np.zeros(n_frames, np.float64)
        ov_frames = int(round(ov * LOC_FPS))
        for k, s0 in enumerate(starts):
            seg = wav[s0:s0 + win_samp]
            if seg.shape[0] < win_samp:
                seg = torch.nn.functional.pad(seg, (0, win_samp - seg.shape[0]))
            f = self.wfe(seg.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(self.dev)
            with torch.no_grad():
                pr = torch.sigmoid(self.seg(f))[0].float().cpu().numpy()   # [1500]
            f0 = int(round(s0 / SR * LOC_FPS))
            m = min(len(pr), n_frames - f0)
            if m <= 0:
                continue
            w = np.ones(m, np.float64)
            if ov_frames > 0 and len(starts) > 1:
                r = min(ov_frames, m)
                if k > 0:                                    # ramp in over the shared head
                    w[:r] *= np.linspace(0.0, 1.0, r, endpoint=False) + 1.0 / max(r, 1)
                if k < len(starts) - 1:                      # ramp out over the shared tail
                    w[-r:] *= (np.linspace(0.0, 1.0, r, endpoint=False) + 1.0 / max(r, 1))[::-1]
            acc[f0:f0 + m] += pr[:m] * w
            wgt[f0:f0 + m] += w
        probs = np.where(wgt > 0, acc / np.maximum(wgt, 1e-9), 0.0)
        return probs.astype(np.float32), len(starts)

    def locator_spans(self, wav):
        """Locator over the whole clip -> [(start_s, end_s, peak_prob), ...] (global timeline)."""
        probs, _ = self.locator_probs(wav)
        return extract_events(probs)

    # -- CLASSIFIER ----------------------------------------------------------- #
    def _folded_indices(self):
        """Indices the v2 classifier masks (folded into `no_burst` during training)."""
        idx = set()
        try:
            fold = json.load(open(_hf_file(BURST_CLF_REPO, "folded_classes.json")))
            for name, meta in fold.items():
                i = meta.get("original_index")
                if i is None and name in self.classes:
                    i = self.classes.index(name)
                if i is not None:
                    idx.add(int(i))
        except Exception:
            # offline / older checkpoint: fall back to the documented list
            for name in ("Blowing a Kiss", "Finger Snaps", "Hand Scratching Head",
                         "Hand Slaps", "Slap Face"):
                if name in self.classes:
                    idx.add(self.classes.index(name))
        return sorted(i for i in idx if 0 <= i < len(self.classes))

    def classify_burst(self, seg_wav):
        """VoiceCLAP -> v2 MLP -> softmax. Returns (label_or_None, prob, p_noburst).

        `label` is None when the `no_burst` class wins the gate; otherwise it is the
        top-1 over the *recognised* burst classes (folded classes are masked out)."""
        import torch
        if seg_wav.shape[0] < int(0.04 * SR):
            return None, 0.0, 1.0
        with torch.no_grad():
            e = self.vc.encode_waveform(seg_wav.unsqueeze(0).to(self.dev)).float()
            if e.dim() == 1:
                e = e.unsqueeze(0)
            logits = self.clf(e)[0]
            for i in self.folded:
                logits[i] = float("-inf")
            p = torch.softmax(logits, -1).cpu().numpy()
        p_nb = float(p[self.NB])
        if p_nb >= NOBURST_GATE:
            return None, p_nb, p_nb
        pp = p.copy(); pp[self.NB] = -1.0
        top = int(pp.argmax())
        return self.classes[top], float(p[top]), p_nb

    # -- the x2 second opinion ------------------------------------------------ #
    def classify_x2(self, cuts):
        """Name a batch of already-cut spans with the x2 head. [] when x2 is not loaded.

        One call per clip, not per span: the ensemble batches by length internally, and
        with the `large-v2` encoder a per-span call would pay a 3584-d forward pass each
        time. Returns one dict per cut, in input order, dropping nothing — the `no_burst`
        class is masked out of the argmax because the v2 gate has already ruled on whether
        these spans are bursts (its probability is still reported)."""
        if self.x2 is None or not cuts:
            return []
        return self.x2.classify(cuts)

    def merge_burst_label(self, v2_label, x2_res):
        """Which vocabulary writes this span, and what it may say. See burst_x2.merge_label.

        `v2_label is None` means the v2 `no_burst` gate rejected the span, and that stays
        the last word regardless of policy: the x2 head names bursts, it does not vote on
        their existence (unless BURST_X2_GATE is set explicitly)."""
        if v2_label is None:
            return {"label": None, "written": None, "vocab": None, "tier": None,
                    "source": None, "policy": self.label_source, "out_of_x2_vocab": None}
        if self.x2 is None:
            return {"label": v2_label, "written": v2_label, "vocab": "v2-83", "tier": "named",
                    "source": "v2", "policy": "v2", "out_of_x2_vocab": None}
        return burst_x2.merge_label(v2_label, x2_res, self.x2_table, self.label_source)

    def x2_meta(self):
        """What the x2 head was configured to be, recorded on every clip so a stored result
        can always be read back against the thresholds that produced it."""
        if self.x2 is None:
            return {"enabled": False, "label_source": "v2",
                    "classifier": f"{BURST_CLF_REPO}/{BURST_CLF_FILE}", "n_classes": 83}
        t = self.x2_table
        return {"enabled": True, "repo": burst_x2.X2_REPO, "head": self.x2_head,
                "encoder": t.encoder, "encoder_dim": self.x2.dim,
                "n_members": len(self.x2.members), "n_classes": len(self.x2.classes),
                "label_source": self.label_source, "recall_source": t.source,
                "thresholds": {"strict_min": t.strict_min, "hedge_min": t.hedge_min,
                               "family_min": t.family_min},
                "tiers": {k: len(v) for k, v in t.summary().items()},
                "no_burst_gate": X2_GATE or None, "all_spans": X2_ALL_SPANS}

    @staticmethod
    def _x2_fields(x2_res):
        """The x2 columns attached to every span, whether or not it writes the caption."""
        if not x2_res:
            return {}
        return {"x2_label": x2_res["label"], "x2_prob": round(x2_res["prob"], 3),
                "x2_top": [[n, round(p, 3)] for n, p in x2_res["labels"]],
                "x2_family": x2_res["family"],
                "x2_p_noburst": round(x2_res["no_burst_prob"], 3),
                "x2_tier": x2_res["tier"], "x2_written": x2_res["written"],
                "x2_recall": x2_res["recall"], "x2_family_recall": x2_res["family_recall"]}

    # -- caption composition -------------------------------------------------- #
    def global_caption(self, preds, seed=0, ei_gender=None, template=None):
        """Top-5 VoiceNet + Top-3 emotions + genuineness + Age/Gender/Tempo. `ei_gender`
        (Empathic-Insight gender, -2..+2) gates the gender phrase: near zero -> omitted.
        `template` defaults to PROC_TEMPLATE (terse 'tags' by default)."""
        d = caption_detail(preds, self.baseline, k_voicenet=5, k_emonet=3, synonym_seed=seed,
                           template=(template or PROC_TEMPLATE), ei_gender=ei_gender)
        return render_caption(d), d

    def sentence_caption(self, preds, seed=0, template=None):
        """Top-3 VoiceNet + Top-3 emotions ONLY (no Age/Gender/Tempo, no quality). Defaults to the
        terse 'tags' surface form (PROC_TEMPLATE)."""
        tmpl = template or PROC_TEMPLATE
        d = caption_detail(preds, self.baseline, k_voicenet=3, k_emonet=3,
                           always_on=[], synonym_seed=seed, template=tmpl)
        if tmpl == "tags":
            tags = ([e["tag"] for e in d["voicenet"] if e.get("tag")]
                    + [e["tag"] for e in d["emotions"] if e.get("tag")])
            txt = ", ".join(tags) if tags else "unremarkable"
            return txt, {"voicenet": d["voicenet"], "emotions": d["emotions"]}
        vn = [e["phrase"] for e in d["voicenet"]]
        emo = [e["phrase"] for e in d["emotions"]]
        parts = vn + emo
        txt = ("Delivered " + "; ".join(parts) + ".") if parts else "An even, unremarkable delivery."
        return txt, {"voicenet": d["voicenet"], "emotions": d["emotions"]}

    # -- burst insertion ------------------------------------------------------ #
    @staticmethod
    def variant_a(words, bursts):
        """Inline transcript with `(Class)` inserted at each burst time between the
        nearest words. `bursts` = [(start_s,end_s,label,prob), ...]. Returns a string."""
        bl = sorted(bursts, key=lambda b: b[0])
        words_ts = [w for w in words if w.get("start") is not None]
        n = len(words_ts)
        if n == 0:
            # no ASR timestamps -> just the bursts in time order (non-speech clip)
            return " ".join(f"({lab})" for _, _, lab, _ in bl).strip()
        toks = []
        bi = 0
        prev_end = None
        for idx in range(n + 1):
            # flush any burst whose midpoint falls before the next word's start
            nxt_start = words_ts[idx]["start"] if idx < n else float("inf")
            burst_here = False
            while bi < len(bl):
                a, b, lab, pr = bl[bi]
                if (a + b) / 2.0 <= nxt_start:
                    toks.append(f"({lab})")
                    bi += 1; burst_here = True
                else:
                    break
            if idx < n:
                w = words_ts[idx]
                # [pause X.Xs] on a word gap the burst didn't already fill
                if PAUSE_THR > 0 and prev_end is not None and not burst_here:
                    gap = w["start"] - prev_end
                    if gap >= PAUSE_THR:
                        toks.append(f"[pause {gap:.1f}s]")
                toks.append(w["w"])
                prev_end = w.get("end", w["start"])
        return " ".join(toks).strip()

    def dur_ok(self, label, dur):
        """Duration gate, applied *after* the locator's own `min_duration` post-processing.

        With locator v2 the floor is 0.10 s (real bursts have a ~180 ms median duration), and the
        stricter transient floor of v1 is disabled by default — the v2 classifier can no longer
        emit the hand/body classes it was there to suppress. Both remain tunable via
        BURST_MIN_DUR / BURST_TRANSIENT_MIN_DUR. Returns (ok, floor_used)."""
        if label is None:
            return False, 0.0
        floor = TRANSIENT_MIN_DUR if self.label_group.get(label) in TRANSIENT_GROUPS else MIN_BURST_DUR
        return dur >= floor, floor

    def _assign_bursts(self, sent_out, words, kept_full):
        """Assign each kept locator burst to a sentence (by time overlap, nearest as fallback) and
        build one inline SCRIPT line per sentence with `(Burst)` placed inline via word timestamps.
        Guarantees every kept burst appears exactly once."""
        S = len(sent_out)
        buckets = [[] for _ in range(max(S, 1))]
        for x in kept_full:
            mid = (x["start"] + x["end"]) / 2.0
            best, bestd = None, float("inf")
            for i, s in enumerate(sent_out):
                a, b = s.get("start"), s.get("end")
                if a is None or b is None:
                    continue
                if a - 0.2 <= mid <= b + 0.2:
                    best = i; break
                d = min(abs(mid - a), abs(mid - b))
                if d < bestd:
                    bestd, best = d, i
            buckets[best if best is not None else 0].append(x)
        lines = []
        prev_send = None
        for i, s in enumerate(sent_out):
            a, b = s.get("start"), s.get("end")
            sw = [w for w in words if w.get("start") is not None and a is not None and b is not None
                  and a - 1e-6 <= w["start"] <= b + 1e-6]
            # `written` is the recall-tiered text (== `label` whenever x2 is off or `named`)
            bx = sorted(buckets[i], key=lambda x: x["start"])
            sb = [(x["start"], x["end"], x.get("written") or x["label"], x["prob"]) for x in bx]
            if sw and sb:
                inline = self.variant_a(sw, sb)
            elif sb:
                inline = (s["text"].rstrip(".") + " " + " ".join(f"({l})" for _, _, l, _ in sb)).strip()
            else:
                inline = s["text"]
            # [pause X.Xs] before this sentence when there was a silent gap after the previous one
            if PAUSE_THR > 0 and prev_send is not None and a is not None and (a - prev_send) >= PAUSE_THR:
                inline = f"[pause {a - prev_send:.1f}s] " + inline
            lines.append({"cue": s["caption"], "text": inline,
                          "bursts": [l for _, _, l, _ in sb],
                          "burst_vocab": [x.get("vocab") for x in bx]})
            if b is not None:
                prev_send = b
        return lines

    def procedural_caption(self, result):
        """The default procedural caption: GENERAL line + a SCRIPT with one line per sentence,
        each `(style cue) sentence text` with the KEPT (locator-detected, confirmed, duration-gated)
        vocal bursts inserted inline as `(Burst Name)`."""
        out = ["GENERAL: " + result["global_caption"], "SCRIPT:"]
        for ln in result.get("script_lines", []):
            out.append(f"({ln['cue']}) {ln['text']}")
        return "\n".join(out)

    @staticmethod
    def variant_b_phrase(label, seed=0, tier="named"):
        """A small procedurally-generated phrase attaching a burst to a sentence.

        `tier` comes from the x2 recall floor (`named` / `hedged` / `family` / `generic`).
        Only `hedged` changes the wording — *"with what sounds like a (Yawn?)"* rather than
        *"with an audible (Yawn)"* — because the other two tiers already carry their
        uncertainty in the label itself and saying it twice reads as mush. With the x2 head
        off, every burst is `named` and the six original templates are the only ones used,
        so this is byte-identical to the previous behaviour."""
        import random
        tpl = burst_x2.phrases_for(tier)
        rng = random.Random((seed or 0) ^ stable_hash(label))
        return rng.choice(tpl).replace("{b}", label)

    # -- full clip ------------------------------------------------------------ #
    def process(self, path, cid=None, mp3_out=None):
        """Score one audio file end-to-end and return a result dict with both
        burst-insertion variants. If `mp3_out` is given, also write a 120 kbps
        mono mp3 there for the demo player."""
        import torch
        cid = cid or os.path.splitext(os.path.basename(path))[0]
        wav = decode_16k(path)
        dur = wav.shape[0] / SR
        seed = stable_hash(cid) % (1 << 30)   # stable across processes (see caption.stable_hash)

        # --- ASR sentences + words ---
        transcript, words, sents = self.parakeet(wav)

        # --- global scoring ---
        g_vn = self.voicenet(wav)
        units_wav = [wav]                                # index 0 = global
        # sentence segments (fall back to whole clip when no sentence timestamps)
        seg_bounds = []
        for s in sents:
            a, b = s.get("start"), s.get("end")
            if a is None or b is None or b <= a:
                seg_bounds.append((0.0, dur))
            else:
                seg_bounds.append((max(0.0, a - 0.1), min(dur, b + 0.1)))
        for (a, b) in seg_bounds:
            units_wav.append(wav[int(a * SR):int(b * SR)])

        # --- EmoNet over all units in one pass ---
        emo_embeds = [self.emonet_embed(w) for w in units_wav] if self.use_emonet else None
        emo_scores = self.emonet_score(emo_embeds) if self.use_emonet else [{} for _ in units_wav]

        g_preds = {"dims": g_vn["dims"], "emo": emo_scores[0], "genu": g_vn["genu"], "blend": g_vn["blend"]}
        ei_gender = self.gender_ei(emo_embeds[0]) if self.use_emonet else None
        g_text, g_detail = self.global_caption(g_preds, seed, ei_gender=ei_gender)

        # --- per-sentence scoring + captions ---
        # Variant B's classification is hoisted out of the loop so the x2 head sees all the
        # sentence segments as one batch. With x2 off this is the same work in the same order.
        sent_v2 = [self.classify_burst(units_wav[i + 1]) for i in range(len(sents))]
        sent_x2 = {}
        if self.x2 is not None:
            ix = [i for i, (lab, _, _) in enumerate(sent_v2) if lab is not None]
            for i, r in zip(ix, self.classify_x2([units_wav[i + 1] for i in ix])):
                sent_x2[i] = r
        sent_out = []
        for i, s in enumerate(sents):
            seg = units_wav[i + 1]
            s_vn = self.voicenet(seg)
            s_preds = {"dims": s_vn["dims"], "emo": emo_scores[i + 1],
                       "genu": s_vn["genu"], "blend": s_vn["blend"]}
            s_text, _ = self.sentence_caption(s_preds, seed ^ (i + 1))
            # Variant B burst for this sentence
            lab, prob, p_nb = sent_v2[i]
            m = self.merge_burst_label(lab, sent_x2.get(i))
            b_text = s_text
            if m["written"]:
                b_text = (s_text.rstrip(".") + ", "
                          + self.variant_b_phrase(m["written"], seed ^ (i + 1), m["tier"]) + ".")
            sent_out.append({
                "text": s.get("text", ""), "start": s.get("start"), "end": s.get("end"),
                "caption": s_text,
                # `variant_b_burst` is what the caption says; `variant_b_label` is the raw
                # class, and `variant_b_vocab` which vocabulary earned the right to say it.
                "variant_b_burst": m["written"], "variant_b_prob": round(prob, 3),
                "variant_b_label": lab, "variant_b_vocab": m["vocab"],
                "variant_b_tier": m["tier"], "variant_b_caption": b_text,
                # raw scores for on-the-fly caption augmentation (see augment.py)
                "scores": {"dims": s_vn["dims"], "emo": emo_scores[i + 1],
                           "genu": round(s_vn["genu"], 4), "blend": round(s_vn["blend"], 4)},
                **self._x2_fields(sent_x2.get(i)),
            })

        # --- Variant A: locator over the whole clip (30 s windows, stitched) ---
        probs, n_windows = self.locator_probs(wav)
        spans = extract_events(probs)
        a_bursts = []
        x2_cuts, x2_ix = [], []
        for (a, b, pk) in spans:
            dur_span = b - a
            # give the classifier a little context around the located span
            ca, cb = max(0.0, a - CLF_CONTEXT), min(dur, b + CLF_CONTEXT)
            seg = wav[int(ca * SR):int(cb * SR)]
            lab, prob, p_nb = self.classify_burst(seg)   # lab is None if no_burst gate fires
            ok, floor = self.dur_ok(lab, dur_span)       # duration gate on the span
            gated = (lab is not None) and (not ok)       # confirmed a class but span too short
            a_bursts.append({"start": round(a, 2), "end": round(b, 2), "dur": round(dur_span, 2),
                             "peak": round(pk, 3), "label": lab, "prob": round(prob, 3),
                             "p_noburst": round(p_nb, 3), "dur_floor": round(floor, 2),
                             "dur_gated": gated, "kept": (lab is not None) and ok})
            if self.x2 is not None and (X2_ALL_SPANS or ((lab is not None) and ok)):
                x2_cuts.append(seg)
                x2_ix.append(len(a_bursts) - 1)

        # --- the x2 second opinion over this clip's spans, in one batched pass ---
        for i, r in zip(x2_ix, self.classify_x2(x2_cuts)):
            a_bursts[i].update(self._x2_fields(r))
            a_bursts[i]["_x2"] = r
        for x in a_bursts:
            r = x.pop("_x2", None)
            m = self.merge_burst_label(x["label"], r)
            # `written` is the text that goes into the script; `vocab` says which vocabulary
            # earned the right to say it (v2-83 / x2-17 / x2-17-family / generic).
            x["written"] = m["written"]
            x["vocab"] = m["vocab"]
            x["tier"] = m["tier"]
            x["label_source"] = m["source"]
            if m.get("out_of_x2_vocab") is not None:
                x["out_of_x2_vocab"] = m["out_of_x2_vocab"]
            if m.get("agrees_with_v2") is not None:
                x["x2_agrees"] = m["agrees_with_v2"]
            # Opt-in second gate. Off by default; when on, the rejection is recorded, not silent.
            if X2_GATE and x["kept"] and x.get("x2_p_noburst", 0.0) >= X2_GATE:
                x["kept"] = False
                x["x2_gated"] = True

        kept = [(x["start"], x["end"], x["written"], x["prob"]) for x in a_bursts if x["kept"]]
        variant_a_inline = self.variant_a(words, kept)
        # default procedural output: kept bursts mapped inline into per-sentence script lines
        script_lines = self._assign_bursts(sent_out, words, [x for x in a_bursts if x["kept"]])

        if mp3_out:
            write_mp3(path, mp3_out, os.environ.get("BC_MP3_BITRATE", "128k"))

        return {
            "id": cid, "dur": round(dur, 2), "transcript": transcript,
            "genu": round(g_vn["genu"], 2), "blend": round(g_vn["blend"], 2),
            "ei_gender": (round(ei_gender, 3) if ei_gender is not None else None),
            "gender_gated": (ei_gender is not None and abs(ei_gender) < EI_GENDER_GATE),
            # raw global scores for on-the-fly caption augmentation (see augment.py)
            "scores": {"dims": g_vn["dims"], "emo": emo_scores[0],
                       "genu": round(g_vn["genu"], 4), "blend": round(g_vn["blend"], 4)},
            "global_caption": g_text,
            "sentences": sent_out,
            "script_lines": script_lines,
            "variant_a_bursts": a_bursts,
            "variant_a_inline": variant_a_inline,
            "n_spans": len(spans), "n_words_ts": len([w for w in words if w.get("start") is not None]),
            "n_locator_windows": n_windows,
            "locator": f"{LOCATOR_REPO}/{LOCATOR_FILE}",
            "classifier": f"{BURST_CLF_REPO}/{BURST_CLF_FILE}",
            "x2": self.x2_meta(),
            "locator_postproc": {"threshold": LOCATOR_THR, "merge_gap": MERGE_GAP,
                                 "min_duration": MIN_BURST_DUR, "no_burst_gate": NOBURST_GATE,
                                 "chunk_sec": CHUNK_SEC, "chunk_overlap": CHUNK_OVERLAP},
            "pause": {"threshold": PAUSE_THR, "word_end": WORD_END,
                      "silence_floor": SILENCE_FLOOR},
            "emonet": self.use_emonet,
        }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Procedural voice captions with inserted vocal bursts.")
    ap.add_argument("audio", nargs="+", help="audio file(s) or glob(s)")
    ap.add_argument("--out", default="burst_caption_results.json")
    ap.add_argument("--mp3-dir", default=None, help="also write 120kbps mono mp3s here")
    ap.add_argument("--no-emonet", action="store_true")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--x2", action="store_true",
                    help="also run laion/vocal-burst-detector-x2 (17 classes) on every span")
    ap.add_argument("--x2-head", default=None, choices=["large-v2", "commercial"],
                    help="x2 encoder: large-v2 (default, best) or commercial (free here)")
    ap.add_argument("--label-source", default=None, choices=["v2", "x2", "union"],
                    help="which vocabulary writes the caption (default v2 = unchanged)")
    A = ap.parse_args()
    paths = []
    for a in A.audio:
        paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    bc = BurstCaptioner(device=A.device, use_emonet=not A.no_emonet,
                        use_x2=(True if (A.x2 or A.x2_head or
                                         (A.label_source and A.label_source != "v2")) else None),
                        x2_head=A.x2_head, label_source=A.label_source)
    results = []
    for i, p in enumerate(paths):
        cid = os.path.splitext(os.path.basename(p))[0]
        mp3 = os.path.join(A.mp3_dir, cid + ".mp3") if A.mp3_dir else None
        try:
            r = bc.process(p, cid=cid, mp3_out=mp3)
        except Exception as e:
            r = {"id": cid, "error": repr(e)}
        results.append(r)
        print(f"  [{i+1}/{len(paths)}] {cid}: {r.get('global_caption','ERR')[:90]}", flush=True)
    json.dump({"n": len(results), "results": results}, open(A.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", A.out)


if __name__ == "__main__":
    main()
