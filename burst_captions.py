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
    (``laion/vocalburst-locator``, a 50 fps per-frame burst-probability model)
    over the whole clip at threshold ``0.7``. Every detected span's audio is fed
    to the multi-label burst classifier (VoiceCLAP ``encode_waveform`` -> MLP,
    83 outputs = 82 taxonomy classes + ``no_burst``). If ``P(no_burst) < 0.5``
    the span's **top-1** class is inserted as its own ``(Class Name)`` inline at
    the span's time, positioned between the two ASR words nearest that moment.
  - **Variant B — "sentence-level".** Each sentence segment's audio is run
    through the same classifier; if ``P(no_burst) >= 0.5`` no burst is attached,
    otherwise the **top-1** class is woven into that sentence's caption with a
    small procedurally-generated phrase (``... punctuated by a (Gasp).``).

Scoring stack (all reused from the LAION voice stack):

* VoiceNet 57 dims + genuineness + vocal-burst blend — VoiceCLAP-commercial
  embedding -> best-per-dim MLP heads (``VOICENET_REPO``).
* EmoNet-40 — ``laion/BUD-E-Whisper`` encoder -> ``laion/Empathic-Insight-Voice-Plus``
  per-emotion heads. Optional (set ``BC_EMONET=0`` to skip; captions then fall
  back to VoiceNet + genuineness only).
* Parakeet TDT (``nvidia/parakeet-tdt-0.6b-v3``) — word + sentence timestamps.
* Burst locator + multi-label burst classifier (VoiceCLAP -> MLP 2048x4).

Everything is env-driven (see the ``ENV`` block). Import ``BurstCaptioner`` and
call :meth:`BurstCaptioner.process`, or run this file as a CLI over a set of
audio files to dump a per-clip results JSON.
"""
import os, sys, io, json, glob, math, subprocess, argparse

# --- threading discipline: this box thrashes without it (set BEFORE torch) ----
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/tmp/hf_cache"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from caption import caption_detail, load_baseline, render_caption, _groups  # noqa: E402

# --------------------------------------------------------------------------- #
# ENV — all model locations are overridable.
DEVICE          = os.environ.get("BC_DEVICE", "cuda:0")
VOICENET_REPO   = os.environ.get("VOICENET_REPO", "/run/user/1001/dim_heads/repo")
GENU_PT         = os.environ.get("GENU_PT", "/tmp/genu_pred/genu_commercial_best.pt")
BLEND_PT        = os.environ.get("BLEND_PT", "/tmp/vcblend_pkg/blend_head_commercial.pt")
BURST_MLP_PT    = os.environ.get("BURST_MLP_PT", "/run/user/1001/vb_dataset/model_mlp_multi_m_h2048d4_dp3.pt")
TAXONOMY_JSON   = os.environ.get("VOCALBURST_TAXONOMY", os.path.join(HERE, "vocalburst_taxonomy.json"))
PARAKEET_MODEL  = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
ASR_HELPERS     = os.environ.get("ASR_HELPERS_DIR", "/run/user/1001/asr_moss_analysis")
USE_EMONET      = os.environ.get("BC_EMONET", "1") != "0"
LOCATOR_THR     = float(os.environ.get("BURST_LOCATOR_THR", "0.7"))
NOBURST_GATE    = float(os.environ.get("BURST_NOBURST_GATE", "0.5"))
# Duration gate: very short locator spans are dominated by transient false positives.
# Empirically (character-voice grid) every rejected span was <0.6s, and the 0.10-0.16s
# cluster was almost entirely Slap Face / Lip Smack / Resonant Hum firing at high p even
# though nothing is really there. So we reject spans below a minimum duration, with a
# stricter floor for the transient "smack / click / slap" groups.
MIN_BURST_DUR     = float(os.environ.get("BURST_MIN_DUR", "0.30"))            # global floor (s)
TRANSIENT_MIN_DUR = float(os.environ.get("BURST_TRANSIENT_MIN_DUR", "0.60"))  # stricter for smack/click/slap
TRANSIENT_GROUPS  = {"mouth_and_lip_sounds", "tongue_clicks", "hand_and_body_sounds"}

SR = 16000
VN_TARGET = 480000          # 30 s @ 16 kHz for the VoiceNet embedding


# --------------------------------------------------------------------------- #
# Audio
def decode_16k(path):
    """Decode any audio file to a 1-D float32 16 kHz mono torch tensor."""
    import numpy as np, torch, soundfile as sf, torchaudio
    p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", str(SR), "-f", "wav", "pipe:1"],
                       capture_output=True)
    data, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=False)
    w = torch.from_numpy(np.ascontiguousarray(data))
    if w.dim() == 2:
        w = w.mean(1)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.float()


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
# Burst locator (WhisperSeg) + multi-label burst classifier (VoiceCLAP -> MLP)
def _whisper_seg():
    import torch.nn as nn
    from transformers import WhisperModel
    class WhisperSeg(nn.Module):
        def __init__(s):
            super().__init__(); s.whisper = WhisperModel.from_pretrained("openai/whisper-small")
            d = s.whisper.config.d_model; h = max(256, d // 2)
            s.proj = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1))
            s.temporal = nn.Sequential(nn.Conv1d(h, h, 7, padding=3), nn.GELU(), nn.Dropout(0.1))
            s.out = nn.Linear(h, 1)
        def forward(s, x):
            e = s.whisper.encoder(input_features=x).last_hidden_state
            h = s.proj(e).permute(0, 2, 1); h = s.temporal(h).permute(0, 2, 1)
            return s.out(h).squeeze(-1)
    return WhisperSeg()


def _burst_mlp(out):
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(s, out, d=768, h=2048, depth=4):
            super().__init__()
            L = [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)]
            for _ in range(depth - 1):
                L += [nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)]
            L += [nn.Linear(h, out)]; s.net = nn.Sequential(*L)
        def forward(s, x):
            return s.net(x)
    return MLP(out=out)


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

    def __init__(self, device=DEVICE, use_emonet=USE_EMONET, verbose=True):
        import torch
        torch.set_num_threads(2)                      # critical: box thrashes otherwise
        self.dev = device
        self.verbose = verbose
        self.baseline = load_baseline()
        self._log("loading VoiceCLAP-commercial embedder + VoiceNet heads ...")
        from transformers import AutoModel
        self.vc = AutoModel.from_pretrained(os.path.join(VOICENET_REPO, "voiceclap_commercial"),
                                            trust_remote_code=True).to(device).eval()
        self.reg = {}
        for p in sorted(glob.glob(os.path.join(VOICENET_REPO, "regression", "*.pt"))):
            self.reg[os.path.basename(p)[:-3]] = _load_head(p, device)
        self.dims = sorted(self.reg)
        self.genu = _load_head(GENU_PT, device)
        self.blend = _load_head(BLEND_PT, device)

        # burst taxonomy / classes
        tax = json.load(open(TAXONOMY_JSON))["categories"]
        self.classes = [n for g, d in tax.items() for n in d.get("items", {})] + ["no_burst"]
        self.label_group = {n: g for g, d in tax.items() for n in d.get("items", {})}
        self.NB = len(self.classes) - 1
        self._log(f"burst taxonomy: {self.NB} classes + no_burst")

        self._log("loading burst locator + multi-label classifier ...")
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        self.seg = _whisper_seg().to(device).eval()
        self.seg.load_state_dict(torch.load(hf_hub_download("laion/vocalburst-locator", "model.pt"),
                                            map_location="cpu"), strict=True)
        self.wfe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.clf = _burst_mlp(len(self.classes)).to(device).eval()
        self.clf.load_state_dict(torch.load(BURST_MLP_PT, map_location=device))

        self._log(f"loading Parakeet ({PARAKEET_MODEL}) ...")
        sys.path.insert(0, ASR_HELPERS)
        from transformers import AutoProcessor, ParakeetForTDT
        from run_parakeet import _tokens_to_words, _sentences_from_words
        from asr_common import split_sentences
        self._tokens_to_words = _tokens_to_words
        self._sentences_from_words = _sentences_from_words
        self._split_sentences = split_sentences
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
        sents = self._sentences_from_words(text, words) if words else \
            [dict(text=s, start=None, end=None) for s in self._split_sentences(text)]
        return text, words, sents

    def locator_spans(self, wav):
        """Burst locator @ threshold. Returns [(start_s, end_s, peak_prob), ...]."""
        import torch
        f = self.wfe(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(self.dev)
        with torch.no_grad():
            pr = torch.sigmoid(self.seg(f))[0].float().cpu().numpy()
        n = min(len(pr), int(wav.shape[0] / SR * 50) + 1)
        pr = pr[:n]; out = []; i = 0
        while i < n:
            if pr[i] >= LOCATOR_THR:
                j = i
                while j + 1 < n and pr[j + 1] >= LOCATOR_THR:
                    j += 1
                if j - i >= 2:                          # >= ~60 ms
                    out.append((i / 50.0, (j + 1) / 50.0, float(pr[i:j + 1].max())))
                i = j + 1
            else:
                i += 1
        return out

    def classify_burst(self, seg_wav):
        """VoiceCLAP -> MLP -> sigmoid. Returns (label_or_None, prob, p_noburst)."""
        import torch
        if seg_wav.shape[0] < int(0.1 * SR):
            return None, 0.0, 1.0
        with torch.no_grad():
            e = self.vc.encode_waveform(seg_wav.to(self.dev))
            p = torch.sigmoid(self.clf(e)).squeeze(0).cpu().numpy()
        p_nb = float(p[self.NB])
        if p_nb >= NOBURST_GATE:
            return None, p_nb, p_nb
        top = int(p[:self.NB].argmax())                 # top-1 over the 82 burst classes
        return self.classes[top], float(p[top]), p_nb

    # -- caption composition -------------------------------------------------- #
    def global_caption(self, preds, seed=0):
        """Top-5 VoiceNet + Top-3 emotions + genuineness + Age/Gender/Tempo."""
        d = caption_detail(preds, self.baseline, k_voicenet=5, k_emonet=3,
                           synonym_seed=seed, template="default")
        return render_caption(d), d

    def sentence_caption(self, preds, seed=0):
        """Top-3 VoiceNet + Top-3 emotions ONLY (no Age/Gender/Tempo, no quality)."""
        d = caption_detail(preds, self.baseline, k_voicenet=3, k_emonet=3,
                           always_on=[], synonym_seed=seed, template="default")
        vn = [e["phrase"] for e in d["voicenet"]]
        emo = [e["phrase"] for e in d["emotions"]]
        parts = vn + emo
        if not parts:
            txt = "An even, unremarkable delivery."
        else:
            txt = "Delivered " + "; ".join(parts) + "."
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
        for idx in range(n + 1):
            # flush any burst whose midpoint falls before the next word's start
            nxt_start = words_ts[idx]["start"] if idx < n else float("inf")
            while bi < len(bl):
                a, b, lab, pr = bl[bi]
                if (a + b) / 2.0 <= nxt_start:
                    toks.append(f"({lab})")
                    bi += 1
                else:
                    break
            if idx < n:
                toks.append(words_ts[idx]["w"])
        return " ".join(toks).strip()

    def dur_ok(self, label, dur):
        """Duration gate. Short locator spans are dominated by transient false positives, so a
        span must clear a minimum duration to be kept — a stricter floor for the smack/click/slap
        groups. Returns (ok, floor_used)."""
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
        for i, s in enumerate(sent_out):
            a, b = s.get("start"), s.get("end")
            sw = [w for w in words if w.get("start") is not None and a is not None and b is not None
                  and a - 1e-6 <= w["start"] <= b + 1e-6]
            sb = sorted([(x["start"], x["end"], x["label"], x["prob"]) for x in buckets[i]], key=lambda z: z[0])
            if sw and sb:
                inline = self.variant_a(sw, sb)
            elif sb:
                inline = (s["text"].rstrip(".") + " " + " ".join(f"({l})" for _, _, l, _ in sb)).strip()
            else:
                inline = s["text"]
            lines.append({"cue": s["caption"], "text": inline, "bursts": [l for _, _, l, _ in sb]})
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
    def variant_b_phrase(label, seed=0):
        """A small procedurally-generated phrase attaching a burst to a sentence."""
        import random
        tpl = ["punctuated by a ({b})", "with an audible ({b})", "broken by a ({b})",
               "carrying a ({b})", "interrupted by a ({b})", "marked by a ({b})"]
        rng = random.Random((seed or 0) ^ (hash(label) & 0xFFFFFFFF))
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
        seed = abs(hash(cid)) % (1 << 30)

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
        g_text, g_detail = self.global_caption(g_preds, seed)

        # --- per-sentence scoring + captions ---
        sent_out = []
        for i, s in enumerate(sents):
            seg = units_wav[i + 1]
            s_vn = self.voicenet(seg)
            s_preds = {"dims": s_vn["dims"], "emo": emo_scores[i + 1],
                       "genu": s_vn["genu"], "blend": s_vn["blend"]}
            s_text, _ = self.sentence_caption(s_preds, seed ^ (i + 1))
            # Variant B burst for this sentence
            lab, prob, p_nb = self.classify_burst(seg)
            b_text = s_text
            if lab is not None:
                b_text = s_text.rstrip(".") + ", " + self.variant_b_phrase(lab, seed ^ (i + 1)) + "."
            sent_out.append({
                "text": s.get("text", ""), "start": s.get("start"), "end": s.get("end"),
                "caption": s_text,
                "variant_b_burst": lab, "variant_b_prob": round(prob, 3),
                "variant_b_caption": b_text,
            })

        # --- Variant A: locator over whole clip ---
        spans = self.locator_spans(wav)
        a_bursts = []
        for (a, b, pk) in spans:
            dur_span = b - a
            seg = wav[int(a * SR):int(b * SR)]
            lab, prob, p_nb = self.classify_burst(seg)   # lab is None if no_burst gate fires
            ok, floor = self.dur_ok(lab, dur_span)       # duration gate on the span
            gated = (lab is not None) and (not ok)       # confirmed a class but span too short
            a_bursts.append({"start": round(a, 2), "end": round(b, 2), "dur": round(dur_span, 2),
                             "peak": round(pk, 3), "label": lab, "prob": round(prob, 3),
                             "p_noburst": round(p_nb, 3), "dur_floor": round(floor, 2),
                             "dur_gated": gated, "kept": (lab is not None) and ok})
        kept = [(x["start"], x["end"], x["label"], x["prob"]) for x in a_bursts if x["kept"]]
        variant_a_inline = self.variant_a(words, kept)
        # default procedural output: kept bursts mapped inline into per-sentence script lines
        script_lines = self._assign_bursts(sent_out, words, [x for x in a_bursts if x["kept"]])

        if mp3_out:
            os.makedirs(os.path.dirname(mp3_out), exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                            "-ac", "1", "-b:a", "120k", mp3_out], capture_output=True)

        return {
            "id": cid, "dur": round(dur, 2), "transcript": transcript,
            "genu": round(g_vn["genu"], 2), "blend": round(g_vn["blend"], 2),
            "global_caption": g_text,
            "sentences": sent_out,
            "script_lines": script_lines,
            "variant_a_bursts": a_bursts,
            "variant_a_inline": variant_a_inline,
            "n_spans": len(spans), "n_words_ts": len([w for w in words if w.get("start") is not None]),
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
    A = ap.parse_args()
    paths = []
    for a in A.audio:
        paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    bc = BurstCaptioner(device=A.device, use_emonet=not A.no_emonet)
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
