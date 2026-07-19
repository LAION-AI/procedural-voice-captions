#!/usr/bin/env python3
"""Build baseline distribution statistics for procedural voice captioning.

For every scoring dimension used by the LAION voice-acting stack we estimate the
distribution over "all kinds of speech" (mean / median / std / MAD / p10 / p90 /
n).  Those baselines let a captioner turn a single clip's raw predictions into
z-scores  z = (value - median) / spread  and describe the clip by how far it
deviates from the average voice.

Dimension groups
----------------
* 57 VoiceNet dimensions  (regression heads on the 768-d VoiceCLAP-commercial
  embedding; this INCLUDES Valence `VALN` and Arousal `AROU`).
* genuineness 0-6 and vocal-burst-blend 0-10 (heads on the same embedding).
* 40 EmoNet emotions (BUD-E-Whisper encoder + per-emotion MLP heads).  We use
  EmoNet for its 40 EMOTIONS ONLY - Valence and Arousal are taken from VoiceNet,
  never double-counted from EmoNet.

Sources
-------
1. Emolia (~1000 random clips per language, languages en/de/zh/fr/ko/ja).
   VoiceNet/genu/blend use PRECOMPUTED VoiceCLAP-commercial embeddings; the 40
   EmoNet emotions are computed from a bounded random audio subset per language.
2. ~1000 random takes from laion/moss-character-voices-bestof64 (extreme
   character voices - deliberately widens the spread for VoiceNet/genu/blend).

Stages (run individually or `all`):
    python compute_baseline.py --stage emolia_vn   # CPU: heads on embeddings
    python compute_baseline.py --stage bestof64    # CPU: read precomputed scores
    python compute_baseline.py --stage emonet --gpu 0   # GPU: BUD-E-Whisper
    python compute_baseline.py --stage merge       # write baseline_stats.json
"""
import os, sys, json, glob, argparse, io, tarfile, time, random
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PC_WORK", HERE)
RAW_DIR = os.path.join(WORK, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SEED = 20260719
LANGS = ["en", "de", "zh", "fr", "ko", "ja"]

EMOLIA_INDEX = "/run/user/1001/emolia_index/emolia_index.parquet"
EMB_NPZ      = "/run/user/1001/dim_heads_v3/emb_all3.npz"
VN_REG_DIR   = "/run/user/1001/dim_heads/repo/regression"
GENU_PT      = "/tmp/genu_pred/genu_commercial_best.pt"
BLEND_PT     = "/tmp/vcblend_pkg/blend_head_commercial.pt"
EMONET_REPO  = "laion/Empathic-Insight-Voice-Plus"
EMOLIA_AUDIO = "VoiceNet/emolia-thinking"
TAX_DIR      = os.path.join(HERE, "voice-taxonomies")

# EmoNet 40 emotions -> synonym keywords (from LAION-AI/voice-taxonomies emonet)
def _ensure_taxonomies():
    p = os.path.join(TAX_DIR, "emonet", "emonet_taxonomy.json")
    if not os.path.exists(p):
        os.system(f"git clone --depth 1 https://github.com/LAION-AI/voice-taxonomies.git "
                  f"{TAX_DIR} >/dev/null 2>&1")
    return p

def load_emonet_taxonomy():
    p = _ensure_taxonomies()
    d = json.load(open(p))
    cats = d["categories"]
    return {name: {"synonyms": v.get("keywords", [])} for name, v in cats.items()}

EMO_OVR = {"Hope/Optimism": "model_Hope_Enthusiasm_Optimism_best.pth",
           "Intoxication/Altered States": "model_Intoxication_Altered_States_of_Consciousness_best.pth"}
def emo_file(e): return EMO_OVR.get(e, f"model_{e.replace('/','_').replace(' ','_')}_best.pth")


# ----------------------------------------------------------------------------
# Stage 1: Emolia VoiceNet / genu / blend on precomputed embeddings (CPU)
# ----------------------------------------------------------------------------
def stage_emolia_vn(n_per_lang=1000):
    import torch, torch.nn as nn
    import pandas as pd

    class MLPHead(nn.Module):
        def __init__(self, D, H, p, out):
            super().__init__()
            self.f1 = nn.Linear(D, H); self.act = nn.GELU()
            self.dp = nn.Dropout(p); self.f2 = nn.Linear(H, out)
        def forward(self, x): return self.f2(self.dp(self.act(self.f1(x))))

    def load_head(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["arch"]
        if isinstance(a, dict):
            D, H, p, out = a["D"], a["H"], a.get("p", a.get("dropout", 0.0)), a.get("out", 1)
        else:  # genu/blend store dims at top level
            D = ck.get("D", 768); H = ck["state_dict"]["f1.weight"].shape[0]
            p = ck.get("dropout", 0.0); out = 1
        net = MLPHead(D, H, p, out).eval()
        net.load_state_dict(ck["state_dict"])
        mu = torch.tensor(np.asarray(ck["mu"]), dtype=torch.float32)
        sd = torch.tensor(np.asarray(ck["sd"]), dtype=torch.float32)
        levels = ck.get("levels")
        k = len(levels) if levels else None
        return net, mu, sd, ck.get("name"), k

    print("[emolia_vn] loading embeddings ...", flush=True)
    z = np.load(EMB_NPZ, allow_pickle=True)
    emb_ids = z["ids"]; emb = z["emb"].astype(np.float32)
    id2row = {i: r for r, i in enumerate(emb_ids)}

    print("[emolia_vn] loading emolia index ...", flush=True)
    idx = pd.read_parquet(EMOLIA_INDEX, columns=["__emolia_id__", "language"])
    rng = random.Random(SEED)
    picks = {}   # lang -> list of (emolia_id, emb_row)
    for lang in LANGS:
        ids = idx.loc[idx.language == lang, "__emolia_id__"].tolist()
        avail = [i for i in ids if i in id2row]
        rng.shuffle(avail)
        sel = avail[:n_per_lang]
        picks[lang] = [(i, id2row[i]) for i in sel]
        print(f"[emolia_vn] {lang}: index={len(ids)} with_emb={len(avail)} picked={len(sel)}", flush=True)

    all_rows, lang_of = [], []
    for lang in LANGS:
        for _id, r in picks[lang]:
            all_rows.append(r); lang_of.append(lang)
    X = torch.tensor(emb[all_rows], dtype=torch.float32)   # (N,768)

    # load heads
    reg_heads = {}
    for p in sorted(glob.glob(os.path.join(VN_REG_DIR, "*.pt"))):
        dim = os.path.basename(p)[:-3]
        reg_heads[dim] = load_head(p)
    genu = load_head(GENU_PT)
    blend = load_head(BLEND_PT)
    print(f"[emolia_vn] loaded {len(reg_heads)} VN heads + genu + blend; scoring N={X.shape[0]}", flush=True)

    out = {"lang_of": lang_of, "names": {}, "k": {}, "values": {}, "n_per_lang": {l: len(picks[l]) for l in LANGS}}
    with torch.no_grad():
        for dim, (net, mu, sd, name, k) in reg_heads.items():
            v = net((X - mu) / sd).squeeze(-1).clamp(0, (k - 1) if k else 6)
            out["values"][dim] = v.numpy().tolist()
            out["names"][dim] = name or dim
            out["k"][dim] = k or 7
        net, mu, sd, _, _ = genu
        out["values"]["genuineness"] = net((X - mu) / sd).squeeze(-1).clamp(0, 6).numpy().tolist()
        net, mu, sd, _, _ = blend
        out["values"]["blend"] = net((X - mu) / sd).squeeze(-1).clamp(0, 10).numpy().tolist()

    json.dump(out, open(os.path.join(RAW_DIR, "emolia_vn.json"), "w"))
    print(f"[emolia_vn] wrote raw/emolia_vn.json  ({X.shape[0]} clips)", flush=True)


# ----------------------------------------------------------------------------
# Stage 2: bestof64 precomputed scores (CPU, no heads)
# ----------------------------------------------------------------------------
def stage_bestof64(n_takes=1000, n_files=None):
    import pandas as pd
    from huggingface_hub import hf_hub_download, HfApi
    # score files are grouped by character/shard, so sample ACROSS ALL of them to
    # cover many characters (and hence all 57 dims), not just one archetype.
    files = sorted(s.rfilename for s in HfApi().repo_info(
        "laion/moss-character-voices-bestof64", repo_type="dataset").siblings
        if s.rfilename.startswith("metadata/scores-") and s.rfilename.endswith(".parquet"))
    if n_files:
        files = files[:n_files]
    frames = []
    for f in files:
        p = hf_hub_download("laion/moss-character-voices-bestof64", f, repo_type="dataset")
        frames.append(pd.read_parquet(p, columns=["lang", "character", "dims_json", "emo_json", "genu", "blend"]))
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(n=min(n_takes, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"[bestof64] sampled {len(df)} takes from {len(files)} score files; "
          f"{df['character'].nunique()} characters", flush=True)

    vn_vals, genu_vals, blend_vals = {}, [], []
    for _, r in df.iterrows():
        try:
            dims = json.loads(r["dims_json"]) if isinstance(r["dims_json"], str) else (r["dims_json"] or {})
        except Exception:
            dims = {}
        for d, v in dims.items():
            if v is None: continue
            vn_vals.setdefault(d, []).append(float(v))
        if r["genu"] is not None: genu_vals.append(float(r["genu"]))
        if r["blend"] is not None: blend_vals.append(float(r["blend"]))
    out = {"n_takes": int(len(df)), "vn_values": vn_vals,
           "genuineness": genu_vals, "blend": blend_vals,
           "chars": sorted(df["character"].unique().tolist())}
    json.dump(out, open(os.path.join(RAW_DIR, "bestof64.json"), "w"))
    print(f"[bestof64] wrote raw/bestof64.json  (dims covered: {len(vn_vals)})", flush=True)


# ----------------------------------------------------------------------------
# Stage 3: EmoNet 40 emotions on bounded Emolia audio (ONE GPU, small batch)
# ----------------------------------------------------------------------------
def stage_emonet(gpu="0", per_shard=80, shards_per_lang=None, batch=8):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch, torch.nn as nn, torchaudio
    import pandas as pd
    from collections import OrderedDict
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from huggingface_hub import snapshot_download, hf_hub_download
    DEV = "cuda:0"
    SEQ_LEN, EMBED_DIM, PROJ = 1500, 768, 64
    HIDDEN, DROPS = [64, 32, 16], [0.0, 0.1, 0.1, 0.1]

    class FullEmbeddingMLP(nn.Module):
        def __init__(s, seq, emb, proj, hid, dr):
            super().__init__(); s.flatten = nn.Flatten(); s.proj = nn.Linear(seq * emb, proj)
            L = [nn.ReLU(), nn.Dropout(dr[0])]; cur = proj
            for i, h in enumerate(hid):
                L += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dr[i + 1])]; cur = h
            L.append(nn.Linear(cur, 1)); s.mlp = nn.Sequential(*L)
        def forward(s, x): return s.mlp(s.proj(s.flatten(x)))

    if shards_per_lang is None:
        shards_per_lang = {"en": 3, "de": 2, "zh": 2, "fr": 1, "ko": 1, "ja": 1}

    # choose shards + member keys per language
    print("[emonet] planning shards ...", flush=True)
    idx = pd.read_parquet(EMOLIA_INDEX, columns=["key", "shard", "__emolia_id__", "language"])
    rng = random.Random(SEED + 7)
    plan = []   # (lang, shard, [keys])
    for lang in LANGS:
        sub = idx[idx.language == lang]
        shs = sorted(sub["shard"].unique().tolist())
        rng.shuffle(shs)
        for sh in shs[:shards_per_lang.get(lang, 1)]:
            keys = sub.loc[sub.shard == sh, "key"].tolist()
            rng.shuffle(keys)
            plan.append((lang, sh, keys[:per_shard]))
    print(f"[emonet] {len(plan)} shards planned; ~{sum(len(k) for _,_,k in plan)} clips", flush=True)

    # load EmoNet encoder + heads
    EMOTIONS = list(load_emonet_taxonomy().keys())
    wm = WhisperForConditionalGeneration.from_pretrained(
        "laion/BUD-E-Whisper", torch_dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    wproc = AutoProcessor.from_pretrained("laion/BUD-E-Whisper")
    md = snapshot_download(EMONET_REPO, ignore_patterns=["*.mp3", "*.md", ".gitattributes"])
    avail = {os.path.basename(p): p for p in glob.glob(md + "/**/*.pth", recursive=True)}
    heads = {}
    for e in EMOTIONS:
        fn = emo_file(e)
        if fn not in avail:
            print(f"[emonet] MISSING head {fn} for {e}", flush=True); continue
        m = FullEmbeddingMLP(SEQ_LEN, EMBED_DIM, PROJ, HIDDEN, DROPS).to(DEV)
        sd = torch.load(avail[fn], map_location=DEV)
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in sd.items())
        m.load_state_dict(sd); heads[e] = m.eval().half()
    print(f"[emonet] loaded {len(heads)}/40 emotion heads", flush=True)

    tmpdir = os.path.join(WORK, "emonet_tmp"); os.makedirs(tmpdir, exist_ok=True)
    values = {e: [] for e in heads}
    lang_of, n_clips = [], 0

    @torch.no_grad()
    def encode_batch(wavs16):
        inp = wproc([w.numpy() for w in wavs16], sampling_rate=16000, return_tensors="pt",
                    padding="max_length", truncation=True)
        feats = inp.input_features.to(DEV).to(wm.dtype)
        enc = wm.get_encoder()(feats, return_dict=True).last_hidden_state.half()  # (B,1500,768)
        return enc

    for lang, sh, keys in plan:
        tarpath = os.path.join(tmpdir, sh)
        try:
            hf_hub_download(EMOLIA_AUDIO, sh, repo_type="dataset", local_dir=tmpdir)
        except Exception as ex:
            print(f"[emonet] download FAIL {sh}: {ex}", flush=True); continue
        keyset = set(keys); got = []
        try:
            with tarfile.open(tarpath) as tf:
                for m in tf:
                    if not m.name.endswith(".flac"): continue
                    k = m.name[:-5]
                    if k not in keyset: continue
                    try:
                        raw = tf.extractfile(m).read()
                        w, sr = torchaudio.load(io.BytesIO(raw))
                        if w.dim() == 2: w = w.mean(0)
                        if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
                        got.append(w.float())
                    except Exception:
                        continue
        finally:
            try: os.remove(tarpath)
            except OSError: pass
        # score in small batches
        for b in range(0, len(got), batch):
            chunk = got[b:b + batch]
            enc = encode_batch(chunk)
            with torch.no_grad():
                for e, h in heads.items():
                    vv = h(enc).squeeze(-1).float().cpu().numpy()
                    values[e].extend([float(x) for x in np.atleast_1d(vv)])
            lang_of.extend([lang] * len(chunk)); n_clips += len(chunk)
        print(f"[emonet] {lang} {sh}: scored {len(got)} clips (total {n_clips})", flush=True)

    out = {"n_clips": n_clips, "lang_of": lang_of, "values": values,
           "shards": [[l, s, len(k)] for l, s, k in plan]}
    json.dump(out, open(os.path.join(RAW_DIR, "emonet.json"), "w"))
    print(f"[emonet] wrote raw/emonet.json  ({n_clips} clips, {len(heads)} emotions)", flush=True)


# ----------------------------------------------------------------------------
# Stage 4: merge -> baseline_stats.json
# ----------------------------------------------------------------------------
def _stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    std = float(a.std())
    # Robust spread for z-scoring: 1.4826*MAD (consistent std estimate), but if the
    # MAD collapses (< half the std, e.g. for zero-inflated emotion scores where most
    # clips sit at ~0) fall back to std so z-scores don't explode.
    rmad = 1.4826 * mad
    spread = rmad if rmad >= 0.5 * std else std
    if spread < 1e-6:
        spread = std if std > 1e-6 else 1.0
    return {"mean": round(float(a.mean()), 4), "median": round(med, 4),
            "std": round(std, 4), "mad": round(mad, 4), "spread": round(spread, 4),
            "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "n": int(a.size)}


def stage_merge(emonet_n_note=None):
    tax = load_emonet_taxonomy()
    ev = json.load(open(os.path.join(RAW_DIR, "emolia_vn.json")))
    try:
        bo = json.load(open(os.path.join(RAW_DIR, "bestof64.json")))
    except FileNotFoundError:
        bo = None
    try:
        em = json.load(open(os.path.join(RAW_DIR, "emonet.json")))
    except FileNotFoundError:
        em = None

    stats = {}
    src_n = {"emolia_vn": len(ev["lang_of"])}

    # --- VoiceNet 57 dims ---
    for dim, vals in ev["values"].items():
        if dim in ("genuineness", "blend"):
            continue
        combined = list(vals)
        n_bo = 0
        if bo and dim in bo["vn_values"]:
            combined += bo["vn_values"][dim]; n_bo = len(bo["vn_values"][dim])
        s = _stats(combined)
        s["name"] = ev["names"].get(dim, dim)
        s["group"] = "voicenet"
        s["range"] = [0, ev["k"].get(dim, 7) - 1]
        s["n_emolia"] = len(vals); s["n_bestof64"] = n_bo
        stats[dim] = s

    # --- genuineness + blend (quality) ---
    for key, rng in (("genuineness", 6), ("blend", 10)):
        combined = list(ev["values"][key])
        n_bo = 0
        if bo and bo.get(key):
            combined += bo[key]; n_bo = len(bo[key])
        s = _stats(combined)
        s["name"] = "Genuineness" if key == "genuineness" else "Vocal-burst blend"
        s["group"] = "quality"
        s["range"] = [0, rng]
        s["n_emolia"] = len(ev["values"][key]); s["n_bestof64"] = n_bo
        stats[key] = s

    # --- EmoNet 40 emotions ---
    emo_n = 0
    if em:
        emo_n = em["n_clips"]
        for e, vals in em["values"].items():
            s = _stats(vals)
            if s is None: continue
            s["name"] = e
            s["group"] = "emonet"
            s["synonyms"] = tax.get(e, {}).get("synonyms", [])
            stats[e] = s
    src_n["emonet_emolia_audio"] = emo_n
    if bo: src_n["bestof64"] = bo["n_takes"]

    meta = {
        "description": (
            "Baseline distribution statistics ('the average voice') for procedural "
            "voice captioning. For each scoring dimension we report mean/median/std/MAD/"
            "p10/p90 over a broad sample of speech. A captioner converts a clip's raw "
            "predictions to z = (value - median) / spread and describes the clip by its "
            "largest deviations. The precomputed 'spread' field per dimension is "
            "1.4826*MAD (robust std estimate); it falls back to std when the MAD "
            "collapses below half the std (zero-inflated emotion scores)."),
        "z_score_rule": ("z = (value - median) / spread ; spread = 1.4826*MAD, but falls "
                         "back to std when 1.4826*MAD < 0.5*std (zero-inflated dims). The "
                         "chosen value is stored per dimension as 'spread'."),
        "groups": {
            "voicenet": "57 VoiceNet dimensions incl. Valence (VALN) and Arousal (AROU), "
                        "regression heads on VoiceCLAP-commercial 768-d embeddings, range 0-6.",
            "quality": "genuineness (0-6) and vocal-burst blend (0-10) heads on the same embedding.",
            "emonet": "40 EmoNet emotions, BUD-E-Whisper encoder + per-emotion MLP heads."},
        "valence_arousal_rule": ("Valence and Arousal are taken from VoiceNet's VALN/AROU heads. "
                                 "EmoNet is used for its 40 emotions ONLY; EmoNet's own valence/"
                                 "arousal axes are NOT loaded and NOT double-counted."),
        "models": {
            "voicenet": "laion/voicenet-dimension-predictors-commercial",
            "genuineness": "laion/voiceclap-commercial-genuineness",
            "blend": "laion/voiceclap-commercial-vocalburst-blend",
            "emonet": "laion/Empathic-Insight-Voice-Plus",
            "embedder": "laion/voiceclap-commercial"},
        "sources": {
            "emolia": {"description": "~1000 random clips per language; VoiceNet/genu/blend from "
                       "precomputed VoiceCLAP-commercial embeddings, EmoNet from a bounded random "
                       "audio subset.", "languages": ev["n_per_lang"],
                       "n_voicenet": len(ev["lang_of"]),
                       "n_emonet_audio": emo_n,
                       "emonet_shards": (em["shards"] if em else None)},
            "bestof64": {"description": "~1000 random takes from laion/moss-character-voices-bestof64 "
                         "(extreme character voices; deliberately widens the VoiceNet/genu/blend "
                         "spread).", "n_takes": (bo["n_takes"] if bo else 0),
                         "characters": (bo["chars"] if bo else [])}},
        "notes": ["Date-agnostic: baselines describe relative deviations, not absolute correctness.",
                  "EmoNet emotions come only from the Emolia audio subset (bestof64 emo_json covers "
                  "just per-character emotions, so it is not used for the 40-emotion baseline)."],
        "seed": SEED,
        "source_n": src_n,
    }
    if emonet_n_note:
        meta["notes"].append(emonet_n_note)

    doc = {"_meta": meta}
    doc.update(dict(sorted(stats.items())))
    outp = os.path.join(HERE, "baseline_stats.json")
    json.dump(doc, open(outp, "w"), indent=2, ensure_ascii=False)
    print(f"[merge] wrote {outp}  ({len(stats)} dimensions; emonet n={emo_n})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["emolia_vn", "bestof64", "emonet", "merge", "all"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n_per_lang", type=int, default=1000)
    ap.add_argument("--per_shard", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    A = ap.parse_args()
    if A.stage in ("emolia_vn", "all"): stage_emolia_vn(A.n_per_lang)
    if A.stage in ("bestof64", "all"): stage_bestof64()
    if A.stage in ("emonet", "all"): stage_emonet(A.gpu, A.per_shard, batch=A.batch)
    if A.stage in ("merge", "all"): stage_merge()
