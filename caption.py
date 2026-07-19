#!/usr/bin/env python3
"""Procedural voice captioning from model predictions + baseline statistics.

Given one clip's raw predictions (57 VoiceNet dimensions + genuineness + vocal-
burst blend + 40 EmoNet emotions) and `baseline_stats.json`, describe the voice
in plain English by how far it deviates from the average voice.

Pipeline
--------
1. z-score every dimension:  z = (value - median) / spread,
   spread = 1.4826*MAD  (robust std; falls back to std when MAD == 0).
2. Pick the top-k VoiceNet dimensions by |z| (default k=5) and the top-k EmoNet
   emotions by |z| (default k=3); both k are configurable.
3. ALWAYS include Age (AGEV), Gender (GEND), Register (REGS) and Tempo (TEMP),
   even when they are close to the baseline.
4. Emotion wording rotates through each emotion's synonym cluster for variety.
5. Genuineness gate: extreme emotion wording is only allowed when genuineness is
   at/above its baseline median; below -> intensity is capped; well below ->
   emotion phrases are dropped.  A graded genuineness descriptor is emitted
   (deeply/genuinely -> genuine -> slightly genuine -> measured/performed).
6. Each phrase carries a direction (above/below baseline) and an intensity from
   |z| (Extremely / Very / Notably / Somewhat).

Usage
-----
    from caption import caption, caption_detail, load_baseline
    base = load_baseline()                       # bundled baseline_stats.json
    text = caption(preds, base, k_voicenet=5, k_emonet=3)

    # CLI:
    python caption.py preds.json                 # JSON: {"dims":{...},"emo":{...},
                                                 #        "genu":x,"blend":y}
    python caption.py - --kv 5 --ke 3            # read JSON from stdin
"""
import os, sys, json, math, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(HERE, "baseline_stats.json")
EPS = 1e-6

ALWAYS_ON = ["AGEV", "GEND", "REGS", "TEMP"]        # always described

# |z| -> intensity adverb
INTENSITY = [(2.0, "Extremely"), (1.5, "Very"), (1.0, "Notably"), (0.5, "Somewhat")]
NEUTRAL_BAND = 0.5   # |z| below this = "about average"

# Curated high-/low-end adjective phrases per VoiceNet dimension (from the rubric).
DESC = {
    "AGEV": {"hi": "elderly-sounding",                 "lo": "young and childlike"},
    "AROU": {"hi": "high-energy and animated",         "lo": "calm and low-energy"},
    "ARSH": {"hi": "surging upward in energy",         "lo": "collapsing in energy"},
    "ATCK": {"hi": "sharp and hard in onset",          "lo": "soft and gentle in onset"},
    "BKGN": {"hi": "clean and quiet in background",     "lo": "noisy in the background"},
    "BRGT": {"hi": "bright and ringing",               "lo": "dark and muffled"},
    "CHNK": {"hi": "long-phrased and unbroken",        "lo": "choppy and fragmented"},
    "CLRT": {"hi": "crisply articulated",              "lo": "blurry and mumbled"},
    "COGL": {"hi": "effortful and strained",           "lo": "effortless and fluent"},
    "DARC": {"hi": "wide in dynamic swell",            "lo": "fading in loudness"},
    "DFLU": {"hi": "disfluent and halting",            "lo": "smooth and flawless"},
    "EMPH": {"hi": "strongly emphasized",              "lo": "flat and unemphasized"},
    "ESTH": {"hi": "beautiful and pleasing",           "lo": "unpleasant and harsh"},
    "EXPL": {"hi": "explicit in content",              "lo": "clean and safe in content"},
    "FOCS": {"hi": "intensely focused",                "lo": "dissociated and inward"},
    "FULL": {"hi": "full and cinematic",               "lo": "thin and paper-like"},
    "GEND": {"hi": "masculine and deep-pitched",       "lo": "feminine and high-pitched"},
    "HARM": {"hi": "pure and tonal",                   "lo": "noisy and aperiodic"},
    "METL": {"hi": "metallic and ringing",             "lo": "organic and soft"},
    "RANG": {"hi": "wide-ranging in pitch",            "lo": "flat and monotone"},
    "RCQL": {"hi": "pristine in recording quality",    "lo": "poor in recording quality"},
    "REGS": {"hi": "high in vocal register",           "lo": "low and bassy in register"},
    "RESP": {"hi": "audibly breathing and gasping",    "lo": "seamless and inaudible in breathing"},
    "ROUG": {"hi": "rough and raspy",                  "lo": "smooth and clean"},
    "R_CHST": {"hi": "rich in deep chest resonance",   "lo": "thin in chest resonance"},
    "R_HEAD": {"hi": "bright in head resonance",       "lo": "thin in head resonance"},
    "R_MASK": {"hi": "forward in mask resonance",      "lo": "thin in mask resonance"},
    "R_MIXD": {"hi": "balanced and blended in resonance", "lo": "one-dimensional in resonance"},
    "R_NASL": {"hi": "heavily nasal in resonance",     "lo": "denasal and clear"},
    "R_ORAL": {"hi": "bright in oral resonance",       "lo": "muted in oral resonance"},
    "R_THRT": {"hi": "deep in throat resonance",       "lo": "thin in throat resonance"},
    "SMTH": {"hi": "mechanically even in rhythm",      "lo": "jerky and uneven in rhythm"},
    "STNC": {"hi": "dominant and commanding",          "lo": "meek and submissive"},
    "STRU": {"hi": "clearly and logically structured", "lo": "scattered and disorganized"},
    "S_ASMR": {"hi": "intimate and ASMR-like",         "lo": "loud and non-intimate"},
    "S_AUTH": {"hi": "authoritative and commanding",   "lo": "submissive and unassertive"},
    "S_CART": {"hi": "cartoonish and exaggerated",     "lo": "naturalistic and understated"},
    "S_CASU": {"hi": "casual and informal",            "lo": "formal and scripted"},
    "S_CONV": {"hi": "conversational and interactive", "lo": "monologue-like"},
    "S_DRAM": {"hi": "dramatic and theatrical",        "lo": "understated and flat"},
    "S_FORM": {"hi": "formal and rigid",               "lo": "casual and loose"},
    "S_MONO": {"hi": "introspective and monologic",    "lo": "dialogic and outward"},
    "S_NARR": {"hi": "narrator-like and storybook",    "lo": "non-narrative in style"},
    "S_NEWS": {"hi": "news-anchor-like",               "lo": "non-broadcast in style"},
    "S_PLAY": {"hi": "playful and humorous",           "lo": "serious and humorless"},
    "S_RANT": {"hi": "ranting and angry",              "lo": "calm and even-tempered"},
    "S_STRY": {"hi": "storytelling in style",          "lo": "non-narrative in style"},
    "S_TECH": {"hi": "didactic and teacherly",         "lo": "non-instructional in style"},
    "S_WHIS": {"hi": "breathy and whispered",          "lo": "loud and non-whispered"},
    "TEMP": {"hi": "fast and rapid in tempo",          "lo": "slow and deliberate in tempo"},
    "TENS": {"hi": "tight and tense",                  "lo": "relaxed and loose"},
    "VALN": {"hi": "joyful and positive",              "lo": "distressed and negative"},
    "VALS": {"hi": "brightening in mood",              "lo": "darkening in mood"},
    "VFLX": {"hi": "accelerating in pace",             "lo": "decelerating in pace"},
    "VOLT": {"hi": "emotionally volatile",             "lo": "emotionally steady and static"},
    "VULN": {"hi": "raw and vulnerable",               "lo": "guarded and armored"},
    "WARM": {"hi": "warm and enveloping",              "lo": "cold and sterile"},
}
# Neutral ("about average") descriptor per always-on dimension.
NEUTRAL = {"AGEV": "middle-aged in voice", "GEND": "gender-neutral in pitch",
           "REGS": "mid-register", "TEMP": "moderate in tempo"}


# --------------------------------------------------------------------------- #
def load_baseline(path=_BASE_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _spread(stat):
    """Robust scale for z-scoring. Prefer the precomputed 'spread' field; otherwise
    use 1.4826*MAD, falling back to std when the MAD collapses (< half the std, as
    for zero-inflated emotion scores)."""
    sp = stat.get("spread")
    if sp is not None and float(sp) > EPS:
        return float(sp)
    mad = float(stat.get("mad", 0.0) or 0.0)
    std = float(stat.get("std", 0.0) or 0.0)
    rmad = 1.4826 * mad
    sp = rmad if rmad >= 0.5 * std else std
    return sp if sp > EPS else 1.0


def _val(v):
    if isinstance(v, dict):
        for k in ("value", "reg_score", "reg", "r", "score"):
            if k in v:
                return float(v[k])
        raise ValueError(f"no scalar in {v}")
    return float(v)


def _intensity(az, cap=None):
    """|z| -> adverb, optionally capped at a maximum tier index."""
    order = INTENSITY
    if cap is not None:
        order = [t for t in INTENSITY if t[1] != "Extremely" and (cap != "Notably" or t[1] in ("Notably", "Somewhat"))]
    for thr, word in order:
        if az >= thr:
            return word
    return "Somewhat"


def _zscore(value, stat):
    return (value - float(stat["median"])) / (_spread(stat) + EPS)


# --------------------------------------------------------------------------- #
def _genuineness_gate(genu, baseline):
    """Return (gate, descriptor, zg). gate in {open, capped, dropped}."""
    st = baseline.get("genuineness")
    if genu is None or st is None:
        return "open", None, None
    g = _val(genu)
    zg = _zscore(g, st)
    if g >= float(st["median"]):
        gate = "open"
    elif zg > -1.0:
        gate = "capped"
    else:
        gate = "dropped"
    if zg >= 0.5:
        desc = "deeply and genuinely felt"
    elif zg >= 0.0:
        desc = "genuine in delivery"
    elif zg > -1.0:
        desc = "only slightly genuine, somewhat performed"
    else:
        desc = "measured and performed rather than genuine"
    return gate, desc, zg


def _emotion_word(name, stat, seed):
    syns = list(stat.get("synonyms") or [])
    if not syns:
        return name.lower()
    rng = random.Random((seed if seed is not None else random.randrange(1 << 30)) ^ (hash(name) & 0xFFFFFFFF))
    return rng.choice(syns)


# --------------------------------------------------------------------------- #
def caption_detail(preds, baseline=None, k_voicenet=5, k_emonet=3,
                   always_on=ALWAYS_ON, synonym_seed=None):
    """Return a structured breakdown of the caption.

    `preds` accepts either a flat {code: value} mapping or a nested
    {"dims": {...}, "emo": {...}, "genu": x, "blend": y} dict.
    Returns {"voicenet": [...], "emotions": [...], "quality": [...],
             "genuineness_gate": {...}}.
    """
    if baseline is None:
        baseline = load_baseline()
    k_voicenet = max(0, int(k_voicenet)); k_emonet = max(0, int(k_emonet))

    # normalise prediction layout ------------------------------------------------
    dims, emo, genu, blend = {}, {}, None, None
    nested = isinstance(preds.get("dims"), dict) or isinstance(preds.get("emo"), dict)
    if nested:
        dims = dict(preds.get("dims", {}))
        emo = dict(preds.get("emo", {}))
        genu = preds.get("genu"); blend = preds.get("blend")
    else:
        for code, v in preds.items():
            if code in ("genuineness", "genu"):
                genu = v
            elif code in ("blend",):
                blend = v
            elif code in baseline and baseline[code].get("group") == "emonet":
                emo[code] = v
            else:
                dims[code] = v
    if genu is None and "genuineness" in dims:
        genu = dims.pop("genuineness")

    gate, gen_desc, zg = _genuineness_gate(genu, baseline)

    # ---- VoiceNet ----
    scored = []
    for code, v in dims.items():
        st = baseline.get(code)
        if not st or st.get("group") != "voicenet" or code not in DESC:
            continue
        try:
            val = _val(v)
        except Exception:
            continue
        scored.append((code, val, _zscore(val, st)))
    scored.sort(key=lambda t: abs(t[2]), reverse=True)

    chosen, seen = [], set()
    for code, val, z in scored[:k_voicenet]:
        seen.add(code); chosen.append((code, val, z, False))
    for code in always_on:
        if code in seen:
            for i, (c, val, z, _) in enumerate(chosen):
                if c == code:
                    chosen[i] = (c, val, z, True)
            continue
        for code2, val, z in scored:
            if code2 == code:
                chosen.append((code, val, z, True)); break

    vn = []
    for code, val, z, always in chosen:
        az = abs(z); direction = "above" if z >= 0 else "below"
        d = DESC[code]
        if az < NEUTRAL_BAND:
            phrase = NEUTRAL.get(code, f"average in {code.lower()}")
            adverb = "About-average"
        else:
            adverb = _intensity(az)
            phrase = f"{adverb.lower()} {d['hi' if z >= 0 else 'lo']}"
        vn.append({"dim": code, "name": baseline[code].get("name", code),
                   "value": round(val, 3), "z": round(z, 2), "direction": direction,
                   "intensity": adverb, "always_on": always, "phrase": phrase})
    # order: the top-k deviations first (already sorted by |z|), always-on extras appended

    # ---- EmoNet emotions (genuineness-gated) ----
    emos = []
    if gate != "dropped" and k_emonet > 0:
        es = []
        for name, v in emo.items():
            st = baseline.get(name)
            if not st or st.get("group") != "emonet":
                continue
            try:
                val = _val(v)
            except Exception:
                continue
            es.append((name, val, _zscore(val, st)))
        es.sort(key=lambda t: abs(t[2]), reverse=True)
        cap = "Notably" if gate == "capped" else None
        for name, val, z in es[:k_emonet]:
            az = abs(z)
            if az < NEUTRAL_BAND:
                continue
            word = _emotion_word(name, baseline[name], synonym_seed)
            adverb = _intensity(az, cap=cap)
            if z >= 0:
                phrase = f"{adverb.lower()} carrying {word}"
                direction = "above"
            else:
                phrase = f"notably free of {word}"
                direction = "below"
            emos.append({"emotion": name, "value": round(val, 3), "z": round(z, 2),
                         "direction": direction, "intensity": adverb,
                         "synonym": word, "phrase": phrase})

    # ---- quality (genuineness + blend) ----
    quality = []
    if gen_desc is not None:
        quality.append({"dim": "genuineness", "name": "Genuineness",
                        "value": round(_val(genu), 3), "z": round(zg, 2) if zg is not None else None,
                        "phrase": gen_desc})
    if blend is not None and "blend" in baseline:
        bz = _zscore(_val(blend), baseline["blend"])
        if abs(bz) >= NEUTRAL_BAND:
            ph = ("interwoven with vocal bursts (laughs, gasps, sighs)" if bz >= 0
                  else "clean of non-verbal vocal bursts")
            quality.append({"dim": "blend", "name": "Vocal-burst blend",
                            "value": round(_val(blend), 3), "z": round(bz, 2), "phrase": ph})

    return {"voicenet": vn, "emotions": emos, "quality": quality,
            "genuineness_gate": {"gate": gate, "descriptor": gen_desc, "z": zg}}


def caption(preds, baseline=None, k_voicenet=5, k_emonet=3,
            always_on=ALWAYS_ON, synonym_seed=None, as_text=True):
    """Return the caption as a single sentence (`as_text`) or a list of phrases."""
    d = caption_detail(preds, baseline, k_voicenet, k_emonet, always_on, synonym_seed)
    phrases = [e["phrase"] for e in d["voicenet"]]
    phrases += [e["phrase"] for e in d["emotions"]]
    phrases += [e["phrase"] for e in d["quality"]]
    if not as_text:
        return phrases
    if not phrases:
        return "An average, unremarkable voice."
    body = "; ".join(phrases)
    return "A voice that is " + body + "."


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Procedural voice caption from predictions.")
    ap.add_argument("preds", help="path to prediction JSON, or '-' for stdin")
    ap.add_argument("--kv", type=int, default=5, help="top-k VoiceNet dims (default 5)")
    ap.add_argument("--ke", type=int, default=3, help="top-k EmoNet emotions (default 3)")
    ap.add_argument("--seed", type=int, default=None, help="synonym-rotation seed")
    ap.add_argument("--baseline", default=_BASE_PATH)
    ap.add_argument("--json", action="store_true", help="print structured detail as JSON")
    A = ap.parse_args()
    raw = sys.stdin.read() if A.preds == "-" else open(A.preds).read()
    preds = json.loads(raw)
    base = load_baseline(A.baseline)
    detail = caption_detail(preds, base, A.kv, A.ke, synonym_seed=A.seed)
    if A.json:
        print(json.dumps(detail, indent=2, ensure_ascii=False))
    else:
        print(caption(preds, base, A.kv, A.ke, synonym_seed=A.seed))
        print()
        for e in detail["voicenet"]:
            tag = " *always*" if e["always_on"] else ""
            print(f"  VN  {e['dim']:6s} z={e['z']:+.2f}  {e['phrase']}{tag}")
        for e in detail["emotions"]:
            print(f"  EMO {e['emotion']:22s} z={e['z']:+.2f}  {e['phrase']}")
        for e in detail["quality"]:
            print(f"  Q   {e['dim']:12s}  {e['phrase']}")
