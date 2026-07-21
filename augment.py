#!/usr/bin/env python3
"""On-the-fly caption augmentation from stored scores — *score once, caption many times*.

Run the four scoring models (VoiceNet / EmoNet / genuineness / vocal-burst blend) ONCE per clip and
store only the **raw scores** — per sentence and for the whole clip — plus the ASR sentences and the
confirmed vocal-burst spans. Then, during training, regenerate a *different* procedural caption every
epoch straight from those scores: different surface template, different synonym rotation, optional
dim/emotion shuffle, terse tags vs. prose, gender gated or not. No audio and no model calls at train
time — it is a few microseconds of string formatting.

Why: the target voice description then varies in wording/structure across epochs while the *content*
(which dims/emotions are salient, the bursts, the transcript) stays fixed, so a downstream TTS/caption
model learns the score->text mapping instead of memorizing one phrasing. It also lets you switch the
whole dataset between terse-`tags` and prose targets for free.

Producing the score record
---------------------------
    from burst_captions import BurstCaptioner
    cap = BurstCaptioner()
    r = cap.process("clip.wav")
    record = score_record(r)          # small JSON-serializable dict; store this, drop the audio
    json.dump(record, open("clip.scores.json", "w"))

Augmenting at train time (per example, per epoch)
-------------------------------------------------
    from augment import load_record, augment_script
    rec = load_record("clip.scores.json")
    text = augment_script(rec, seed=epoch * 1_000_003 + example_id)   # fresh phrasing each epoch
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caption import (caption_detail, render_caption, load_baseline, TEMPLATE_NAMES, ALWAYS_ON,
                     EI_GENDER_GATE)

_BASE = None
def _baseline():
    global _BASE
    if _BASE is None:
        _BASE = load_baseline()
    return _BASE


def score_record(result):
    """Distill a BurstCaptioner.process() result into the minimal record needed to re-caption:
    global + per-sentence raw scores, the sentence texts/timings, kept burst labels per sentence,
    and the EI gender value (for the gender gate). Audio is not needed after this."""
    kept = [x for x in result.get("variant_a_bursts", []) if x.get("kept")]
    sents = []
    for s in result.get("sentences", []):
        a, b = s.get("start"), s.get("end")
        sb = [x["label"] for x in kept
              if a is not None and b is not None and a - 0.2 <= (x["start"] + x["end"]) / 2 <= b + 0.2]
        sents.append({"text": s.get("text", ""), "start": a, "end": b,
                      "scores": s.get("scores", {}), "bursts": sb})
    return {"id": result.get("id"), "ei_gender": result.get("ei_gender"),
            "global_scores": result.get("scores", {}), "sentences": sents}


def load_record(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def augment_global(rec, seed, k_voicenet=5, k_emonet=3, template=None, gate_gender=True):
    """Regenerate the GENERAL line from stored global scores with a (seeded) random template."""
    rng = random.Random(seed)
    tmpl = template or TEMPLATE_NAMES[rng.randrange(len(TEMPLATE_NAMES))]
    d = caption_detail(rec["global_scores"], _baseline(), k_voicenet=k_voicenet, k_emonet=k_emonet,
                       always_on=ALWAYS_ON, synonym_seed=rng.randrange(1 << 30), template=tmpl,
                       shuffle_dims=rng.random() < 0.5,
                       ei_gender=(rec.get("ei_gender") if gate_gender else None))
    return render_caption(d)


def augment_sentence(sent, seed, k_voicenet=3, k_emonet=3, template=None):
    """Regenerate one sentence's style cue from its stored scores, no Age/Gender/Tempo."""
    rng = random.Random(seed)
    tmpl = template or TEMPLATE_NAMES[rng.randrange(len(TEMPLATE_NAMES))]
    d = caption_detail(sent.get("scores", {}), _baseline(), k_voicenet=k_voicenet, k_emonet=k_emonet,
                       always_on=[], synonym_seed=rng.randrange(1 << 30), template=tmpl,
                       shuffle_dims=rng.random() < 0.5)
    if tmpl == "tags":
        tags = ([e["tag"] for e in d["voicenet"] if e.get("tag")]
                + [e["tag"] for e in d["emotions"] if e.get("tag")])
        return ", ".join(tags) if tags else "unremarkable"
    parts = [e["phrase"] for e in d["voicenet"]] + [e["phrase"] for e in d["emotions"]]
    return ("Delivered " + "; ".join(parts) + ".") if parts else "An even, unremarkable delivery."


def augment_script(rec, seed, template=None, gate_gender=True):
    """Full procedural caption (GENERAL + per-sentence SCRIPT with inline bursts) regenerated from
    the stored scores. Pass a fixed `template` to force one surface form, or leave None for a
    per-call random pick (recommended for training-time variety)."""
    out = ["GENERAL: " + augment_global(rec, seed, template=template, gate_gender=gate_gender), "SCRIPT:"]
    for i, s in enumerate(rec.get("sentences", [])):
        cue = augment_sentence(s, seed ^ (i + 1) * 0x9E3779B1, template=template)
        text = s.get("text", "")
        if s.get("bursts"):
            text = (text.rstrip(".") + " " + " ".join(f"({b})" for b in s["bursts"])).strip()
        out.append(f"({cue}) {text}")
    return "\n".join(out)


if __name__ == "__main__":
    # demo: regenerate 4 varied captions from one stored record
    rec = load_record(sys.argv[1]) if len(sys.argv) > 1 else {
        "ei_gender": 1.4, "global_scores": {"dims": {"WARM": 6.0, "GEND": 4.6, "AGEV": 4.5, "REGS": 2.0,
        "TEMP": 1.5, "S_WHIS": 5.5, "RESP": 5.0, "STNC": 0.5}, "emo": {"Sadness": 6.0, "Sexual Lust": 5.0},
        "genu": 2.0, "blend": 7.0}, "sentences": [{"text": "Just relax now.", "start": 0.0, "end": 1.5,
        "scores": {"dims": {"WARM": 6.0, "S_WHIS": 5.0}, "emo": {"Contentment": 5.0}, "genu": 2.0, "blend": 3.0},
        "bursts": ["Soft Sigh"]}]}
    for ep in range(4):
        print(f"--- epoch {ep} ---\n{augment_script(rec, seed=ep * 1_000_003)}\n")
