#!/usr/bin/env python3
"""On-the-fly caption augmentation from stored scores — *score once, caption many times*.

Run Humaneness Ears Medium once per clip (the legacy model stack remains optional) and
store the predictions — per sentence and for the whole clip — plus the ASR sentences and
predicted vocal-burst intervals. Then, during training, regenerate a *different* procedural caption every
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

_BASE = {}
def _baseline(selector="dramabox"):
    global _BASE
    if selector not in _BASE:
        _BASE[selector] = load_baseline(selector)
    return _BASE[selector]


def score_record(result):
    """Distill a BurstCaptioner.process() result into the minimal record needed to re-caption:
    global + per-sentence raw scores, the sentence texts/timings, kept burst labels per sentence,
    and the EI gender value (for the gender gate). Audio is not needed after this.

    Each kept burst is stored as a small dict rather than a bare string, because with the x2
    head on, *what a caption may say* and *which class was predicted* are no longer the same
    thing: `written` is the recall-tiered text (`Chuckle`, `Breathy Giggle?`, `Breath`,
    `Vocal Burst`), `label` the raw class, and `vocab` the vocabulary that earned the right to
    say it. Records written before this change hold plain strings and still load — see
    `_burst_text`."""
    kept = [x for x in result.get("variant_a_bursts", []) if x.get("kept")]
    sents = []
    for s in result.get("sentences", []):
        a, b = s.get("start"), s.get("end")
        sb = [{"written": x.get("written") or x.get("label"), "label": x.get("label"),
               **({"start": x["start"], "end": x["end"], "prob": x.get("prob"),
                   "top3_classes": x.get("top3_classes"), "onset_probability": x.get("onset_probability")}
                  if result.get("score_backend") == "humaneness-medium" else {}),
               "vocab": x.get("vocab", "v2-83"), "tier": x.get("tier", "named")}
              for x in kept
              if a is not None and b is not None and a - 0.2 <= (x["start"] + x["end"]) / 2 <= b + 0.2]
        sents.append({"text": s.get("text", ""), "start": a, "end": b,
                      "scores": s.get("scores", {}), "bursts": sb})
        if "humaneness_ears_medium" in s:
            sents[-1]["humaneness_ears_medium"] = s["humaneness_ears_medium"]
    rec = {"id": result.get("id"), "ei_gender": result.get("ei_gender"),
           "global_scores": result.get("scores", {}), "sentences": sents}
    if result.get("x2"):                      # which naming head produced these labels
        rec["x2"] = result["x2"]
    if result.get("score_backend") == "humaneness-medium":
        rec.update(score_backend="humaneness-medium", baseline="humaneness-medium",
            humaneness_ears_medium=result["humaneness_ears_medium"],
            normalization_provenance=result.get("normalization_provenance"),
            caption_settings=result.get("caption_settings"),
            burst_events=result.get("variant_a_bursts", []),
            words=result.get("words", []), asr=result.get("asr"), dur=result.get("dur"),
            prompt_format_id=result.get("prompt_format_id"))
    else:
        rec["baseline"] = "dramabox"
    return rec


def _burst_text(b):
    """A stored burst is either a bare label (records written before the x2 head existed) or a
    dict carrying the recall-tiered `written` form. Both are accepted, forever."""
    if isinstance(b, dict):
        return b.get("written") or b.get("label") or ""
    return b


def load_record(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def augment_global(rec, seed, k_voicenet=None, k_emonet=None, template=None, gate_gender=True):
    """Regenerate the GENERAL line from stored global scores with a (seeded) random template."""
    rng = random.Random(seed)
    tmpl = template or TEMPLATE_NAMES[rng.randrange(len(TEMPLATE_NAMES))]
    selector=rec.get("baseline", "humaneness-medium" if rec.get("score_backend")=="humaneness-medium" else "dramabox")
    if rec.get("score_backend")=="humaneness-medium":
        from humaneness_caption_settings import load_settings,caption_view_from_scores
        settings=rec.get("caption_settings") or load_settings()
        base=_baseline(selector)
        d=caption_detail(caption_view_from_scores(rec["global_scores"],base,settings),base,
            k_voicenet=settings['kv'] if k_voicenet is None else k_voicenet,
            k_emonet=settings['ke'] if k_emonet is None else k_emonet,
            always_on=['TEMP'],synonym_seed=rng.randrange(1<<30),template=template or settings['template'])
        return render_caption(d)
    k_voicenet=5 if k_voicenet is None else k_voicenet
    k_emonet=3 if k_emonet is None else k_emonet
    d = caption_detail(rec["global_scores"], _baseline(selector), k_voicenet=k_voicenet, k_emonet=k_emonet,
                       always_on=ALWAYS_ON, synonym_seed=rng.randrange(1 << 30), template=tmpl,
                       shuffle_dims=rng.random() < 0.5,
                       ei_gender=(rec.get("ei_gender") if gate_gender else None))
    return render_caption(d)


def augment_sentence(sent, seed, k_voicenet=None, k_emonet=None, template=None, baseline=None):
    """Regenerate one sentence's style cue from its stored scores, no Age/Gender/Tempo."""
    rng = random.Random(seed)
    tmpl = template or TEMPLATE_NAMES[rng.randrange(len(TEMPLATE_NAMES))]
    selector=baseline or ("humaneness-medium" if sent.get("scores",{}).get("score_backend")=="humaneness-medium" else "dramabox")
    if sent.get("scores",{}).get("score_backend")=="humaneness-medium":
        from humaneness_caption_settings import load_settings,caption_view_from_scores
        settings=sent.get('caption_settings') or load_settings();base=_baseline(selector)
        detail=caption_detail(caption_view_from_scores(sent.get('scores',{}),base,settings),base,
            k_voicenet=min(3,settings['kv']) if k_voicenet is None else k_voicenet,
            k_emonet=settings['ke'] if k_emonet is None else k_emonet,
            always_on=[],synonym_seed=rng.randrange(1<<30),template=template or settings['template'])
        return render_caption(detail)
    k_voicenet=3 if k_voicenet is None else k_voicenet
    k_emonet=3 if k_emonet is None else k_emonet
    d = caption_detail(sent.get("scores", {}), _baseline(selector), k_voicenet=k_voicenet, k_emonet=k_emonet,
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
    if rec.get('score_backend') == 'humaneness-medium':
        from prompt_format import render_timed, TimingUnavailable
        if rec.get('dur') is None or rec.get('asr', {}).get('supported_language') is False:
            raise TimingUnavailable('A timed Medium record needs the measured audio duration and supported ASR alignment')
        sentences = [dict(s, caption=augment_sentence(s, seed ^ (i+1)*0x9E3779B1,
                       template=template, baseline=rec.get('baseline'))) for i,s in enumerate(rec.get('sentences', []))]
        lines, _ = render_timed(sentences, rec.get('words', []), rec.get('burst_events', []), rec['dur'])
        return 'GENERAL: '+augment_global(rec, seed, template=template, gate_gender=gate_gender)+'\nSCRIPT:\n'+'\n'.join(lines)
    out = ["GENERAL: " + augment_global(rec, seed, template=template, gate_gender=gate_gender), "SCRIPT:"]
    for i, s in enumerate(rec.get("sentences", [])):
        cue = augment_sentence(s, seed ^ (i + 1) * 0x9E3779B1, template=template, baseline=rec.get("baseline"))
        text = s.get("text", "")
        if s.get("bursts"):
            marks = [_burst_text(b) for b in s["bursts"]]
            text = (text.rstrip(".") + " " + " ".join(f"({b})" for b in marks if b)).strip()
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
