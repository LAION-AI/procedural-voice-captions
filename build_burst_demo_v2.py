#!/usr/bin/env python3
"""Build `docs/burst-captions-v2/` — the current output of the default pipeline.

Re-annotates a handful of clips that already live in this repo (`docs/audio/` and
`docs/burst-captions/audio/`) with the **current defaults**:

* LOCATOR   `laion/vocalburst-locator` — `model_v2.pt`, 30 s windows stitched onto
            one timeline, post-processing 0.50 / 0.10 / 0.10;
* CLASSIFIER `laion/vocal-burst-detector-v2` — softmax over 82 burst classes + `no_burst`.

Where the same clip appears in the old `docs/burst-captions/results.json` (locator v1 +
the multi-label classifier at 0.75 / 0.30 / 0.30) the old script line is shown next to
the new one, so the change is visible rather than asserted.

The page embeds every clip as a 128 kbps mono mp3, so it is self-contained.

    python build_burst_demo_v2.py                 # all clips
    python build_burst_demo_v2.py --no-long       # skip the constructed long-form clip
    python build_burst_demo_v2.py --recaption     # re-caption results.json, no models

`--recaption` exists because a caption is a pure function of the stored raw scores plus
the baseline (see augment.py). `results.json` already carries every clip's and every
sentence's 57 dims + 40 emotions, so when only the *captioner* changes — a new baseline,
reliability weighting — the page can be rebuilt exactly, with no GPU and no audio. The
detected bursts, spans, transcripts and timings are untouched by such a change and are
therefore reused verbatim rather than recomputed.
"""
import os, sys, json, html, re, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burst_captions as B
import caption as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "docs", "burst-captions-v2")
AUDIO_OUT = os.path.join(OUT_DIR, "audio")
DEMO_AUDIO = os.path.join(HERE, "docs", "audio")
BURST_AUDIO = os.path.join(HERE, "docs", "burst-captions", "audio")
OLD_RESULTS = os.path.join(HERE, "docs", "burst-captions", "results.json")

# Clips already committed to this repo. (kind, id, source path)
CHARACTER = ["pain_scream5", "zombie5", "goblin5", "mouse5", "sad_woman5", "asmr_woman5",
             "ranting5", "dragon5", "evil_ghost5"]
IN_THE_WILD = ["ZH_B00039_S04341_W000000", "FR_B00001_S08689_W000031",
               "EN_B00000_S03298_W000123", "EN_B00026_S00187_W000083"]
# A >30 s clip has to be constructed — nothing in the repo is longer than ~18 s. This one is
# an honest concatenation, labelled as such on the page, and exists to exercise the windowing.
LONG_PARTS = ["zombie5", "mouse5", "goblin5", "ranting5", "sad_woman5",
              "asmr_woman5", "dragon5", "pain_scream5", "evil_ghost5"]
LONG_ID = "longform_concat_91s"


def pick_clips(with_long=True):
    clips = []
    for cid in CHARACTER:
        p = os.path.join(BURST_AUDIO, cid + ".mp3")
        if os.path.exists(p):
            clips.append(("character", cid, p))
    for cid in IN_THE_WILD:
        p = os.path.join(BURST_AUDIO, cid + ".mp3")
        if not os.path.exists(p):
            p = os.path.join(DEMO_AUDIO, cid + ".mp3")
        if os.path.exists(p):
            clips.append(("in-the-wild", cid, p))
    if with_long:
        p = build_long_clip()
        if p:
            clips.append(("long-form", LONG_ID, p))
    return clips


def build_long_clip():
    """Concatenate several demo clips into one >30 s wav so the 30 s windowing is exercised."""
    import torch, soundfile as sf
    parts = []
    for cid in LONG_PARTS:
        p = os.path.join(BURST_AUDIO, cid + ".mp3")
        if os.path.exists(p):
            parts.append(B.decode_16k(p))
    if not parts:
        return None
    wav = torch.cat(parts)
    os.makedirs(AUDIO_OUT, exist_ok=True)
    out = os.path.join(AUDIO_OUT, LONG_ID + ".wav")
    sf.write(out, wav.numpy(), B.SR)
    return out


def old_lookup():
    """{clip_id: old result} from the previous demo run (locator v1 defaults)."""
    if not os.path.exists(OLD_RESULTS):
        return {}
    try:
        d = json.load(open(OLD_RESULTS))
        return {r["id"]: r for r in d.get("results", []) if "id" in r}
    except Exception:
        return {}


def esc(s):
    return html.escape(s or "")


def mark(text):
    """Highlight (bursts / cues) and [pauses] in a script line."""
    t = esc(text)
    t = re.sub(r"\[([^\]]+)\]", r'<span class="pause">[\1]</span>', t)
    t = re.sub(r"\(([^)]+)\)", r'<span class="burst">(\1)</span>', t)
    return t


def render_card(kind, r, old):
    if "error" in r:
        return f'<div class="card"><b>{esc(r["id"])}</b> — ERROR: {esc(r["error"])}</div>'
    sid = r["id"]
    lines = []
    for ln in r.get("script_lines", []):
        lines.append(f'<div class="sl"><span class="cue">({esc(ln["cue"])})</span> {mark(ln["text"])}</div>')
    script = "".join(lines) or "<i>(no sentences)</i>"

    spans = []
    for b in r.get("variant_a_bursts", []):
        if b["kept"]:
            spans.append(f'<span class="sp keep">{b["start"]:.2f}–{b["end"]:.2f}s → {esc(b["label"])} '
                         f'(p={b["prob"]:.2f})</span>')
        else:
            why = (f'P(no_burst)={b["p_noburst"]:.2f}' if b["label"] is None
                   else f'{esc(b["label"])} but {b["dur"]:.2f}s &lt; {b["dur_floor"]:.2f}s')
            spans.append(f'<span class="sp disc">{b["start"]:.2f}–{b["end"]:.2f}s → dropped ({why})</span>')
    spans_html = " ".join(spans) if spans else '<span class="sp">no span above threshold</span>'

    oldhtml = ""
    o = old.get(sid)
    if o and o.get("variant_a_inline"):
        okept = [x for x in o.get("variant_a_bursts", []) if x.get("kept")]
        oldhtml = (f'<details class="old"><summary>previous output — locator v1 + multi-label classifier, '
                   f'defaults 0.75 / 0.30 / 0.30 ({len(okept)} burst(s) kept of {o.get("n_spans", 0)} spans)'
                   f'</summary><div class="oldline">{mark(o["variant_a_inline"])}</div></details>')

    src = "audio/%s.mp3" % esc(sid)
    win = r.get("n_locator_windows", 1)
    winnote = (f' · <b>{win} locator windows</b>' if win > 1 else " · 1 locator window")
    emo_note = "" if r.get("emonet") else ' <span class="warn">(EmoNet off)</span>'
    note = ('<div class="note">Constructed by concatenating the nine character clips above — the only '
            'way to show the &gt;30 s path with audio that ships in this repo.</div>'
            if kind == "long-form" else "")
    return f"""<div class="card" id="{esc(sid)}">
  <div class="hd"><span class="tag {kind}">{esc(kind)}</span>
    <a class="cid" href="#{esc(sid)}">{esc(sid)}</a>
    <span class="meta">{r['dur']:.1f}s{winnote} · {len([b for b in r.get('variant_a_bursts',[]) if b['kept']])} burst(s) kept
    of {r.get('n_spans',0)} located · genu {r['genu']} · blend {r['blend']}{emo_note}</span></div>
  {note}
  <audio controls preload="none" src="{src}"></audio>
  <div class="sec"><div class="lbl">GENERAL</div><div class="gcap">{esc(r['global_caption'])}</div></div>
  <div class="sec"><div class="lbl">SCRIPT</div>{script}</div>
  <div class="sec"><div class="lbl">Located spans</div><div class="spans">{spans_html}</div></div>
  {oldhtml}
</div>"""


CSS = """
body{background:#0f1117;color:#e6e9f0;font:14px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;padding:20px 26px;max-width:1080px}
h1{font-size:25px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;color:#cfe7ff}
a{color:#7cc4ff}code{background:#1b1f2b;padding:1px 5px;border-radius:4px;font-size:12.5px}
.sub{color:#8b93a7;margin-top:2px}
.box{background:#12141b;border:1px solid #23283a;border-radius:10px;padding:14px 18px;margin:14px 0;color:#aab3c5}
.box b{color:#cfe7ff}
.legend{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;align-items:baseline;margin-top:8px}
.card{background:#161922;border:1px solid #262b3a;border-radius:10px;padding:13px 16px;margin:14px 0}
.hd{display:flex;align-items:center;gap:9px;margin-bottom:7px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.character{background:rgba(255,150,90,.18);color:#ffb27a}
.in-the-wild{background:rgba(120,180,255,.18);color:#8fbaff}
.long-form{background:rgba(150,255,190,.15);color:#8fe6b4}
.cid{font-weight:600;color:#e6e9f0;text-decoration:none}
.cid:hover{color:#7cc4ff}
.card{scroll-margin-top:14px}.meta{color:#7f889c;font-size:12px}.warn{color:#e0b062}
.note{color:#8fe6b4;font-size:12.5px;margin:2px 0 6px}
audio{height:34px;width:360px;margin:3px 0}
.sec{margin-top:10px}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#6f7a90;margin-bottom:4px}
.gcap{background:#11141c;border-left:3px solid #4a6cff;padding:7px 11px;border-radius:5px}
.sl{background:#11141c;border-left:3px solid #2a3040;padding:5px 11px;border-radius:5px;margin:4px 0}
.cue{color:#9fb4d8}
.burst{color:#ffb27a;font-weight:700}
.pause{color:#7f889c}
.spans .sp{display:inline-block;background:#1c2130;padding:2px 8px;border-radius:5px;margin:2px;font-size:11.5px;color:#aeb6c8}
.sp.keep{background:rgba(255,150,90,.16);color:#ffb27a;font-weight:600}
.sp.disc{color:#7f889c;font-style:italic}
details.old{margin-top:10px;color:#7f889c;font-size:12.5px}
details.old summary{cursor:pointer}
.oldline{background:#11141c;border-left:3px solid #3a3040;padding:6px 11px;border-radius:5px;margin-top:5px;color:#9aa3b5}
table{border-collapse:collapse;margin-top:8px;font-size:13px}
th,td{border:1px solid #262b3a;padding:5px 10px;text-align:left}th{color:#cfe7ff}
"""


def kind_of(cid):
    if cid == LONG_ID:
        return "long-form"
    return "character" if cid in CHARACTER else "in-the-wild"


def recaption(results, baseline=None):
    """Re-derive every caption on the page from the stored raw scores.

    Only the captioner's output changes: `global_caption`, each sentence's `caption` /
    `variant_b_caption`, and the cue on each script line. Bursts, spans, transcripts and
    timings are model outputs that a captioner change cannot affect, so they are left
    exactly as measured."""
    base = C.load_baseline(baseline)
    tmpl = os.environ.get("PROC_TEMPLATE", B.PROC_TEMPLATE)
    for r in results:
        if not isinstance(r.get("scores"), dict):
            continue
        seed = B.stable_hash(r["id"]) % (1 << 30)
        d = C.caption_detail(r["scores"], base, k_voicenet=5, k_emonet=3, synonym_seed=seed,
                             template=tmpl, ei_gender=r.get("ei_gender"))
        r["global_caption"] = C.render_caption(d)
        lines = r.get("script_lines") or []
        for i, s in enumerate(r.get("sentences") or []):
            if not isinstance(s.get("scores"), dict):
                continue
            sd = C.caption_detail(s["scores"], base, k_voicenet=3, k_emonet=3, always_on=[],
                                  synonym_seed=seed ^ (i + 1), template=tmpl)
            s["caption"] = C.render_caption(sd)
            lab = s.get("variant_b_burst")
            s["variant_b_caption"] = (
                s["caption"] if lab is None else
                s["caption"].rstrip(".") + ", " +
                B.BurstCaptioner.variant_b_phrase(lab, seed ^ (i + 1)) + ".")
            if i < len(lines):
                lines[i]["cue"] = s["caption"]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--device", default=os.environ.get("BC_DEVICE", "cuda:0"))
    ap.add_argument("--recaption", action="store_true",
                    help="rebuild from results.json with the current captioner, no models")
    ap.add_argument("--baseline", default=None, help="baseline name or path")
    A = ap.parse_args()

    res_path = os.path.join(OUT_DIR, "results.json")
    old = old_lookup()

    if A.recaption:
        blob = json.load(open(res_path, encoding="utf-8"))
        results = [(kind_of(r["id"]), r) for r in recaption(blob["results"], A.baseline)]
        blob["defaults"]["baseline"] = os.path.basename(C.baseline_path(A.baseline))
        blob["defaults"]["reliability_weighting"] = C.RELIABILITY_WEIGHTING
        blob["defaults"]["spread_floor_frac"] = round(C.SPREAD_FLOOR_FRAC, 4)
        blob["results"] = [r for _, r in results]
        json.dump(blob, open(res_path, "w"), ensure_ascii=False, indent=1)
        print(f"re-captioned {len(results)} clips from stored scores")
    else:
        os.makedirs(AUDIO_OUT, exist_ok=True)
        use_emonet = os.environ.get("BC_EMONET", "1") != "0"
        bc = B.BurstCaptioner(device=A.device, use_emonet=use_emonet, baseline=A.baseline)
        clips = pick_clips(with_long=not A.no_long)
        print(f"selected {len(clips)} clips")

        results = []
        for i, (kind, cid, path) in enumerate(clips):
            mp3 = os.path.join(AUDIO_OUT, cid + ".mp3")
            try:
                r = bc.process(path, cid=cid, mp3_out=mp3)
            except Exception as e:
                r = {"id": cid, "error": repr(e)}
            results.append((kind, r))
            print(f"  [{i+1}/{len(clips)}] {cid}: {r.get('global_caption', 'ERR')[:80]}", flush=True)

        json.dump({"n": len(results),
                   "defaults": {"locator": f"{B.LOCATOR_REPO}/{B.LOCATOR_FILE}",
                                "classifier": f"{B.BURST_CLF_REPO}/{B.BURST_CLF_FILE}",
                                "threshold": B.LOCATOR_THR, "merge_gap": B.MERGE_GAP,
                                "min_duration": B.MIN_BURST_DUR, "no_burst_gate": B.NOBURST_GATE,
                                "chunk_sec": B.CHUNK_SEC, "chunk_overlap": B.CHUNK_OVERLAP,
                                "scorer": "laion/Empathic-Insight-Voice-Plus",
                                "baseline": os.path.basename(C.baseline_path(A.baseline)),
                                "reliability_weighting": C.RELIABILITY_WEIGHTING,
                                "spread_floor_frac": round(C.SPREAD_FLOOR_FRAC, 4)},
                   "results": [r for _, r in results]},
                  open(res_path, "w"), ensure_ascii=False, indent=1)

        # drop the intermediate wav for the long clip; the mp3 is what the page uses
        wav_tmp = os.path.join(AUDIO_OUT, LONG_ID + ".wav")
        if os.path.exists(wav_tmp):
            os.remove(wav_tmp)

    cards = "\n".join(render_card(k, r, old) for k, r in results)
    n_burst = sum(len([b for b in r.get("variant_a_bursts", []) if b.get("kept")]) for _, r in results)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Procedural Voice Captions — locator v2 + classifier v2</title><style>{CSS}</style></head><body>
<h1>Procedural voice captions <span style="color:#ffb27a">— current defaults</span></h1>
<p class="sub">{len(results)} clips re-annotated with <b>vocalburst-locator <code>model_v2.pt</code></b> +
<b>vocal-burst-detector-v2</b> · {n_burst} bursts written into the scripts ·
<a href="../">← caption grid</a> · <a href="../burst-captions/">← previous burst page (v1)</a></p>

<div class="box"><p><b>How to read a caption.</b> Every clip gets one <b>GENERAL</b> line describing how the
voice sounds overall, then a <b>SCRIPT</b> with one line per sentence.</p>
<div class="legend">
<span class="cue">(round brackets)</span><span>a <b>delivery cue</b> at the head of a sentence, or a
<span class="burst">(vocal burst)</span> written inline at the moment the locator found it.</span>
<span class="pause">[square brackets]</span><span>a <b>silence</b>, <code>[pause X.Xs]</code>, taken from the
Parakeet-v3 word timestamps (any gap ≥ 0.30 s).</span>
</div></div>

<div class="box"><p><b>The two burst models are different stages.</b>
The <b>locator</b> (<code>laion/vocalburst-locator</code>, <code>model_v2.pt</code>) says <i>where</i> a burst is:
a 50 fps burst probability per frame over a fixed 30 s window. Longer audio is scanned in overlapping
30 s windows whose frame probabilities are stitched onto one timeline before events are extracted, so a
burst on a seam is neither lost nor counted twice. The <b>classifier</b>
(<code>laion/vocal-burst-detector-v2</code>) says <i>what</i> it is: it names the cut-out span with one of
82 taxonomy classes, or vetoes it with <code>no_burst</code>.</p>
<p>Post-processing here is <b>threshold {B.LOCATOR_THR} · merge_gap {B.MERGE_GAP} s · min_duration {B.MIN_BURST_DUR} s</b>
— the operating point measured for <code>model_v2.pt</code> on 992 held-out real clips (event F1 0.607),
not the v1 defaults of 0.65 / 0.30 / 0.50. Spans whose <code>P(no_burst) ≥ {B.NOBURST_GATE}</code> are dropped.
Where the same clip was annotated by the old pipeline, expand <i>previous output</i> under a card to compare.</p></div>

{cards}

<p style="color:#6f7a90;margin-top:26px;font-size:12px">Generated by <code>build_burst_demo_v2.py</code> ·
pipeline in <code>burst_captions.py</code>. Every dimension, emotion, timestamp and burst on this page is a
model prediction; nothing is hand-written.</p>
</body></html>"""
    open(os.path.join(OUT_DIR, "index.html"), "w").write(doc)
    print("wrote", os.path.join(OUT_DIR, "index.html"))


if __name__ == "__main__":
    main()
