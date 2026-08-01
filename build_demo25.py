#!/usr/bin/env python3
"""Build `docs/burst-captions-25/` — 25 clips through the current best stack, end to end.

Everything on the page is produced in one pass by `burst_captions.BurstCaptioner`:

* LOCATOR    `laion/vocalburst-locator` — `model_v2.pt`, post-processing 0.50 / 0.10 / 0.10
* CLASSIFIER `laion/vocal-burst-detector-v2` — softmax over 82 burst classes + `no_burst`
* SCORING    VoiceNet 57 dims + genuineness + blend on the VoiceCLAP-commercial embedding;
             the 40 emotions and the gender gate from `laion/Empathic-Insight-Voice-Plus`
* CAPTIONS   the in-domain DramaBox `baseline_stats.json` + reliability weighting

The clips are 25 of the 100 multilingual clips that already ship in `docs/audio/`, chosen
deterministically (see `pick_clips`): burst-bearing clips first, then filled per language,
so all six languages are represented. The page therefore adds **no new audio bytes** — it
references `../audio/*.mp3`, the same 128 kbps mono files the caption grid already uses.

Each card also shows what the SAME scores produced under the previous captioner (emolia
baseline, no reliability weighting, no spread floor), recomputed from the stored scores,
so the effect of the swap is visible per clip rather than asserted.

    python build_demo25.py                      # score + render
    python build_demo25.py --from-results       # re-render from results.json, no models
"""
import argparse
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caption as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "docs", "burst-captions-25")
DEMO_AUDIO = os.path.join(HERE, "docs", "audio")
GRID_JSON = os.path.join(HERE, "docs", "captions.json")
RESULTS = os.path.join(OUT_DIR, "results.json")

# Per-language quota, 25 total, roughly proportional to the 100-clip corpus
# (en 30 / de 25 / zh 15 / fr 12 / ko 10 / ja 8).
QUOTA = {"en": 6, "de": 5, "zh": 4, "ko": 4, "fr": 3, "ja": 3}


def pick_clips():
    """25 clip ids from the bundled 100-clip corpus: within each language, clips the
    blend head flags as burst-bearing come first (they exercise the locator), then the
    rest, both in id order. Deterministic — no sampling, no seed."""
    grid = json.load(open(GRID_JSON, encoding="utf-8"))
    by_lang = {}
    for c in grid:
        by_lang.setdefault(c["lang"], []).append(c)
    out = []
    for lang, n in QUOTA.items():
        cs = sorted(by_lang.get(lang, []), key=lambda c: (not c.get("burst"), c["id"]))
        out += [(lang, c["id"], c.get("burst", False)) for c in cs[:n]]
    return out


# --------------------------------------------------------------------------- #
def esc(s):
    return html.escape(s or "")


def mark(text):
    t = esc(text)
    t = re.sub(r"\[([^\]]+)\]", r'<span class="pause">[\1]</span>', t)
    t = re.sub(r"\(([^)]+)\)", r'<span class="burst">(\1)</span>', t)
    return t


def legacy_caption(scores):
    """The GENERAL line the previous captioner produced from these same scores:
    emolia baseline, no spread floor, no reliability weighting."""
    saved = C.SPREAD_FLOOR_FRAC
    C.SPREAD_FLOOR_FRAC = 0.0
    try:
        base = C.load_baseline("emolia")
        base.pop("_spread_floors", None)
        d = C.caption_detail(scores, base, k_voicenet=5, k_emonet=3,
                             template=os.environ.get("PROC_TEMPLATE", "tags"),
                             reliability_weighting=False)
        return C.render_caption(d), d
    finally:
        C.SPREAD_FLOOR_FRAC = saved


def dim_table(detail):
    rows = []
    for e in detail["voicenet"]:
        if e.get("always_on"):
            continue
        rows.append(f"<tr><td><code>{esc(e['dim'])}</code></td><td>{esc(e['name'])}</td>"
                    f"<td class='n'>{e['z']:+.2f}</td><td class='n'>{e.get('reliability', 1):.2f}</td>"
                    f"<td class='n'>{e.get('rank_score', abs(e['z'])):.2f}</td></tr>")
    if not rows:
        return ""
    return ("<table class='dims'><tr><th>dim</th><th>name</th><th>z</th><th>r</th>"
            "<th>|z|·r</th></tr>" + "".join(rows) + "</table>")


def render_card(r):
    if "error" in r:
        return f'<div class="card"><b>{esc(r["id"])}</b> — ERROR: {esc(r["error"])}</div>'
    sid = r["id"]
    lines = "".join(
        f'<div class="sl"><span class="cue">({esc(ln["cue"])})</span> {mark(ln["text"])}</div>'
        for ln in r.get("script_lines", []))
    script = lines or "<i>(no sentences)</i>"

    spans = []
    for b in r.get("variant_a_bursts", []):
        if b["kept"]:
            spans.append(f'<span class="sp keep">{b["start"]:.2f}–{b["end"]:.2f}s → '
                         f'{esc(b["label"])} (p={b["prob"]:.2f})</span>')
        else:
            why = (f'P(no_burst)={b["p_noburst"]:.2f}' if b["label"] is None
                   else f'{esc(b["label"])} but {b["dur"]:.2f}s &lt; {b["dur_floor"]:.2f}s')
            spans.append(f'<span class="sp disc">{b["start"]:.2f}–{b["end"]:.2f}s → dropped '
                         f'({why})</span>')
    spans_html = " ".join(spans) or '<span class="sp">no span above threshold</span>'

    old = ""
    if r.get("legacy_caption"):
        old = (f'<details class="old"><summary>same scores under the previous captioner '
               f'(emolia baseline · no reliability weighting · no spread floor)</summary>'
               f'<div class="oldline">{esc(r["legacy_caption"])}</div>'
               f'{r.get("legacy_dims_html","")}</details>')

    kept = len([b for b in r.get("variant_a_bursts", []) if b["kept"]])
    return f"""<div class="card" id="{esc(sid)}">
  <div class="hd"><span class="tag lang">{esc(r.get('lang','?'))}</span>
    <a class="cid" href="#{esc(sid)}">{esc(sid)}</a>
    <span class="meta">{r['dur']:.1f}s · {kept} burst(s) kept of {r.get('n_spans',0)} located
    · genu {r['genu']} · blend {r['blend']}</span></div>
  <audio controls preload="none" src="../audio/{esc(sid)}.mp3"></audio>
  <div class="sec"><div class="lbl">GENERAL</div><div class="gcap">{esc(r['global_caption'])}</div></div>
  <div class="sec"><div class="lbl">Top dimensions — z, reliability r, effective rank |z|·r</div>
    {r.get('dims_html','')}</div>
  <div class="sec"><div class="lbl">SCRIPT</div>{script}</div>
  <div class="sec"><div class="lbl">Located spans</div><div class="spans">{spans_html}</div></div>
  {old}
</div>"""


CSS = """
body{background:#0f1117;color:#e6e9f0;font:14px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;padding:20px 26px;max-width:1080px}
h1{font-size:25px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;color:#cfe7ff}
a{color:#7cc4ff}code{background:#1b1f2b;padding:1px 5px;border-radius:4px;font-size:12.5px}
.sub{color:#8b93a7;margin-top:2px}
.box{background:#12141b;border:1px solid #23283a;border-radius:10px;padding:14px 18px;margin:14px 0;color:#aab3c5}
.box b{color:#cfe7ff}
.card{background:#161922;border:1px solid #262b3a;border-radius:10px;padding:13px 16px;margin:14px 0;scroll-margin-top:14px}
.hd{display:flex;align-items:center;gap:9px;margin-bottom:7px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;background:rgba(120,180,255,.18);color:#8fbaff;text-transform:uppercase}
.cid{font-weight:600;color:#e6e9f0;text-decoration:none}
.cid:hover{color:#7cc4ff}
.meta{color:#7f889c;font-size:12px}
audio{height:34px;width:360px;margin:3px 0}
.sec{margin-top:10px}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#6f7a90;margin-bottom:4px}
.gcap{background:#11141c;border-left:3px solid #4a6cff;padding:7px 11px;border-radius:5px}
.sl{background:#11141c;border-left:3px solid #2a3040;padding:5px 11px;border-radius:5px;margin:4px 0}
.cue{color:#9fb4d8}.burst{color:#ffb27a;font-weight:700}.pause{color:#7f889c}
.spans .sp{display:inline-block;background:#1c2130;padding:2px 8px;border-radius:5px;margin:2px;font-size:11.5px;color:#aeb6c8}
.sp.keep{background:rgba(255,150,90,.16);color:#ffb27a;font-weight:600}
.sp.disc{color:#7f889c;font-style:italic}
details.old{margin-top:10px;color:#7f889c;font-size:12.5px}
details.old summary{cursor:pointer}
.oldline{background:#11141c;border-left:3px solid #6b4030;padding:6px 11px;border-radius:5px;margin-top:5px;color:#c9a08c}
table{border-collapse:collapse;margin-top:6px;font-size:12.5px;width:100%}
table.dims td,table.dims th{border:1px solid #262b3a;padding:3px 9px;text-align:left}
table.dims th{color:#cfe7ff;font-weight:600}
table.dims td.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("BC_DEVICE", "cuda:0"))
    ap.add_argument("--from-results", action="store_true",
                    help="re-render the page from results.json without loading models")
    A = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    clips = pick_clips()
    if A.from_results:
        blob = json.load(open(RESULTS, encoding="utf-8"))
        results = blob["results"]
        defaults = blob["defaults"]
    else:
        import burst_captions as B
        bc = B.BurstCaptioner(device=A.device)
        results = []
        for i, (lang, cid, _has_burst) in enumerate(clips):
            path = os.path.join(DEMO_AUDIO, cid + ".mp3")
            try:
                r = bc.process(path, cid=cid)          # mp3 already in docs/audio
            except Exception as e:
                r = {"id": cid, "error": repr(e)}
            r["lang"] = lang
            results.append(r)
            print(f"  [{i+1}/{len(clips)}] {cid}: {str(r.get('global_caption','ERR'))[:90]}",
                  flush=True)
        defaults = {"locator": f"{B.LOCATOR_REPO}/{B.LOCATOR_FILE}",
                    "classifier": f"{B.BURST_CLF_REPO}/{B.BURST_CLF_FILE}",
                    "threshold": B.LOCATOR_THR, "merge_gap": B.MERGE_GAP,
                    "min_duration": B.MIN_BURST_DUR, "no_burst_gate": B.NOBURST_GATE,
                    "scorer": "laion/Empathic-Insight-Voice-Plus",
                    "baseline": os.path.basename(C.baseline_path(None)),
                    "reliability_weighting": C.RELIABILITY_WEIGHTING,
                    "spread_floor_frac": round(C.SPREAD_FLOOR_FRAC, 4)}
        json.dump({"n": len(results), "defaults": defaults, "results": results},
                  open(RESULTS, "w"), ensure_ascii=False, indent=1)

    # per-card tables + the legacy comparison, both pure functions of the stored scores
    base = C.load_baseline()
    n_legacy_rmixd = 0
    for r in results:
        if not isinstance(r.get("scores"), dict):
            continue
        d = C.caption_detail(r["scores"], base, k_voicenet=5, k_emonet=3,
                             template=os.environ.get("PROC_TEMPLATE", "tags"))
        r["dims_html"] = dim_table(d)
        try:
            txt, ld = legacy_caption(r["scores"])
            r["legacy_caption"] = txt
            r["legacy_dims_html"] = dim_table(ld)
            if any(e["dim"] == "R_MIXD" and not e["always_on"] for e in ld["voicenet"]):
                n_legacy_rmixd += 1
        except Exception as e:
            r["legacy_caption"] = f"(unavailable: {e})"

    n_now_rmixd = sum(1 for r in results if r.get("dims_html", "").find(">R_MIXD<") >= 0)
    n_burst = sum(len([b for b in r.get("variant_a_bursts", []) if b.get("kept")])
                  for r in results)
    n_spans = sum(r.get("n_spans", 0) for r in results)
    cards = "\n".join(render_card(r) for r in results)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Procedural Voice Captions — 25-clip grid, current stack</title><style>{CSS}</style></head><body>
<h1>25 clips through the current stack</h1>
<p class="sub">locator <code>model_v2.pt</code> · <code>vocal-burst-detector-v2</code> ·
Empathic-Insight-Voice-Plus · in-domain baseline + reliability weighting ·
{n_burst} bursts written into the scripts from {n_spans} located spans ·
<a href="../">← caption grid</a> · <a href="../burst-captions-v2/">← 14-clip page</a></p>

<div class="box"><p><b>What this page shows.</b> Every clip is scored, transcribed, located,
classified and captioned in one pass by <code>burst_captions.py</code> at its defaults. The
<b>GENERAL</b> line describes the voice; the <b>SCRIPT</b> gives one delivery cue per sentence,
with <span class="burst">(vocal bursts)</span> written in where the locator found them and
<span class="pause">[pause X.Xs]</span> at every silence ≥ 0.30 s.</p>
<p>The 25 clips are drawn from the 100 multilingual clips already in this repo, so the page
adds no new audio: the players point at <code>../audio/</code>.</p></div>

<div class="box"><p><b>Why the captions changed.</b> Each card lists its top dimensions with
three numbers: the z-score, the dimension's measured predictor reliability <code>r</code>
(held-out <code>reg_pearson</code>), and the effective ranking score <code>|z|·r</code> that
decides which dimensions make the caption.</p>
<p>Expand <i>same scores under the previous captioner</i> on any card to see what the identical
scores produced against the old emolia baseline with no reliability weighting. Across these
{len(results)} clips <code>R_MIXD</code> ("mixed resonance", the weakest head in the VoiceNet
release at r = 0.392) took a top-5 slot on <b>{n_legacy_rmixd} of {len(results)}</b> clips under
the old configuration and <b>{n_now_rmixd} of {len(results)}</b> now.</p></div>

{cards}

<p style="color:#6f7a90;margin-top:26px;font-size:12px">Generated by <code>build_demo25.py</code> ·
pipeline in <code>burst_captions.py</code> · captioner in <code>caption.py</code>.
Every dimension, emotion, timestamp and burst on this page is a model prediction; nothing is
hand-written.</p>
</body></html>"""
    open(os.path.join(OUT_DIR, "index.html"), "w").write(doc)
    print(f"wrote {OUT_DIR}/index.html — {len(results)} clips, {n_burst} bursts kept, "
          f"R_MIXD top-5: legacy {n_legacy_rmixd}/{len(results)} -> now {n_now_rmixd}/{len(results)}")


if __name__ == "__main__":
    main()
