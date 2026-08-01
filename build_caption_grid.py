#!/usr/bin/env python3
"""Rebuild the 100-clip caption grid — `docs/captions.json` + the DATA block in `docs/index.html`.

The grid is the landing page: every clip in `docs/audio/` captioned, with the 11 surface
templates assigned round-robin. It is regenerated here because the captioner changed
(in-domain baseline + reliability weighting), and the page's captions were produced by the
old one.

Unlike the original run this also stores each clip's **raw scores** (57 VoiceNet dims +
40 EmoNet emotions + genuineness + blend) in `docs/captions.json`, so the next captioner
change can be re-rendered with `--from-scores` — no GPU, no audio. The scores are kept out
of the inlined DATA block in `index.html`, which only needs the display fields.

    python build_caption_grid.py                 # score audio, then render
    python build_caption_grid.py --from-scores   # re-caption stored scores, no models
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caption as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "docs", "audio")
CAPTIONS = os.path.join(HERE, "docs", "captions.json")
INDEX = os.path.join(HERE, "docs", "index.html")

LANG_NAME = {"en": "English", "de": "German", "zh": "Chinese", "fr": "French",
             "ko": "Korean", "ja": "Japanese"}
DISPLAY_FIELDS = ["id", "lang", "lang_name", "dur", "mp3", "burst", "genu", "blend",
                  "template", "caption", "phrases", "voicenet", "emotions", "quality",
                  "genuineness_gate"]


def clip_ids():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(AUDIO, "*.mp3")))


def render(entry, baseline, template):
    """Caption one clip from its stored raw scores."""
    scores = entry["scores"]
    d = C.caption_detail(scores, baseline, k_voicenet=5, k_emonet=3,
                         synonym_seed=C.stable_hash(entry["id"]) % (1 << 30),
                         template=template, ei_gender=entry.get("ei_gender"))
    out = dict(entry)
    out["template"] = d["template"]
    out["caption"] = C.render_caption(d)
    out["phrases"] = ([e["phrase"] for e in d["voicenet"]]
                      + [e["phrase"] for e in d["emotions"]]
                      + [e["phrase"] for e in d["quality"]])
    out["voicenet"] = d["voicenet"]
    out["emotions"] = d["emotions"]
    out["quality"] = d["quality"]
    out["genuineness_gate"] = d["genuineness_gate"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("BC_DEVICE", "cuda:0"))
    ap.add_argument("--from-scores", action="store_true",
                    help="re-caption the scores already in docs/captions.json")
    A = ap.parse_args()

    if A.from_scores:
        entries = json.load(open(CAPTIONS, encoding="utf-8"))
        missing = [e["id"] for e in entries if not isinstance(e.get("scores"), dict)]
        if missing:
            sys.exit(f"{len(missing)} entries have no stored scores; run without "
                     f"--from-scores once (e.g. {missing[:3]})")
    else:
        import burst_captions as B
        bc = B.BurstCaptioner(device=A.device)
        entries = []
        ids = clip_ids()
        for i, cid in enumerate(ids):
            path = os.path.join(AUDIO, cid + ".mp3")
            wav = B.decode_16k(path)
            vn = bc.voicenet(wav)
            embed = bc.emonet_embed(wav)
            emo = bc.emonet_score([embed])[0] if embed is not None else {}
            scores = {"dims": vn["dims"], "emo": emo,
                      "genu": round(vn["genu"], 4), "blend": round(vn["blend"], 4)}
            lang = cid.split("_")[0].lower()
            entries.append({
                "id": cid, "lang": lang, "lang_name": LANG_NAME.get(lang, lang.upper()),
                "dur": round(wav.shape[0] / B.SR, 1), "mp3": cid + ".mp3",
                # "burst": the vocal-burst blend head puts this clip above its baseline
                # median, i.e. the clip is likely to contain non-verbal bursts.
                "burst": bool(vn["blend"] > 2.5),
                "genu": round(vn["genu"], 2), "blend": round(vn["blend"], 2),
                "ei_gender": bc.gender_ei(embed),
                "scores": scores,
            })
            if (i + 1) % 10 == 0:
                print(f"  scored {i+1}/{len(ids)}", flush=True)

    baseline = C.load_baseline()
    names = C.TEMPLATE_NAMES
    out = [render(e, baseline, names[i % len(names)]) for i, e in enumerate(entries)]

    json.dump(out, open(CAPTIONS, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {CAPTIONS} ({len(out)} clips)")

    # Replace the single inlined `const DATA = [...]` line in the landing page, leaving
    # every other byte of that hand-built page untouched. Scores are display-irrelevant
    # and stay out of the HTML.
    display = [{k: e[k] for k in DISPLAY_FIELDS if k in e} for e in out]
    html = open(INDEX, encoding="utf-8").read()
    new_line = "const DATA = " + json.dumps(display, ensure_ascii=False) + ";"
    html2, n = re.subn(r"^const DATA = .*?;$", lambda _m: new_line, html,
                       count=1, flags=re.M)
    if n != 1:
        sys.exit("could not locate the `const DATA = ...;` line in docs/index.html")
    open(INDEX, "w", encoding="utf-8").write(html2)
    print(f"patched {INDEX} ({len(html2)} bytes)")

    n_rmixd = sum(1 for e in out
                  if any(v["dim"] == "R_MIXD" and not v["always_on"] for v in e["voicenet"]))
    print(f"R_MIXD in top-5 on {n_rmixd}/{len(out)} clips")


if __name__ == "__main__":
    main()
