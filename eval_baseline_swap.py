#!/usr/bin/env python3
"""Measure what the baseline swap + reliability weighting actually change.

Runs the captioner over stored raw scores under four configurations and reports how
often each VoiceNet dimension takes a top-k slot, plus the |z| distribution. No models
and no audio — `docs/burst-captions-v2/results.json` already carries every clip's and
every sentence's 57 dims + 40 emotions.

    A  emolia   baseline, no spread floor, no reliability   <- what main did
    B  emolia   baseline, + floor + reliability
    C  dramabox baseline, + floor, no reliability
    D  dramabox baseline, + floor + reliability             <- the new default

    python eval_baseline_swap.py                 # the 14 demo clips (global captions)
    python eval_baseline_swap.py --sentences     # + every sentence-level caption
    python eval_baseline_swap.py --dim R_MIXD    # focus dimension (default R_MIXD)
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caption as C  # noqa: E402

CONFIGS = [
    ("A  emolia   / no floor / no reliability", "emolia", 0.0, False),
    ("B  emolia   / floor    / reliability   ", "emolia", 1.0 / 3.0, True),
    ("C  dramabox / floor    / no reliability", "dramabox", 1.0 / 3.0, False),
    ("D  dramabox / floor    / reliability   ", "dramabox", 1.0 / 3.0, True),
]


def records(path, with_sentences):
    r = json.load(open(path, encoding="utf-8"))
    for clip in r["results"]:
        if isinstance(clip.get("scores"), dict):
            yield clip["id"], clip["scores"], 5
        if not with_sentences:
            continue
        for i, s in enumerate(clip.get("sentences") or []):
            if isinstance(s.get("scores"), dict):
                yield f"{clip['id']}#s{i}", s["scores"], 3


def run(cfg, recs, focus):
    label, base_name, floor, rel = cfg
    C.SPREAD_FLOOR_FRAC = floor
    base = C.load_baseline(base_name)
    base.pop("_spread_floors", None)
    wins = collections.Counter()
    n_top = 0
    focus_abs_z, other_abs_z = [], []
    sel_rel, selections = [], {}
    for _id, scores, kv in recs:
        d = C.caption_detail(scores, base, k_voicenet=kv, k_emonet=3,
                             reliability_weighting=rel)
        top = [e for e in d["voicenet"] if not e["always_on"]]
        for e in top:
            wins[e["dim"]] += 1
            sel_rel.append(e["reliability"])
        selections[_id] = {e["dim"] for e in top}
        n_top += 1
        # |z| of every scored dim under this configuration, focus dim vs the rest
        for code, v in (scores.get("dims") or {}).items():
            st = base.get(code)
            if not st or st.get("group") != "voicenet" or code not in C.DESC:
                continue
            az = abs(C._zscore(C._val(v), st, base))
            (focus_abs_z if code == focus else other_abs_z).append(az)
    return label, wins, n_top, focus_abs_z, other_abs_z, sel_rel, selections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "docs", "burst-captions-v2",
                                                      "results.json"))
    ap.add_argument("--sentences", action="store_true")
    ap.add_argument("--dim", default="R_MIXD")
    A = ap.parse_args()

    recs = list(records(A.results, A.sentences))
    print(f"{len(recs)} captions from {A.results}"
          f"{' (clips + sentences)' if A.sentences else ' (clips only)'}\n")

    saved = C.SPREAD_FLOOR_FRAC
    rows = [run(c, recs, A.dim) for c in CONFIGS]
    C.SPREAD_FLOOR_FRAC = saved

    print(f"{'configuration':40s} {A.dim+' top-k wins':>18s} {'mean |z| '+A.dim:>16s} "
          f"{'median |z| others':>18s} {'mean r of selected':>20s}")
    for label, wins, n_top, fz, oz, sr, _sel in rows:
        w = wins[A.dim]
        print(f"{label:40s} {w:>6d}/{n_top:<11d} {statistics.mean(fz) if fz else 0:>16.2f} "
              f"{statistics.median(oz) if oz else 0:>18.2f} {statistics.mean(sr) if sr else 0:>20.3f}")

    # What reliability weighting alone buys, holding the baseline fixed (A->B, C->D).
    for a, b in ((0, 1), (2, 3)):
        sa, sb = rows[a][6], rows[b][6]
        moved = sum(len(sa[k] ^ sb[k]) // 2 for k in sa)
        tot = sum(len(sa[k]) for k in sa)
        print(f"\n  reliability weighting, {rows[a][0].split()[0]} -> {rows[b][0].split()[0]}: "
              f"{moved}/{tot} selected slots changed "
              f"({100.0*moved/max(tot,1):.1f} %), mean reliability of selected dims "
              f"{statistics.mean(rows[a][5]):.3f} -> {statistics.mean(rows[b][5]):.3f}")

    for label, wins, n_top, fz, oz, sr, _sel in rows:
        print(f"\n{label} — most frequent top-k dimensions")
        for dim, c in wins.most_common(8):
            print(f"    {dim:8s} {c:4d}/{n_top}  ({100.0*c/n_top:.0f} %)")


if __name__ == "__main__":
    main()
