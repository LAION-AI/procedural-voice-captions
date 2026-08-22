#!/usr/bin/env python3
"""Export the emotion ECDF as a small, stdlib-consumable JSON.

The 40 Empathic-Insight emotion heads are so zero-inflated that a (median, spread)
pair cannot describe them: measured over the full corpus, `Awe` is zero on 91.3 % of
utterances and its IQR is 0.0002. Any z-score built on that is meaningless, and the
group spread floor cannot rescue it because the floor is derived from the same
collapsed spreads.

A percentile answers the question the caption actually asks -- "is this clip unusually
X?" -- without assuming a shape. This exports, per emotion, the value thresholds at a
few upper-tail levels plus a knot table for general value->percentile lookup, from the
same tie-aware mid-rank ECDF used to caption the corpus (132,833,726 utterances).

Consumed by `emotion_gate.py`, which is stdlib-only (bisect).
"""
import argparse, json
import numpy as np

NBIN = 4096
LEVELS = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
KNOT_P = [round(x, 4) for x in np.arange(0.0, 1.0001, 0.005)]

EMO_PRETTY = {  # corpus column -> Empathic-Insight taxonomy key used by caption.py
    "emo_Astonishment_Surprise": "Astonishment/Surprise",
    "emo_Emotional_Numbness": "Emotional Numbness",
    "emo_Fatigue_Exhaustion": "Fatigue/Exhaustion",
    "emo_Hope_Enthusiasm_Optimism": "Hope/Optimism",
    "emo_Impatience_and_Irritability": "Impatience and Irritability",
    "emo_Intoxication_Altered_States_of_Consciousness": "Intoxication/Altered States",
    "emo_Jealousy_and_Envy": "Jealousy & Envy",
    "emo_Malevolence_Malice": "Malevolence/Malice",
    "emo_Pleasure_Ecstasy": "Pleasure/Ecstasy",
    "emo_Sexual_Lust": "Sexual Lust",
    "emo_Thankfulness_Gratitude": "Thankfulness/Gratitude",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capnorm", required=True)
    ap.add_argument("--out", default="emotion_percentiles.json")
    A = ap.parse_args()

    z = np.load(A.capnorm, allow_pickle=True)
    fields = [str(x) for x in z["fields"]]
    H = z["hist"].astype(np.float64)
    LO, HI = z["lo"].astype(np.float64), z["hi"].astype(np.float64)
    n_total = int(z["n"][0])

    out = {}
    for d, col in enumerate(fields):
        c = H[d]
        n = c.sum()
        w = (HI[d] - LO[d]) / NBIN
        cum = np.cumsum(c)
        # mid-rank percentile of each bucket, and the bucket's upper edge value
        below = cum - c
        mid = (below + 0.5 * c) / max(n, 1.0)
        edge = np.empty(NBIN + 2)
        edge[0] = LO[d]
        edge[1:NBIN + 1] = LO[d] + np.arange(1, NBIN + 1) * w
        edge[NBIN + 1] = HI[d]

        def value_at(p):
            i = int(np.searchsorted(cum, p * n, side="left"))
            i = min(max(i, 0), NBIN + 1)
            if i == 0:
                return float(LO[d])
            if i >= NBIN + 1:
                return float(HI[d])
            b = cum[i - 1]
            frac = (p * n - b) / c[i] if c[i] > 0 else 0.0
            return float(LO[d] + (i - 1 + min(max(frac, 0.0), 1.0)) * w)

        key = EMO_PRETTY.get(col, col[4:].replace("_", " ") if col.startswith("emo_") else col)
        out[key] = {
            "column": col,
            "pct_at_or_below_zero": round(float(c[: max(int((0 - LO[d]) / w) + 2, 1)].sum() / max(n, 1) * 100), 2),
            "thresholds": {f"{p:.2f}": round(value_at(p), 6) for p in LEVELS},
            "knots_v": [round(value_at(p), 6) for p in KNOT_P],
        }

    doc = {
        "_meta": {
            "description": (
                "Per-emotion empirical distribution of the Empathic-Insight heads over the "
                "full annotated LAION-TTS corpus. `thresholds[p]` is the raw score at "
                "percentile p; `knots_v` are the values at knots_p for value->percentile "
                "lookup by interpolation."),
            "estimator": "tie-aware mid-rank ECDF, 4096 fixed bins per emotion",
            "n_rows": n_total,
            "scope": "POOLED across datasets and languages",
            "knots_p": KNOT_P,
            "recommended_gate": {
                "rule": "name an emotion when its percentile >= 0.90, at most 3, ranked by percentile",
                "floor": 0.90, "top_n": 3,
                "none_clause": "no dominant emotion",
                "why": ("an absolute threshold on raw scores names whichever head happens to "
                        "sit highest on its own scale; measured on 165,516,420 captions an "
                        "absolute 1.0 gate named Interest on 90.6 % of all rows and "
                        "Bitterness on 0.1 %."),
            },
        }
    }
    doc.update(dict(sorted(out.items())))
    json.dump(doc, open(A.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {A.out}: {len(out)} emotions over {n_total:,} rows")
    for k in ["Interest", "Awe", "Infatuation", "Sadness"]:
        t = out[k]["thresholds"]
        print(f"  {k:12s} zeros={out[k]['pct_at_or_below_zero']:5.1f}%  p90={t['0.90']:.4f}  p99={t['0.99']:.4f}")


if __name__ == "__main__":
    main()
