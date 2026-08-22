#!/usr/bin/env python3
"""Build `baseline_stats_laiontts.json` from the LAION-TTS corpus histograms.

WHY ANOTHER BASELINE
--------------------
`baseline_stats.json` (DramaBox, 256,000 clips, spread = IQR/1.349) and
`baseline_stats_emolia.json` (4,703 + 1,000 clips, spread = 1.4826*MAD with an std
fallback) both estimate "the average voice" from a sample. This one estimates it from
**130,785,282 utterances** -- the whole annotated LAION-TTS corpus, 8 datasets, every
language -- so it is the same quantity measured with ~511x more data than the current
default.

It is written in the EXACT schema `caption.py` already consumes (`median`, `spread`,
`p10`, `p90`, `mean`, `std`, `mad`, `n`, `name`, `group`, `range`,
`reliability_reg_pearson`, `synonyms`), using the shipped default's own
`spread = IQR/1.349` convention, so it drops into `BASELINES` without touching the
z-scoring code. Per-dimension metadata that is a property of the MODEL rather than of
the corpus -- display name, group, range, reliability, emotion synonyms -- is carried
over from the reference baseline, exactly as `import_baseline.py` does.

HOW IT IS MEASURED
------------------
Not from a sample. `traj2/stats/globalhist.npz` is an exact fixed-bin histogram (4096
bins per dimension plus underflow/overflow) accumulated over every scored row in the
corpus. Quantiles are read off the cumulative histogram with linear interpolation
inside the containing bin; mean and std are computed from bin centres; MAD is a second
weighted-quantile pass over |centre - median|. Bin width is (hi-lo)/4096, which for
these ranges is far finer than the reported 4 decimal places.

WHAT THIS FILE DOES *NOT* FIX
-----------------------------
A median and a spread describe a distribution by its centre and width. Several
Empathic-Insight emotion heads are so zero-inflated that no (centre, width) pair
describes them usefully -- `Infatuation` is exactly zero on 87.7 % of the corpus,
`Awe` on 91.3 % -- which is the pathology the `1.4826*MAD < 0.5*std` fallback in
`compute_baseline.py` was invented to paper over. For the emotion clause specifically,
a percentile gate is strictly better than a z-score, and the companion `capnorm.npz`
(a tie-aware mid-rank ECDF over 132,833,726 rows) provides one. See the README section
"A percentile gate for the emotion clause".

Usage:
    python make_laiontts_baseline.py --hist /path/to/globalhist.npz \
        --reference baseline_stats.json --out baseline_stats_laiontts.json
"""
import argparse, json, os
import numpy as np

NBIN = 4096

# EmoNet taxonomy key -> corpus column. Only the non-mechanical ones need naming.
EMO_OVERRIDE = {
    "Astonishment/Surprise": "emo_Astonishment_Surprise",
    "Emotional Numbness": "emo_Emotional_Numbness",
    "Fatigue/Exhaustion": "emo_Fatigue_Exhaustion",
    "Hope/Optimism": "emo_Hope_Enthusiasm_Optimism",
    "Impatience and Irritability": "emo_Impatience_and_Irritability",
    "Intoxication/Altered States": "emo_Intoxication_Altered_States_of_Consciousness",
    "Jealousy & Envy": "emo_Jealousy_and_Envy",
    "Malevolence/Malice": "emo_Malevolence_Malice",
    "Pleasure/Ecstasy": "emo_Pleasure_Ecstasy",
    "Sexual Lust": "emo_Sexual_Lust",
    "Thankfulness/Gratitude": "emo_Thankfulness_Gratitude",
}
QUALITY = {"genuineness": "genuineness_0_6", "blend": "blend_0_10"}


def column_for(key, group):
    if group == "voicenet":
        return f"vn_{key}_reg"
    if group == "quality":
        return QUALITY.get(key)
    return EMO_OVERRIDE.get(key, "emo_" + key.replace(" ", "_"))


class Hist:
    """Fixed-bin histogram with underflow at index 0 and overflow at NBIN+1."""

    def __init__(self, counts, lo, hi):
        self.c = counts.astype(np.float64)
        self.lo, self.hi = float(lo), float(hi)
        self.w = (self.hi - self.lo) / NBIN
        self.n = float(self.c.sum())
        # representative value of every bucket, incl. the two tails
        self.x = np.empty(NBIN + 2)
        self.x[0] = self.lo
        self.x[1:NBIN + 1] = self.lo + (np.arange(NBIN) + 0.5) * self.w
        self.x[NBIN + 1] = self.hi

    def quantile(self, q):
        """Linear interpolation inside the bin that contains the q-th value."""
        target = q * self.n
        cum = np.cumsum(self.c)
        i = int(np.searchsorted(cum, target, side="left"))
        i = min(max(i, 0), NBIN + 1)
        if i == 0:
            return self.lo
        if i == NBIN + 1:
            return self.hi
        below = cum[i - 1]
        if self.c[i] <= 0:
            return self.lo + (i - 1) * self.w
        frac = (target - below) / self.c[i]
        return self.lo + (i - 1 + min(max(frac, 0.0), 1.0)) * self.w

    def mean_std(self):
        p = self.c / max(self.n, 1.0)
        m = float((p * self.x).sum())
        v = float((p * (self.x - m) ** 2).sum())
        return m, float(np.sqrt(max(v, 0.0)))

    def mad(self, med):
        """Median of |x - med|, as a weighted quantile over the same buckets."""
        d = np.abs(self.x - med)
        o = np.argsort(d)
        cum = np.cumsum(self.c[o])
        j = int(np.searchsorted(cum, 0.5 * self.n, side="left"))
        return float(d[o[min(j, len(o) - 1)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--reference", default="baseline_stats.json")
    ap.add_argument("--out", default="baseline_stats_laiontts.json")
    A = ap.parse_args()

    z = np.load(A.hist, allow_pickle=True)
    fields = [str(x) for x in z["fields"]]
    H, LO, HI = z["hist"], z["lo"], z["hi"]
    idx = {f: i for i, f in enumerate(fields)}
    ref = json.load(open(A.reference))

    stats, missing, n_rows = {}, [], 0
    for key, rs in ref.items():
        if key == "_meta":
            continue
        col = column_for(key, rs.get("group", ""))
        if col is None or col not in idx:
            missing.append(key)
            continue
        h = Hist(H[idx[col]], LO[idx[col]], HI[idx[col]])
        n_rows = max(n_rows, int(h.n))
        med = h.quantile(0.5)
        q1, q3 = h.quantile(0.25), h.quantile(0.75)
        spread = (q3 - q1) / 1.349
        mean, std = h.mean_std()
        if spread < 1e-6:                      # a dimension with no usable spread
            spread = std if std > 1e-6 else 1.0
        s = {"mean": round(mean, 4), "median": round(med, 4), "std": round(std, 4),
             "mad": round(h.mad(med), 4), "spread": round(spread, 4),
             "p10": round(h.quantile(0.10), 4), "p90": round(h.quantile(0.90), 4),
             "n": int(h.n), "name": rs.get("name", key), "group": rs["group"],
             "source": "laion_tts_corpus_full", "corpus_column": col}
        if "range" in rs:
            s["range"] = rs["range"]
        if "reliability_reg_pearson" in rs:
            s["reliability_reg_pearson"] = rs["reliability_reg_pearson"]
        if "synonyms" in rs:
            s["synonyms"] = rs["synonyms"]
        stats[key] = s

    meta = {
        "description": (
            "Baseline distribution statistics measured over the FULL annotated LAION-TTS "
            "corpus rather than a sample: 130,785,282 utterances, 8 datasets, all languages. "
            "Schema and z-score rule are identical to baseline_stats.json so this file is a "
            "drop-in for BASELINES/PVC_BASELINE."),
        "z_score_rule": "z = (value - median) / spread ; spread = IQR/1.349 (robust sigma).",
        "spread_definition": "IQR/1.349, matching the shipped dramabox default.",
        "estimator": (
            "Exact fixed-bin histogram (4096 bins/dimension + underflow/overflow) accumulated "
            "over every scored row; quantiles by linear interpolation within the containing bin."),
        "scope": "POOLED across datasets and languages (not per-dataset, not per-language).",
        "source": "traj2/stats/globalhist.npz, LAION-TTS annotated corpus v1",
        "n_rows": n_rows,
        "carried_from_reference": [
            "name", "group", "range", "reliability_reg_pearson", "synonyms"],
        "reference": os.path.basename(A.reference),
        "dimensions": len(stats),
        "not_measured_here": missing,
        "caveat_zero_inflated_emotions": (
            "Several emotion heads are extremely zero-inflated (Infatuation 87.7 % zeros, "
            "Awe 91.3 %, Pain 84.6 %), so no median/spread pair characterises them well. For "
            "the emotion clause prefer the percentile gate described in the README."),
    }
    doc = {"_meta": meta}
    doc.update(dict(sorted(stats.items())))
    json.dump(doc, open(A.out, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {A.out}: {len(stats)} dimensions over {n_rows:,} rows")
    if missing:
        print(f"NOT measured here ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
