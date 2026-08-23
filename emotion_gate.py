#!/usr/bin/env python3
"""Percentile gate for the emotion clause. Stdlib only (bisect + json).

WHY NOT A Z-SCORE
-----------------
`caption.py` ranks emotions by |z| = |value - median| / spread. That assumes each
emotion head has a median and a width worth dividing by. Measured over the full
LAION-TTS corpus (132,833,726 utterances) most of them do not:

    Awe            94.4 % of clips at or below zero, IQR 0.0002
    Infatuation    90.8 %                            IQR 0.0071
    Sadness        85.5 %
    Interest        0.0 %                            median 2.01

The heads are not on a common scale, so ANY absolute or width-normalised threshold
ranks them by their own scale rather than by the clip. With an absolute 1.0 gate,
measured across 165,516,420 regenerated captions, `Interest` was named on 90.6 % of
all rows and `Bitterness` on 0.1 %.

A percentile makes "reads as sadness" mean "unusually sad for this corpus", which is
what a reader already assumes it means, and needs no distributional assumption.

USAGE
    from emotion_gate import EmotionGate
    g = EmotionGate()                      # loads emotion_percentiles.json
    g.percentile("Sadness", 0.42)          -> 0.93
    g.select({"Sadness": 0.42, "Interest": 2.1, ...})
                                           -> ["sadness"]  (or [] => no dominant emotion)
"""
import bisect, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "emotion_percentiles.json")

FLOOR = float(os.environ.get("PVC_EMO_PERCENTILE_FLOOR", "0.90"))
TOP_N = int(os.environ.get("PVC_EMO_TOP_N", "3"))
NONE_CLAUSE = "no dominant emotion"


class EmotionGate:
    def __init__(self, path=None):
        d = json.load(open(path or DEFAULT_PATH))
        self.meta = d.pop("_meta")
        self.knots_p = self.meta["knots_p"]
        self.emo = d
        self.n_rows = self.meta["n_rows"]

    def percentile(self, name, value):
        """Empirical percentile of `value` for emotion `name`, in [0, 1].

        Ties matter here: when a head is zero on most of the corpus, every zero shares
        one percentile. Interpolating between knots that hold the same value would
        invent an ordering inside the tie, so the LOWEST knot index carrying this value
        wins -- a clip sitting exactly on the pile is not "above" the pile.
        """
        e = self.emo.get(name)
        if e is None:
            return None
        v = e["knots_v"]
        i = bisect.bisect_left(v, value)
        if i <= 0:
            return self.knots_p[0]
        if i >= len(v):
            return self.knots_p[-1]
        if v[i] == value:
            return self.knots_p[i]
        lo_v, hi_v = v[i - 1], v[i]
        if hi_v <= lo_v:
            return self.knots_p[i]
        f = (value - lo_v) / (hi_v - lo_v)
        return self.knots_p[i - 1] + f * (self.knots_p[i] - self.knots_p[i - 1])

    def threshold(self, name, p=FLOOR):
        e = self.emo.get(name)
        return None if e is None else e["thresholds"].get(f"{p:.2f}")

    def select(self, scores, floor=FLOOR, top_n=TOP_N):
        """scores: {emotion_name: raw_value} -> list of names in the top (1-floor) tail,
        highest percentile first, at most top_n. Empty list means no dominant emotion."""
        ranked = []
        for k, val in scores.items():
            if val is None:
                continue
            u = self.percentile(k, float(val))
            if u is not None and u >= floor:
                ranked.append((u, k))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        return [k for _, k in ranked[:top_n]]


if __name__ == "__main__":
    g = EmotionGate()
    print(f"{len(g.emo)} emotions, ECDF over {g.n_rows:,} utterances")
    print(f"gate: percentile >= {FLOOR}, at most {TOP_N}, else {NONE_CLAUSE!r}\n")
    for k in ["Interest", "Concentration", "Awe", "Sadness", "Infatuation"]:
        print(f"  {k:14s} p90 threshold = {g.threshold(k):8.4f}   "
              f"percentile(1.0) = {g.percentile(k, 1.0):.3f}")
