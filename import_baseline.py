#!/usr/bin/env python3
"""Convert an out-of-repo baseline measurement into this repo's `baseline_stats.json` schema.

`compute_baseline.py` builds the *emolia* baseline. A baseline can also be measured on
any other corpus — this script takes such a measurement, written as

    {"stats": {DIM: {mean, median, std, mad, spread, p10, p90, n, name, group,
                     range, reliability_reg_pearson, source}, ...},
     "meta":  {...}}

and normalises it into the flat `{DIM: {...}, "_meta": {...}}` layout that
`caption.load_baseline()` expects. Four things have to be reconciled, and all four were
real defects in the first DramaBox measurement:

1. **EmoNet key names.** A scoring run keyed its emotions by the Empathic-Insight *file*
   stem (`Astonishment_Surprise`, `Hope_Enthusiasm_Optimism`, …). This repo — and
   `burst_captions.EMOS` — uses the taxonomy names (`Astonishment/Surprise`,
   `Hope/Optimism`, …). Unmapped, 10 of the 40 emotions silently kept their *old*
   emolia statistics while a dead duplicate entry sat next to them with `group: ""`,
   which `caption.py` ignores. The mapping is inverted from `burst_captions._emo_file()`,
   so it cannot drift from the loader.
2. **`vocalburst_blend` vs `blend`.** Same story for the vocal-burst blend head.
3. **Synonym clusters.** The measurement carries no `synonyms`, but the captioner needs
   them to word emotions. They are carried over per emotion from the reference baseline.
4. **`group` / `range` / `name`.** Filled from the reference baseline when absent.

Nothing is invented: every field is either measured, or copied from the reference
baseline and marked with `source`.

    python import_baseline.py measured.json --out baseline_stats.json \
        --reference baseline_stats_emolia.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from burst_captions import EMOS, _emo_file  # noqa: E402  (the loader's own name -> file rule)

# Inverse of the loader's name -> checkpoint-file rule, so a scoring run that keyed its
# output by file stem can be mapped back onto the taxonomy names without a hand list.
FILE_STEM_TO_EMO = {_emo_file(e)[len("model_"):-len("_best.pth")]: e for e in EMOS}
# Non-emotion aliases used by out-of-repo scorers.
ALIASES = {"vocalburst_blend": "blend", "genu": "genuineness"}

NUMERIC = ("mean", "median", "std", "mad", "spread", "p10", "p90")


def canonical(key):
    if key in ALIASES:
        return ALIASES[key]
    return FILE_STEM_TO_EMO.get(key, key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measured", help="JSON from the out-of-repo baseline run")
    ap.add_argument("--reference", default=os.path.join(HERE, "baseline_stats_emolia.json"),
                    help="baseline supplying synonyms/group/range and any dim not measured")
    ap.add_argument("--out", default=os.path.join(HERE, "baseline_stats.json"))
    ap.add_argument("--description", default="", help="free-text provenance for _meta")
    A = ap.parse_args()

    raw = json.load(open(A.measured, encoding="utf-8"))
    measured = raw.get("stats", raw)
    src_meta = raw.get("meta", {})
    ref = json.load(open(A.reference, encoding="utf-8"))
    ref_meta = ref.get("_meta", {})

    # A measurement can contain the SAME dimension twice — once freshly measured under a
    # raw key and once passed through from its own reference under the canonical key
    # (that is exactly what happened to the 10 renamed emotions and to `blend`). Group the
    # candidates by canonical name first and keep the freshly measured one; never let
    # ordering decide.
    def is_passthrough(st):
        return str(st.get("source", "")).startswith("carried_over")

    cands = {}
    renamed = {}
    for key, st in measured.items():
        if key == "_meta" or not isinstance(st, dict) or "median" not in st:
            continue
        name = canonical(key)
        if name != key:
            renamed[key] = name
        cands.setdefault(name, []).append((key, st))

    out, dropped = {}, []
    for name, lst in cands.items():
        fresh = [c for c in lst if not is_passthrough(c[1])]
        key, st = (fresh or lst)[0]
        dropped += [k for k, _ in lst if k != key]
        r = ref.get(name, {})
        e = {k: st[k] for k in NUMERIC if isinstance(st.get(k), (int, float))}
        e["n"] = int(st.get("n") or r.get("n") or 0)
        e["name"] = st.get("name") if (st.get("name") and st.get("name") != key) else r.get("name", name)
        e["group"] = st.get("group") or r.get("group") or ""
        if r.get("range") is not None or st.get("range") is not None:
            e["range"] = st.get("range") if st.get("range") is not None else r.get("range")
        if st.get("reliability_reg_pearson") is not None:
            e["reliability_reg_pearson"] = st["reliability_reg_pearson"]
        elif r.get("reliability_reg_pearson") is not None:
            e["reliability_reg_pearson"] = r["reliability_reg_pearson"]
        if r.get("synonyms"):
            e["synonyms"] = list(r["synonyms"])          # wording clusters are not measured
        e["source"] = st.get("source") or "measured"
        out[name] = e

    # Any dimension the reference has and the measurement does not keeps its reference
    # entry, flagged, so the file stays a drop-in replacement rather than losing dims.
    carried = []
    for name, r in ref.items():
        if name == "_meta" or not isinstance(r, dict) or name in out:
            continue
        e = dict(r)
        e["source"] = f"carried_over_from:{os.path.basename(A.reference)}"
        out[name] = e
        carried.append(name)

    groups = {}
    for e in out.values():
        groups.setdefault(e.get("group") or "?", 0)
        groups[e.get("group") or "?"] += 1

    meta = dict(ref_meta)
    meta["description"] = (A.description or ref_meta.get("description", ""))
    meta["imported_from"] = {"file": os.path.basename(A.measured), **src_meta}
    meta["renamed_keys"] = renamed
    meta["carried_over"] = carried
    meta["counts"] = groups
    out["_meta"] = meta

    with open(A.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, sort_keys=True)

    print(f"wrote {A.out}: {len(out) - 1} dimensions {groups}")
    print(f"renamed {len(renamed)}: {renamed}")
    print(f"carried over from reference ({len(carried)}): {carried}")
    if dropped:
        print(f"dropped duplicate raw keys: {dropped}")


if __name__ == "__main__":
    main()
