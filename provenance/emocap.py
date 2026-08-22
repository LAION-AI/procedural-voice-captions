#!/usr/bin/env python
"""Stage 3: re-derive the emotion clause of every caption from NORMALISED values.

The old clause named an emotion when its raw score cleared an ABSOLUTE 1.0. The
40 heads are not on a common scale -- Interest has median 2.08 and never hits
zero, Infatuation has median -0.02 and is zero on 87.7 % of clips -- so that gate
named Interest on 94 % of clips and Sadness on almost none. It was reporting the
scale of the head.

The new clause names an emotion when it is in the top 10 % FOR THAT EMOTION,
against the pooled tie-aware mid-rank ECDF in capnorm.npz. "reads as sadness"
now means "unusually sad for this corpus", which is what a reader assumes it
means. A clip that is ordinary on all 40 gets an explicit "no dominant emotion"
rather than being forced to pick one.

Only the emotion clause is touched. The GEND/BKGN polarity repair, the delivery,
timbre, speech, affect, style and recording clauses and the burst handling are
left exactly as they are.
"""
import argparse, collections, glob, json, os, re, shutil, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pyarrow.parquet as pq

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
IDX = "/e/data1/datasets/products/TTS Training Data/laion-tts-annotated-v1/index"
D = f"{NB}/emonorm/out"
sys.path.insert(0, f"{NB}/traj2/code")
from common import cols
import gridcaption as GC

EMO, _ = cols("mls")
U_FLOOR = 0.90          # top 10 % for that emotion
TOP_N = 3
NBIN = 4096
NONE_CLAUSE = "no dominant emotion"


def pretty(e):
    return e[4:].replace("_", " ").replace("/", " or ").lower()


def scan(arg):
    path, want = arg
    try:
        have = set(pq.ParquetFile(path).schema_arrow.names)
        c = ["uid"] + [e for e in EMO if e in have]
        t = pq.read_table(path, columns=c).to_pydict()
    except Exception:
        return {}
    out = {}
    for i, u in enumerate(t["uid"]):
        if u in want:
            out[u] = [float(t[e][i]) if e in t else np.nan for e in EMO]
    return out


class Norm:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        f = [str(x) for x in z["fields"]]
        assert f == EMO, "capnorm field order must match the index column order"
        h = z["hist"].astype(np.float64)
        n = np.maximum(h.sum(1, keepdims=True), 1.0)
        below = np.cumsum(h, axis=1) - h
        self.tab = ((below + 0.5 * h) / n).astype(np.float32)
        self.lo, self.hi = z["lo"].astype(np.float64), z["hi"].astype(np.float64)
        self.n = int(z["n"][0])

    def u(self, X):
        w = (self.hi - self.lo) / NBIN
        k = np.floor((np.asarray(X, np.float64) - self.lo) / w) + 1.0
        np.clip(k, 0, NBIN + 1, out=k)
        k = k.astype(np.int32)
        Y = np.empty(k.shape, np.float32)
        for d in range(k.shape[1]):
            Y[:, d] = self.tab[d][k[:, d]]
        return Y


READS = re.compile(r"(^|;\s*)reads as [^;]*", re.I)


def rewrite(cap, names):
    """Swap the emotion clause. Everything else in the caption is untouched."""
    new = ("reads as " + ", ".join(names)) if names else NONE_CLAUSE
    if READS.search(cap or ""):
        return READS.sub(lambda m: (m.group(1) or "") + new, cap, count=1)
    if not cap:
        return new
    # no emotion clause existed: put it where procedural_caption would have,
    # immediately before the genuineness figure
    parts = [c.strip() for c in cap.rstrip(".").split(";")]
    for i, c in enumerate(parts):
        if c.lower().startswith("genuineness"):
            parts.insert(i, new)
            break
    else:
        parts.append(new)
    return "; ".join(parts) + ("." if cap.rstrip().endswith(".") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=U_FLOOR)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    A = ap.parse_args()

    targets = [
        ("grid+vc", f"{NB}/traj2/out/gridcaption.json"),
        ("strict", f"{NB}/strict/out/caption.json"),
    ]
    need = collections.defaultdict(set)
    for _, p in targets:
        for key in json.load(open(p))["clips"]:
            ds, u = key.split("|", 1)
            need[ds].add(u)
    print("uids needed: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(need.items())))

    raw = {}
    for ds, want in sorted(need.items()):
        fs = sorted(glob.glob(f"{IDX}/{ds}/*.parquet"))
        found = {}
        with ProcessPoolExecutor(A.workers) as ex:
            for r in ex.map(scan, [(f, want) for f in fs], chunksize=8):
                found.update(r)
                if len(found) >= len(want):
                    break
        for u, v in found.items():
            raw[f"{ds}|{u}"] = v
        print(f"[{ds}] raw emotion scores for {len(found)}/{len(want)}", flush=True)

    N = Norm(f"{D}/capnorm.npz")
    keys = sorted(raw)
    X = np.array([raw[k] for k in keys], np.float64)
    U = N.u(np.nan_to_num(X, nan=0.0))
    print(f"normalised {len(keys)} clips against {N.n:,} rows", flush=True)

    before, after, nnone, hist_n = collections.Counter(), collections.Counter(), 0, collections.Counter()
    newcap = {}
    for i, k in enumerate(keys):
        u = U[i]
        order = np.argsort(-u)
        names = [pretty(EMO[j]) for j in order[:A.top] if u[j] >= A.floor]
        if not names:
            nnone += 1
        hist_n[len(names)] += 1
        for nm in names:
            after[nm] += 1
        newcap[k] = names

    changed = 0
    for tag, path in targets:
        C = json.load(open(path))
        if not A.dry_run and not os.path.exists(path + ".bak-absthr"):
            shutil.copy2(path, path + ".bak-absthr")
        n_t = 0
        for key, v in C["clips"].items():
            old = v.get("caption") or ""
            m = READS.search(old)
            if m:
                for nm in [x.strip() for x in
                           re.sub(r"^;?\s*reads as ", "", m.group(0).strip(), flags=re.I).split(",")]:
                    if nm:
                        before[nm] += 1
            if key not in newcap:
                continue
            cap = rewrite(old, newcap[key])
            v["caption"] = cap
            v["parsed"] = GC.parse_caption(cap)
            v["emo_named"] = newcap[key]
            v["emo_none"] = not newcap[key]
            n_t += 1
            changed += 1
        C.setdefault("stats", {})
        C["stats"]["emotion_clause"] = dict(
            normaliser="emonorm/out/capnorm.npz", floor=A.floor, top_n=A.top,
            rule=f"named when in the top {100*(1-A.floor):.0f} % for that emotion, "
                 f"pooled tie-aware mid-rank ECDF over {N.n:,} utterances",
            none_clause=NONE_CLAUSE)
        if not A.dry_run:
            json.dump(C, open(path, "w"))
        print(f"[{tag}] rewrote {n_t} captions -> {path}")

    nb, na = sum(before.values()), sum(after.values())
    print(f"\nBEFORE (absolute 1.0 gate): {nb:,} emotion mentions over "
          f"{len(keys):,} clips")
    for e, c in before.most_common(8):
        print(f"   {e:26s} {c:6,}  {c/len(keys)*100:5.1f} % of clips")
    print(f"\nAFTER (top {100*(1-A.floor):.0f} % percentile gate): {na:,} mentions")
    for e, c in after.most_common(12):
        print(f"   {e:26s} {c:6,}  {c/len(keys)*100:5.1f} % of clips")
    print(f"\nclips naming no emotion: {nnone:,} ({nnone/len(keys)*100:.1f} %)")
    print("emotions named per clip:", dict(sorted(hist_n.items())))
    print(f"distinct emotions ever named: before {len(before)}, after {len(after)}")
    never = [pretty(e) for e in EMO if pretty(e) not in after]
    print(f"never named after ({len(never)}): {', '.join(never) if never else 'none'}")
    json.dump(dict(before=dict(before), after=dict(after), n_clips=len(keys),
                   none_pct=round(nnone / len(keys) * 100, 2),
                   per_clip=dict(sorted(hist_n.items())), never_named=never,
                   floor=A.floor, top_n=A.top),
              open(f"{D}/caption_shift.json", "w"), indent=1)


if __name__ == "__main__":
    main()
