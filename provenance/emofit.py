#!/usr/bin/env python
"""Stage 2: diagnose the scale problem, fit the caption normaliser, tune the gate.

NORMALISATION SCOPE -- the decision and why.
The corpus already carries `traj2/stats/globalhist.npz`: an EXACT tie-aware
mid-rank ECDF over 130,785,282 utterances from 8 datasets and every language in
the corpus. That is the same estimator this task calls for and 60x more data than
a few-million sample, so it is reused rather than re-estimated. Two changes:

  * `vprof_vc` was generated after those statistics and is absent from them, so
    its histogram is fitted here and merged into the SAME bins. The shift this
    causes to every other dataset's percentiles is measured and reported, because
    the trajectory miner normalises with the unmerged globalhist and any drift
    between the two would make the caption and the miner disagree about a chain.
  * POOLED, not per-dataset. Per-dataset normalisation would force every corpus
    to the same emotional profile by construction, erasing real differences
    (evasnippets is a deliberately bucket-stratified reference set and genuinely
    runs hotter). Pooled also keeps the caption on the exact scale the miner
    selects trajectories with, which is what makes "reads as X" and "this chain
    ramps X" refer to the same quantity. The per-dataset alternative is measured
    below so the choice is evidenced rather than asserted.
"""
import glob, json, os, sys
import numpy as np

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
IDX = "/e/data1/datasets/products/TTS Training Data/laion-tts-annotated-v1/index"
D = f"{NB}/emonorm/out"
sys.path.insert(0, f"{NB}/traj2/code")
from common import cols
from normhist import NormHist

EMO, VN = cols("mls")
NBIN = 4096


def midrank_table(hist):
    n = np.maximum(hist.sum(1, keepdims=True), 1.0)
    below = np.cumsum(hist, axis=1) - hist
    return ((below + 0.5 * hist) / n).astype(np.float32)


def apply_tab(X, tab, lo, hi, nbin=NBIN):
    w = (hi - lo) / nbin
    k = np.floor((X - lo) / w) + 1.0
    np.clip(k, 0, nbin + 1, out=k)
    k = k.astype(np.int32)
    Y = np.empty_like(X, np.float32)
    for d in range(X.shape[1]):
        Y[:, d] = tab[d][k[:, d]]
    return Y


def main():
    z = np.load(f"{NB}/traj2/stats/globalhist.npz", allow_pickle=True)
    allf = [str(x) for x in z["fields"]]
    gi = [allf.index(e) for e in EMO]
    GH = z["hist"][gi].astype(np.int64)
    LO, HI = z["lo"][gi].astype(np.float64), z["hi"][gi].astype(np.float64)
    print(f"globalhist: {GH.sum(1)[0]:,} rows, {len(EMO)} emotion fields")

    samples = {}
    for f in sorted(glob.glob(f"{D}/raw_*.npy")):
        ds = os.path.basename(f)[4:-4]
        samples[ds] = np.load(f)
    print("samples: " + ", ".join(f"{k} {len(v):,}" for k, v in samples.items()))

    # ---------------- the diagnosis ----------------
    diag = {}
    pool = np.concatenate(list(samples.values()), 0)
    for i, e in enumerate(EMO):
        v = pool[:, i]
        v = v[np.isfinite(v)]
        diag[e] = dict(median=round(float(np.median(v)), 4),
                       zeros_pct=round(float((v <= 0).mean() * 100), 2),
                       p90=round(float(np.percentile(v, 90)), 4),
                       mean=round(float(v.mean()), 4))
    worst = sorted(diag.items(), key=lambda x: -x[1]["median"])
    print("\nraw scale, pooled sample (the defect):")
    for e, s in worst[:4] + worst[-4:]:
        print(f"  {e[4:]:34s} median {s['median']:6.3f}  zeros {s['zeros_pct']:5.1f} %  "
              f"p90 {s['p90']:6.3f}")

    # ---------------- fit vprof and merge ----------------
    vp = samples.get("vprof_vc")
    VH = np.zeros_like(GH)
    if vp is not None:
        w = (HI - LO) / NBIN
        for d in range(len(EMO)):
            k = np.floor((vp[:, d].astype(np.float64) - LO[d]) / w[d]) + 1.0
            np.clip(k, 0, NBIN + 1, out=k)
            VH[d] = np.bincount(k.astype(np.int64), minlength=NBIN + 2)[:NBIN + 2]
        print(f"\nvprof_vc histogram fitted from {len(vp):,} rows "
              f"({VH.sum(1)[0]/GH.sum(1)[0]*100:.2f} % of the pooled corpus)")
    MH = GH + VH
    TAB_G, TAB_M = midrank_table(GH.astype(np.float64)), midrank_table(MH.astype(np.float64))

    # how far does adding vprof move everyone else's percentiles?
    probe = pool[np: len(pool): max(1, len(pool) // 200_000)] if False else pool[::max(1, len(pool)//200_000)]
    UG = apply_tab(probe.astype(np.float32), TAB_G, LO.astype(np.float32), HI.astype(np.float32))
    UM = apply_tab(probe.astype(np.float32), TAB_M, LO.astype(np.float32), HI.astype(np.float32))
    dd = np.abs(UM - UG)
    shift = dict(median=round(float(np.median(dd)), 5), p95=round(float(np.percentile(dd, 95)), 5),
                 max=round(float(dd.max()), 5), n=int(probe.size))
    print(f"merging vprof shifts percentiles by: median {shift['median']:.5f}, "
          f"p95 {shift['p95']:.5f}, max {shift['max']:.5f}")

    # ---------------- pooled vs per-dataset, the scope evidence ----------------
    scope = {}
    for ds, X in samples.items():
        U = apply_tab(X.astype(np.float32), TAB_M, LO.astype(np.float32), HI.astype(np.float32))
        # under a per-dataset norm every dataset's median would be 0.5 by construction
        scope[ds] = dict(
            median_pct=round(float(np.median(U)), 4),
            mean_pct=round(float(U.mean()), 4),
            frac_any_above_85=round(float((U >= 0.85).any(1).mean()), 4))
    print("\npooled-norm median percentile per dataset "
          "(0.500 would mean the dataset IS the corpus average):")
    for ds, s in sorted(scope.items(), key=lambda x: -x[1]["median_pct"]):
        print(f"  {ds:12s} median {s['median_pct']:.3f}  mean {s['mean_pct']:.3f}  "
              f"clips with some emotion >=0.85: {s['frac_any_above_85']*100:5.1f} %")

    # ---------------- tune the gate ----------------
    print("\ngate tuning on the pooled sample (top-3 by percentile, floor U):")
    tune = {}
    sub = pool[::max(1, len(pool) // 400_000)].astype(np.float32)
    US = apply_tab(sub, TAB_M, LO.astype(np.float32), HI.astype(np.float32))
    for U in (0.80, 0.85, 0.90, 0.93):
        n_named = (US >= U).sum(1)
        named = np.minimum(n_named, 3)
        tune[f"{U:.2f}"] = dict(
            none_pct=round(float((n_named == 0).mean() * 100), 2),
            mean_named=round(float(named.mean()), 3),
            interest_pct=round(float((US[:, EMO.index("emo_Interest")] >= U).mean() * 100), 2))
        t = tune[f"{U:.2f}"]
        print(f"  U={U:.2f}  no emotion named {t['none_pct']:5.2f} %  "
              f"mean named {t['mean_named']:.2f}  Interest named {t['interest_pct']:5.2f} %")

    np.savez_compressed(f"{D}/capnorm.npz", fields=np.array(EMO), hist=MH,
                        lo=LO, hi=HI, n=MH.sum(1))
    json.dump(dict(
        source="traj2/stats/globalhist.npz (130,785,282 rows, 8 datasets) + vprof_vc fitted here",
        n_global=int(GH.sum(1)[0]), n_vprof=int(VH.sum(1)[0]), n_total=int(MH.sum(1)[0]),
        estimator="tie-aware mid-rank ECDF, 4096 bins (NormHist 'ecdfm')",
        scope="POOLED across datasets and languages",
        vprof_merge_shift=shift, per_dataset=scope, diagnosis=diag, gate_tuning=tune,
        sample_rows={k: int(len(v)) for k, v in samples.items()}),
        open(f"{D}/capnorm.json", "w"), indent=1)
    print(f"\n-> {D}/capnorm.npz + capnorm.json")


if __name__ == "__main__":
    main()
