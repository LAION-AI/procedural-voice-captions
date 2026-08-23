#!/usr/bin/env python
"""Stage 1: stratified sample of the raw emotion scores, and the diagnosis.

The 40 Empathic-Insight emotion heads are NOT on a common scale. `emo_Interest`
is never zero and sits near 2; `emo_Sadness` is zero on 87 % of clips. The
caption generator gates on an ABSOLUTE 1.0, so Interest clears it on essentially
every clip and Sadness never does -- the caption is reporting the scale of the
head, not the emotion of the clip.

This samples shards by STRIDE across the whole shard index of each dataset, so
the sample spans the speakers, languages and recording conditions in their real
proportions rather than whatever happens to sit at the start of the corpus, and
reports the composition so that claim is checkable.
"""
import argparse, collections, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pyarrow.parquet as pq

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
IDX = "/e/data1/datasets/products/TTS Training Data/laion-tts-annotated-v1/index"
D = f"{NB}/emonorm/out"
sys.path.insert(0, f"{NB}/traj2/code")
from common import cols

EMO, VN = cols("mls")
META = ["lang_iso", "top_emotion", "caption_general"]


def scan(arg):
    path, want_meta = arg
    try:
        have = set(pq.ParquetFile(path).schema_arrow.names)
        c = [x for x in EMO if x in have] + [x for x in (META if want_meta else []) if x in have]
        t = pq.read_table(path, columns=c)
    except Exception as e:
        return None
    d = t.to_pydict()
    n = len(d[c[0]])
    X = np.full((n, len(EMO)), np.nan, np.float32)
    for i, e in enumerate(EMO):
        if e in d:
            X[:, i] = np.asarray(d[e], np.float32)
    m = {}
    if want_meta:
        for k in META:
            if k in d:
                m[k] = d[k]
    return X, m, os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="emolia,eurospeech,vprof_vc,podcast,snippets,evasnippets,mls")
    ap.add_argument("--target", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=32)
    A = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    report = {}
    for ds in A.datasets.split(","):
        fs = sorted(glob.glob(f"{IDX}/{ds}/*.parquet"))
        if not fs:
            print(f"[{ds}] no index"); continue
        # rows per shard, from one file, to pick a stride that hits the target
        n1 = pq.ParquetFile(fs[0]).metadata.num_rows
        need = max(1, min(len(fs), int(np.ceil(A.target / max(n1, 1)))))
        step = max(1, len(fs) // need)
        pick = fs[::step][:need]
        t0 = time.time()
        Xs, langs, tops, caps, nsh = [], collections.Counter(), collections.Counter(), 0, 0
        with ProcessPoolExecutor(A.workers) as ex:
            for r in ex.map(scan, [(f, True) for f in pick], chunksize=4):
                if r is None:
                    continue
                X, m, _ = r
                Xs.append(X); nsh += 1
                for l in m.get("lang_iso", []) or []:
                    langs[str(l)] += 1
                for tp in m.get("top_emotion", []) or []:
                    tops[str(tp)] += 1
                for cp in m.get("caption_general", []) or []:
                    if cp and "interest" in cp.lower():
                        caps += 1
        if not Xs:
            continue
        X = np.concatenate(Xs, 0)
        np.save(f"{D}/raw_{ds}.npy", X.astype(np.float32))
        nt = sum(tops.values())
        rep = dict(
            shards_total=len(fs), shards_sampled=nsh, rows=int(len(X)),
            rows_per_shard=int(n1), wall_s=round(time.time() - t0, 1),
            langs=dict(langs.most_common(12)), n_langs=len(langs),
            top_emotion=dict(tops.most_common(8)),
            top_emotion_interest_pct=round(100 * tops.get("Interest", 0) / max(nt, 1), 2),
            caption_has_interest_pct=round(100 * caps / max(nt, 1), 2) if nt else None)
        report[ds] = rep
        print(f"[{ds}] {len(X):,} rows from {nsh}/{len(fs)} shards (stride {step}), "
              f"{len(langs)} languages, {time.time()-t0:.0f}s", flush=True)
        print(f"    top_emotion==Interest {rep['top_emotion_interest_pct']:.1f} % | "
              f"caption contains 'interest' {rep['caption_has_interest_pct']} %", flush=True)
    json.dump(dict(emo=EMO, datasets=report), open(f"{D}/sample_report.json", "w"), indent=1)
    print(f"\n-> {D}/sample_report.json")


if __name__ == "__main__":
    main()
