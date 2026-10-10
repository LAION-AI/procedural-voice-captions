#!/usr/bin/env python3
"""Batched, resumable Medium annotations in additive Parquet sidecars.

Sources are read by TAR byte address; their audio and existing metadata are
never rewritten. Every generated output carries the Medium namespace.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

from humaneness_backend import HumanenessMedium, MODEL_REPO, MODEL_REVISION, NAMESPACE, window_starts


def decode(row):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    with open(row["audio_tar"],"rb") as archive:
        archive.seek(row["audio_offset"]);raw=archive.read(row["audio_size"])
    if len(raw) != row["audio_size"]:
        raise IOError(f"Short source read: {row['id']}")
    wave,rate=sf.read(io.BytesIO(raw),dtype="float32",always_2d=True)
    wave=wave.mean(1)
    if rate!=16000:
        divisor=math.gcd(rate,16000)
        wave=resample_poly(wave,16000//divisor,rate//divisor).astype(np.float32)
    if len(wave)<1600 or not np.isfinite(wave).all():
        raise ValueError(f"Invalid source waveform: {row['id']}")
    return wave,hashlib.sha256(raw).hexdigest()


def annotate(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    model=HumanenessMedium(args.model_dir,device=args.device,threads=4,batch_size=args.batch_size)
    all_rows=[json.loads(line) for line in args.manifest.open()]
    rows=all_rows[args.rank::args.world_size]
    if args.limit:
        rows=rows[:args.limit]
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True)
    fields=[pa.field("sample_id",pa.string()),pa.field("source",pa.string()),pa.field("language",pa.string()),
        pa.field("source_key",pa.string()),pa.field("source_reference_json",pa.string()),pa.field("audio_sha256",pa.string())]
    fields += [pa.field(f"{NAMESPACE}__raw__{name}",pa.float32()) for name in model.names]
    fields += [pa.field(f"{NAMESPACE}__training_z__{name}",pa.float32()) for name in model.names]
    fields += [pa.field(f"{NAMESPACE}__{name}",pa.float32()) for name in ["duration_s","characters_per_second","estimated_burst_count"]]
    fields += [pa.field(f"{NAMESPACE}__orange_timbre_128d",pa.list_(pa.float32(),128)),
        pa.field(f"{NAMESPACE}__orange_identity_250d",pa.list_(pa.float32(),250)),
        pa.field(f"{NAMESPACE}__frame_probabilities_20ms",pa.list_(pa.float32())),
        pa.field(f"{NAMESPACE}__burst_events_json",pa.string()),pa.field(f"{NAMESPACE}__n_windows",pa.int32())]
    schema=pa.schema(fields,metadata={b"model_repo":MODEL_REPO.encode(),b"model_revision":MODEL_REVISION.encode(),
        b"namespace":NAMESPACE.encode(),b"manifest_sha256":manifest_hash.encode(),
        b"original_sources_unchanged":b"true",b"frame_step_seconds":b"0.02"})
    started=time.monotonic();completed=0;processed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for first in range(0,len(rows),args.segment_size):
            part=first//args.segment_size
            path=args.out/f"raw-r{args.rank:02d}-{part:05d}.parquet"
            current=rows[first:first+args.segment_size]
            if path.exists():
                existing=pq.ParquetFile(path)
                if (existing.metadata.num_rows!=len(current)
                    or existing.schema_arrow.metadata.get(b"manifest_sha256")!=manifest_hash.encode()
                    or existing.schema_arrow.metadata.get(b"model_revision")!=MODEL_REVISION.encode()):
                    raise RuntimeError(f"Existing sidecar does not match frozen run: {path}")
                completed+=len(current)
                continue
            decoded=list(executor.map(decode,current))
            tasks=[]
            for i,(wave,digest) in enumerate(decoded):
                for s in window_starts(len(wave)):
                    tasks.append((i,s,wave[s:s+480000]))
            tasks.sort(key=lambda t:len(t[2]))
            windows=[[] for _ in current]
            for j in range(0,len(tasks),args.batch_size):
                batch=tasks[j:j+args.batch_size]
                predictions=model.predict_batch([t[2] for t in batch])
                for (i,start,_),prediction in zip(batch,predictions):
                    windows[i].append((start,prediction))
            records=[]
            for i,(row,(wave,digest)) in enumerate(zip(current,decoded)):
                windows[i].sort(key=lambda x:x[0])
                prediction=model.combine_window_predictions(len(wave),[s for s,_ in windows[i]],[p for _,p in windows[i]])
                record={"sample_id":row["id"],"source":row["source"],"language":row["language"],
                    "source_key":row["source_key"],"source_reference_json":json.dumps(row,ensure_ascii=False,separators=(",",":")),
                    "audio_sha256":digest}
                record.update({f"{NAMESPACE}__raw__{name}":prediction["scores_raw"][name] for name in model.names})
                record.update({f"{NAMESPACE}__training_z__{name}":prediction["scores_training_z"][name] for name in model.names})
                for name in ["duration_s","characters_per_second","estimated_burst_count","orange_timbre_128d","orange_identity_250d","n_windows"]:
                    record[f"{NAMESPACE}__{name}"]=prediction[name]
                record[f"{NAMESPACE}__frame_probabilities_20ms"]=prediction["burst_probability_per_20ms_frame"]
                record[f"{NAMESPACE}__burst_events_json"]=json.dumps(prediction["burst_event_proposals"],separators=(",",":"))
                records.append(record)
            table=pa.Table.from_pylist(records,schema=schema)
            temporary=path.with_suffix(".parquet.tmp")
            pq.write_table(table,temporary,compression="zstd",row_group_size=256)
            temporary.replace(path)
            completed+=len(current)
            processed+=len(current)
            elapsed=time.monotonic()-started
            progress={"rank":args.rank,"completed":completed,"total":len(rows),"elapsed_s":elapsed,
                "processed_in_this_process":processed,
                "samples_per_second":processed/max(elapsed,.001),"updated_unix":time.time()}
            (args.out/f"progress-r{args.rank:02d}.json").write_text(json.dumps(progress,indent=2)+"\n")
            print("ANNOTATED",json.dumps(progress),flush=True)
    done={"rank":args.rank,"n":len(rows),"manifest_sha256":manifest_hash,"model_repo":MODEL_REPO,"model_revision":MODEL_REVISION}
    (args.out/f"DONE-r{args.rank:02d}.json").write_text(json.dumps(done,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["manifest","out","model-dir"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--rank",type=int,default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--world-size",type=int,default=1)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--segment-size",type=int,default=512)
    parser.add_argument("--limit",type=int,default=0)
    annotate(parser.parse_args())
