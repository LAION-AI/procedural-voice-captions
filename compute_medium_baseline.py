#!/usr/bin/env python3
"""Measured prediction means/stds and additive reference z-score sidecars."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

from humaneness_backend import EMOTION_ALIASES, MODEL_REPO, MODEL_REVISION, NAMESPACE


def summarize(values):
    import numpy as np
    result=[]
    for column in values.T:
        valid=column[np.isfinite(column)]
        n=len(valid)
        mean=float(valid.mean(dtype=np.float64)) if n else None
        std=float(valid.std(dtype=np.float64,ddof=0)) if n else None
        result.append({"n":n,"mean":mean,"std":std,"median":float(np.median(valid)) if n else None,
            "z_defined":bool(n and std>1e-6),"normalization":"mean_std"})
    return result


def compute(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    files=sorted(args.sidecars.glob("raw-r*.parquet"))
    if not files:
        raise RuntimeError("No measured Medium sidecars; no statistics will be invented")
    info=json.loads(args.reference_info.read_text())
    schema=pq.ParquetFile(files[0]).schema_arrow
    raw_columns=[k for k in schema.names if k.startswith(NAMESPACE+"__raw__")]
    targets=[k.split("__raw__",1)[1] for k in raw_columns]
    auxiliaries=[NAMESPACE+"__characters_per_second",NAMESPACE+"__estimated_burst_count"]
    columns=raw_columns+auxiliaries
    source,language,ids,sha,matrices=[],[],[],[],[]
    boundary_rounding_clamps=0
    sidecar_inventory=[]
    for path in files:
        table=pq.read_table(path,columns=["sample_id","source","language","audio_sha256",
            NAMESPACE+"__duration_s",NAMESPACE+"__burst_events_json"]+columns)
        metadata=table.schema.metadata or {}
        if metadata.get(b"model_revision")!=MODEL_REVISION.encode() or metadata.get(b"manifest_sha256")!=info["manifest_sha256"].encode():
            raise ValueError(f"Mixed checkpoint or source manifests: {path}")
        ids.extend(table["sample_id"].to_pylist());source.extend(table["source"].to_pylist())
        language.extend(table["language"].to_pylist());sha.extend(table["audio_sha256"].to_pylist())
        matrices.append(np.stack([table[c].to_numpy().astype(np.float64) for c in columns],axis=1))
        for duration,raw_events in zip(table[NAMESPACE+"__duration_s"].to_pylist(),table[NAMESPACE+"__burst_events_json"].to_pylist()):
            for event in json.loads(raw_events):
                if not 0<=event["start_s"]<event["end_s"]<=duration+1e-5:
                    raise ValueError(f"Event outside its audio boundaries: {path}")
                boundary_rounding_clamps+=event["end_s"]>duration
        sidecar_inventory.append({"file":path.name,"rows":len(table),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(ids)!=info["n"] or len(set(ids))!=len(ids):
        raise ValueError(f"Incomplete or duplicated sample coverage: {len(ids)}/{info['n']}")
    values=np.concatenate(matrices);stats=summarize(values)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model predictions: inspect before building calibration")
    names=targets+["characters_per_second","estimated_burst_count"]
    measured=dict(zip(names,stats))
    old=json.loads((Path(__file__).parent/"baseline_stats.json").read_text())
    result={}
    for target,stat in zip(names,stats):
        if target.startswith("emo_"):
            key=EMOTION_ALIASES.get(target[4:],target[4:].replace("_"," "));group="emonet"
        elif target.startswith("vn_") and target.endswith("_reg"):
            key=target[3:-4];group="voicenet"
        elif target in {"genuineness_0_6","blend_0_10","audiobox:CE"}:
            key={"genuineness_0_6":"genuineness","blend_0_10":"blend","audiobox:CE":"content_enjoyment"}[target]
            group="quality"
        else:
            continue
        wording=old.get(key,{})
        result[key]=dict(stat,name=wording.get("name",key),group=group,
            target_name=target,source="Humaneness Ears Medium predictions over frozen S10 + Gemini acting + multilingual Emilia")
        if "synonyms" in wording:result[key]["synonyms"]=wording["synonyms"]
        if "range" in wording:result[key]["teacher_ordinal_reference_range"]=wording["range"]
    # These are descriptive corpus statistics, not an independent evaluation.
    result["_provenance"]={"schema":"humaneness-medium-procedural-mean-std-v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),"model_repo":MODEL_REPO,
        "model_revision":MODEL_REVISION,"model_variant":"medium","namespace":NAMESPACE,
        "formula":"z = (raw prediction - arithmetic mean) / population standard deviation",
        "std_ddof":0,"weighting":"equal weight per successfully annotated clip",
        "prediction_storage":"float32; statistics calculated in float64",
        "n":len(ids),"source_counts":dict(collections.Counter(source)),
        "language_counts":dict(collections.Counter(language)),"source_manifest":info,
        "unique_audio_sha256_count":len(set(sha)),
        "interpretation":"Reference-relative model deviation; not an emotion probability or population-unbiased survey",
        "old_teacher_correlations_reused":False,"ranking_reliability_default":1.0,
        "pleasantness_target":"audiobox:CE; VoiceNet ESTH remains in raw outputs only",
        "original_annotations_preserved":True,"coverage_verified":True,
        "derived_event_boundary_rounding_clamps":boundary_rounding_clamps,
        "event_boundary_rule":"Clamp sub-sample end-time rounding to stored float32 duration in derived sidecars; preserve original end_s in end_s_before_boundary_clamp"}
    result["_raw_targets"]=measured
    result["_by_source"]={s:dict(zip(names,summarize(values[np.asarray(source)==s]))) for s in sorted(set(source))}
    result["_by_language"]={l:dict(zip(names,summarize(values[np.asarray(language)==l]))) for l in sorted(set(language))}
    result["_sidecar_inventory"]=sidecar_inventory
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    baseline_sha=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.normalized_out.mkdir(parents=True,exist_ok=True)
    for path in files:
        table=pq.read_table(path)
        events_column=NAMESPACE+"__burst_events_json"
        repaired=[]
        for duration,raw_events in zip(table[NAMESPACE+"__duration_s"].to_pylist(),table[events_column].to_pylist()):
            events=json.loads(raw_events)
            for event in events:
                if event["end_s"]>duration:
                    event["end_s_before_boundary_clamp"]=event["end_s"]
                    event["end_s"]=duration
            repaired.append(json.dumps(events,separators=(",",":")))
        table=table.set_column(table.schema.get_field_index(events_column),events_column,pa.array(repaired,type=pa.string()))
        for target in names:
            col=(NAMESPACE+"__raw__"+target if target in targets else NAMESPACE+"__"+target)
            stat=measured[target]
            raw=table[col].to_numpy().astype(np.float64)
            z=(raw-stat["mean"])/stat["std"] if stat["z_defined"] else [None]*len(raw)
            table=table.append_column(NAMESPACE+"__reference_z__"+target,pa.array(z,type=pa.float32()))
        metadata=dict(table.schema.metadata or {})
        metadata[b"normalization_sha256"]=baseline_sha.encode()
        metadata[b"reference_z_formula"]=b"(raw - arithmetic mean) / population standard deviation"
        table=table.replace_schema_metadata(metadata)
        output=args.normalized_out/path.name.replace("raw-","annotated-",1)
        temporary=output.with_suffix(".parquet.tmp")
        pq.write_table(table,temporary,compression="zstd",row_group_size=256)
        temporary.replace(output)
    receipt={"n":len(ids),"baseline_sha256":baseline_sha,"model_revision":MODEL_REVISION,
        "normalized_sidecars":len(files),"original_sidecars_preserved":True}
    (args.normalized_out/"NORMALIZATION_READY.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("MEASURED NORMALIZATION READY",json.dumps(receipt),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["sidecars","reference-info","output","normalized-out"]:p.add_argument("--"+name,type=Path,required=True)
    compute(p.parse_args())
