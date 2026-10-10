#!/usr/bin/env python3
"""Freeze additive annotation addresses without rewriting any source archive.

Local paths are CLI arguments. Large data and all outputs stay outside this
repository until selected artefacts are explicitly packaged for publication.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import sqlite3
import tarfile
from pathlib import Path

SEED = 20261010
LANGUAGES = ("de", "en", "fr", "ja", "ko", "zh")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def tar_addresses(path, wanted):
    found = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.name.endswith(".mp3") and member.name[:-4] in wanted:
                found[member.name[:-4]] = (member.offset_data, member.size)
    if set(found) != set(wanted):
        raise ValueError(f"Missing audio: {path}: {len(set(wanted)-set(found))}")
    return found


def prepare(args):
    import numpy as np
    import pyarrow.parquet as pq
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "annotation_manifest.jsonl"
    if manifest_path.exists():
        raise RuntimeError("Frozen manifest exists; choose a fresh output directory")
    ladder = json.loads(args.ladder_manifest.read_text())
    index = json.loads(args.archive_index.read_text())
    archived = {m["name"]:m for m in index["members"]}
    metadata_dir = out / "ladder_meta"
    metadata_dir.mkdir(exist_ok=True)
    rows, ladder_uids = [], set()
    with open(index["archive"], "rb") as archive:
        for shard in [s for s in ladder["shards"] if s["level"] == "t0"]:
            suffix = shard["meta"].split("/whisper_score_regression/",1)[1]
            member = archived["whisper_score_regression/"+suffix]
            target = metadata_dir / (Path(shard["meta"]).stem+".npy")
            archive.seek(member["offset_data"])
            raw = archive.read(member["size"])
            if len(raw) != member["size"]:
                raise IOError("Short archived index read")
            target.write_bytes(raw)
            data = np.load(target,mmap_mode="r")
            for row_index, record in enumerate(data):
                uid = record["uid"].decode().rstrip("\0")
                ladder_uids.add(uid)
                domain = "/".join(Path(shard["audio"]).parts[-4:-1])
                rows.append({"id":f"s10:{domain}:{uid}","source":"s10","source_key":uid,
                    "language":domain.split("/")[0],"duration_metadata_s":float(record["dur_s"]),
                    "audio_tar":shard["audio"],"audio_offset":int(record["audio_offset"]),
                    "audio_size":int(record["audio_size"]),"original_metadata":shard["parquet"],
                    "original_metadata_row":row_index,"source_dataset":"laion/tts-scaling-ladder-de-en",
                    "source_stage":"S10/t0"})
    print("S10 ADDRESSES",len(rows),flush=True)

    with sqlite3.connect(f"file:{args.tts_root}/production.sqlite?mode=ro",uri=True) as conn:
        tts = [(key,json.loads(manifest),shard) for key,manifest,shard in conn.execute(
            "SELECT key,manifest_json,audio_shard FROM samples WHERE audio_shard IS NOT NULL")]
    grouped = collections.defaultdict(list)
    for key, metadata, shard in tts:
        grouped[shard].append((key,metadata))
    for shard, samples in sorted(grouped.items()):
        path = Path(shard)
        if not path.is_absolute():
            path = args.tts_root / shard
        if not path.exists():
            path = args.tts_root / "shards" / Path(shard).name
        addresses = tar_addresses(path,{key for key,_ in samples})
        for key,metadata in samples:
            offset,size = addresses[key]
            rows.append({"id":"gemini:"+key,"source":"gemini_acting","source_key":key,
                "language":metadata["language"],"voice":metadata["voice"],"title":metadata["title"],
                "source_subset":metadata["source"],"audio_tar":str(path),
                "audio_offset":offset,"audio_size":size,
                "original_metadata_member":key+".json", "duration_metadata_s":30.,
                "source_dataset":"TTS-AGI/voice-acting-pipeline-gemini-38-flash-tts-en-de",
                "source_privacy":"original full recording collection private; selected demo clips authorised"})
    print("GEMINI ADDRESSES",len(tts),flush=True)

    index_root = args.emilia_root / "index/emolia"
    files = sorted(index_root.glob("*.parquet"))
    def language_file(path):
        q = pq.ParquetFile(path)
        k = q.schema.names.index("uid")
        stat = q.metadata.row_group(0).column(k).statistics
        uid = stat.min if stat and stat.has_min_max else next(q.iter_batches(batch_size=1,columns=["uid"])).column(0)[0].as_py()
        language = str(uid).split("_",1)[0].lower()
        return language,path,q.metadata.num_rows
    pools = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for language,path,n in executor.map(language_file,files):
            if language in LANGUAGES:
                pools[language].append((path,n))
    print("EMILIA LANGUAGE SHARDS",{k:len(v) for k,v in pools.items()},flush=True)
    rng = random.Random(SEED)
    source_counts = {}
    emilia_rows = []
    for i,language in enumerate(LANGUAGES):
        need = args.emilia_count // len(LANGUAGES) + (i < args.emilia_count % len(LANGUAGES))
        pool = pools[language].copy()
        rng.shuffle(pool)
        if not pool:
            raise RuntimeError(f"No local Emilia shards for {language}")
        # Equal quotas across up to 16 seeded random shards; no head-of-file sampling.
        shard_count = min(16,len(pool))
        picks = []
        for index,(path,n) in enumerate(pool):
            if len(picks) >= need:
                break
            q = pq.ParquetFile(path)
            names = [k for k in ["uid","lang","lang_iso","lang_conf","dur_s","text"] if k in q.schema_arrow.names]
            eligible = [r for r in q.read(columns=names).to_pylist()
                if (r.get("lang_iso") or r.get("lang")) == language
                and r["uid"].split("_",1)[0].lower() == language
                and .1 <= (r.get("dur_s") or 0) <= 30
                and (r.get("lang_conf") is None or r["lang_conf"] >= .8)
                and r["uid"] not in ladder_uids]
            quota = min(need-len(picks), max(1,math_ceil(need/shard_count)))
            selected = rng.sample(eligible,min(quota,len(eligible)))
            tar = args.emilia_root / "data/emolia" / (path.stem+".tar")
            addresses = tar_addresses(tar,{r["uid"] for r in selected})
            for metadata in selected:
                uid = metadata["uid"];offset,size = addresses[uid]
                picks.append({"id":"emilia:"+uid,"source":"emilia_multilingual","source_key":uid,
                    "language":language,"duration_metadata_s":metadata["dur_s"],
                    "source_transcript":metadata.get("text"),"audio_tar":str(tar),
                    "audio_offset":offset,"audio_size":size,"original_metadata":str(path),
                    "source_dataset":"amphion/Emilia-Dataset",
                    "local_collection":"laion-tts-annotated-v1/emolia; preserved existing annotations",
                    "original_audio_license":"See original Emilia versus Emilia-YODAS source terms"})
        if len(picks) != need:
            raise RuntimeError(f"Insufficient {language} candidates: {len(picks)}/{need}")
        source_counts[language]=len(picks)
        emilia_rows.extend(picks)
        print("EMILIA ADDRESSES",language,len(picks),flush=True)
    rows.extend(emilia_rows)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate namespaced sample IDs")
    with manifest_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    information = {"seed":SEED,"counts":dict(collections.Counter(r["source"] for r in rows)),
        "emilia_language_counts":source_counts,"n":len(rows),
        "source_archives_unchanged":True,"normalization_reference_only":True,
        "sampling":"All local S10 t0; all packed Gemini performances at frozen snapshot; Emilia language-balanced random shard quotas then uniform rows",
        "emilia_s10_overlap":"Excluded exact S10 UIDs from Emilia selection",
        "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_repo":"laion/humaneness-ears-base-medium","model_revision":"818506970d93c809a3295a0a02c40dee8ff2bfc3"}
    write_json(out/"reference_manifest_info.json",information)
    demo = []
    rng = random.Random(SEED+1)
    for language in ["de","en"]:
        candidates = [r for r in rows if r["source"]=="gemini_acting" and r["language"]==language
            and r["source_subset"]=="eric_morris_inspired"]
        demo.extend(rng.sample(candidates,10))
    for language,count in zip(LANGUAGES,[4,4,3,3,3,3]):
        demo.extend(rng.sample([r for r in emilia_rows if r["language"]==language],count))
    demo_dir=out/"demo_audio";demo_dir.mkdir(exist_ok=True)
    for i,row in enumerate(demo):
        row=dict(row)
        with open(row["audio_tar"],"rb") as archive:
            archive.seek(row["audio_offset"]);raw=archive.read(row["audio_size"])
        audio=demo_dir/f"demo-{i+1:02d}.mp3";audio.write_bytes(raw)
        row.update(audio_path=str(audio),sha256_mp3=hashlib.sha256(raw).hexdigest())
        demo[i]=row
    write_json(out/"demo_manifest.json",{"n":40,"samples":demo,"seed":SEED+1})
    print("REFERENCE MANIFEST READY",json.dumps(information),flush=True)


def math_ceil(value):
    import math
    return math.ceil(value)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["ladder-manifest","archive-index","tts-root","emilia-root","out"]:
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--emilia-count",type=int,default=38494)
    prepare(p.parse_args())
