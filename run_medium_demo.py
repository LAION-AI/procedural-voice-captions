#!/usr/bin/env python3
"""Caption a selected listening manifest; never treat source text as an ASR result."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def run(args):
    # The same Parakeet implementation is used, with a pre-staged model path.
    if args.parakeet_dir:
        os.environ["PARAKEET_MODEL"] = str(args.parakeet_dir)
    from burst_captions import BurstCaptioner
    cap = BurstCaptioner(device=args.device, score_backend="humaneness-medium",
        humaneness_dir=args.model_dir, baseline=str(args.baseline), threads=4)
    selection = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    for index, source in enumerate(selection["samples"]):
        if index % args.world_size != args.rank:
            continue
        output = args.out / f"demo-{index+1:02d}.json"
        if output.exists():
            old = json.loads(output.read_text())
            if old.get("normalization_sha256") != baseline_hash:
                raise ValueError("Demo baseline changed: choose a new output directory")
            continue
        audio = args.out / f"demo-{index+1:02d}.mp3"
        result = cap.process(source["audio_path"], cid=source["id"],
            mp3_out=str(audio), language=source["language"])
        # Public provenance excludes machine-local paths and private source prompts.
        result["source"] = {k: source[k] for k in ["id", "source", "source_key",
            "language", "source_dataset", "voice", "title", "source_subset"] if k in source}
        result["source_transcript"] = source.get("source_transcript")
        result["source_transcript_note"] = "Existing corpus metadata; not used to compute ASR, scores, or timestamps."
        result["normalization_sha256"] = baseline_hash
        result["audio_mp3_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
        result["audio_mp3_bitrate_kbps"] = 128
        result["audio_mp3_channels"] = 1
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)
        print("CAPTIONED", source["id"], "bursts", result["n_spans"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "out", "model-dir", "baseline"]:
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--parakeet-dir", type=Path)
    p.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    run(p.parse_args())
