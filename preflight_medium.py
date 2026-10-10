#!/usr/bin/env python3
"""Check actual pinned Medium weights, batching and Parakeet word timestamps."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def main(args):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from humaneness_backend import HumanenessMedium
    waves=[]
    for path in args.audio:
        wave,rate=sf.read(path,dtype="float32",always_2d=True)
        wave=wave.mean(1)
        if rate!=16000:
            import math
            d=math.gcd(rate,16000);wave=resample_poly(wave,16000//d,rate//d).astype(np.float32)
        waves.append(wave[:480000])
    model=HumanenessMedium(args.model_dir,device="cuda:0",threads=4)
    started=time.monotonic();batched=model.predict_batch(waves);elapsed=time.monotonic()-started
    single=model.predict_batch([waves[0]])[0]
    difference=max(abs(batched[0]["scores_training_z"][k]-single["scores_training_z"][k]) for k in model.names)
    if difference>.08:
        raise RuntimeError(f"Batched/unbatched regression disagreement: {difference}")
    # Original public inference is compared separately, on exactly the same WAV.
    wav_path=args.output.parent/"preflight-input.wav";wav_path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(wav_path,waves[0],16000,subtype="FLOAT")
    sys.path.insert(0,str(args.model_dir))
    spec=importlib.util.spec_from_file_location("original_he_inference",args.model_dir/"inference.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=module.Predictor(args.model_dir,device="cuda:0",threads=4).predict(wav_path,include_frames=True)
    raw_error=max(abs(reference["scores_raw"][k]-single["scores_raw"][k]) for k in model.names)
    if raw_error>1e-5:
        raise RuntimeError(f"Pinned public inference disagreement: {raw_error}")
    for prediction in batched:
        if not np.isfinite(list(prediction["scores_raw"].values())).all():raise ValueError("Nonfinite predictions")
        for event in prediction["burst_event_proposals"]:
            if not 0<=event["start_s"]<event["end_s"]<=prediction["duration_s"]:raise ValueError("Invalid event times")
    result={"passed":True,"model_parameter_count":sum(p.numel() for p in model.model.parameters()),
        "batch_size":len(waves),"first_batch_elapsed_s":elapsed,
        "batch_unbatch_max_training_z_error":difference,"public_inference_max_raw_error":raw_error,
        "raw_score_count":len(model.names),"event_class_count":len(model.classes),
        "model_finite_predictions":True,"event_start_end_checks":True}
    if args.parakeet_dir:
        from transformers import AutoProcessor,ParakeetForTDT
        from burst_captions import BurstCaptioner,_asr_helpers
        cap=BurstCaptioner.__new__(BurstCaptioner);cap.dev="cuda:0";cap.verbose=False
        cap._tokens_to_words,cap._sentences_from_words,cap._split_sentences=_asr_helpers()
        cap.pk_proc=AutoProcessor.from_pretrained(args.parakeet_dir,local_files_only=True)
        cap.pk=ParakeetForTDT.from_pretrained(args.parakeet_dir,local_files_only=True).to(cap.dev).eval()
        transcript,words,sentences=cap.parakeet(torch.from_numpy(waves[0]))
        if not transcript or not words:raise RuntimeError("Parakeet produced no usable word timestamps")
        if not all(w["start"]<=w["end"] and w["end"]<=w.get("end_span",w["end"])+1e-6 for w in words):
            raise ValueError("Word-end trimming enlarged a Parakeet token span")
        result["parakeet"]={"transcript":transcript,"word_count":len(words),"sentence_count":len(sentences),
            "raw_span_preserved":all("end_span" in w for w in words),"timestamp_checks_passed":True}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--audio",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--parakeet-dir",type=Path)
    main(p.parse_args())
