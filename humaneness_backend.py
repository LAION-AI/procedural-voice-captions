"""Pinned Humaneness Ears Medium inference for procedural voice captions.

The original expert scores, when present, remain separate from these outputs.
No ASR decoder or original teacher is loaded by this backend.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

MODEL_REPO = "laion/humaneness-ears-base-medium"
MODEL_REVISION = "818506970d93c809a3295a0a02c40dee8ff2bfc3"
NAMESPACE = "humaneness_ears_medium"
EMOTION_ALIASES = {
    "Astonishment_Surprise": "Astonishment/Surprise",
    "Emotional_Numbness": "Emotional Numbness",
    "Fatigue_Exhaustion": "Fatigue/Exhaustion",
    "Hope_Enthusiasm_Optimism": "Hope/Optimism",
    "Impatience_and_Irritability": "Impatience and Irritability",
    "Intoxication_Altered_States_of_Consciousness": "Intoxication/Altered States",
    "Jealousy_and_Envy": "Jealousy & Envy",
    "Malevolence_Malice": "Malevolence/Malice",
    "Pleasure_Ecstasy": "Pleasure/Ecstasy",
    "Sexual_Lust": "Sexual Lust",
    "Thankfulness_Gratitude": "Thankfulness/Gratitude",
}


def caption_scores(prediction):
    """Lossless raw-score view in caption.py's historical taxonomy keys."""
    raw = prediction["scores_raw"]
    dims = {name[3:-4]: value for name, value in raw.items()
            if name.startswith("vn_") and name.endswith("_reg")}
    emotions = {EMOTION_ALIASES.get(name[4:], name[4:].replace("_", " ")): value
                for name, value in raw.items() if name.startswith("emo_")}
    return {"dims": dims, "emo": emotions, "genu": raw.get("genuineness_0_6"),
            "blend": raw.get("blend_0_10"),
            "content_enjoyment": raw.get("audiobox:CE"),
            "score_backend": "humaneness-medium", "score_namespace": NAMESPACE}


def window_starts(samples, window=480000, overlap=80000):
    """30-second windows, 5-second overlaps, end-anchored final window."""
    if samples <= window:
        return [0]
    starts = list(range(0, samples - window + 1, window - overlap))
    if starts[-1] != samples - window:
        starts.append(samples - window)
    return starts


class HumanenessMedium:
    def __init__(self, repo_dir=None, device="cpu", threads=4, batch_size=32):
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import WhisperFeatureExtractor
        if repo_dir is None:
            repo_dir = os.environ.get("HUMANENESS_EARS_DIR")
        if not repo_dir:
            from huggingface_hub import snapshot_download
            repo_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                allow_patterns=["model.py", "training_normalization.json", "classes.json",
                                "medium/config.json", "medium/model.safetensors",
                                "medium/preprocessor_config.json"])
        self.repo, self.device = Path(repo_dir), device
        self.batch_size = batch_size
        torch.set_num_threads(threads)
        module_name = "_pvc_humaneness_encoder"
        spec = importlib.util.spec_from_file_location(module_name, self.repo / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.norm = json.loads((self.repo / "training_normalization.json").read_text())
        self.classes = json.loads((self.repo / "classes.json").read_text())["names"]
        self.names = self.norm["score_names"]
        if len(self.names) != 192 or len([n for n in self.names if n.startswith("emo_")]) != 40:
            raise ValueError("Unexpected Humaneness Ears target schema")
        self.model = module.LayeredMultiTaskWhisper(self.repo / "medium", len(self.classes),
                                                   initialize_pretrained=False)
        self.model.load_state_dict(load_file(self.repo / "medium/model.safetensors"), strict=True)
        self.model.to(device).eval()
        self.extractor = WhisperFeatureExtractor.from_pretrained(self.repo / "medium", local_files_only=True)
        self.mean = np.asarray(self.norm["score_mean"], dtype=np.float64)
        self.std = np.asarray(self.norm["score_std"], dtype=np.float64)

    def predict_batch(self, waves, *, event_threshold=.5, max_events=32, include_frames=True):
        """Mono 16-kHz arrays, each 0.1–30 s. Raw outputs are never clipped."""
        import numpy as np
        import torch
        arrays = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
        if not arrays:
            return []
        if any(not 1600 <= len(w) <= 480000 or not np.isfinite(w).all() for w in arrays):
            raise ValueError("Each waveform must contain finite 0.1–30 s audio at 16 kHz")
        features = self.extractor(arrays, sampling_rate=16000, padding="longest",
            truncation=True, max_length=480000, return_attention_mask=True, return_tensors="np")
        mel = features["input_features"].astype(np.float32)
        mask = features["attention_mask"].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mask = np.pad(mask, ((0, 0), (0, 1)))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if str(self.device).startswith("cuda") else nullcontext()
        with torch.inference_mode(), autocast:
            out = self.model(torch.from_numpy(mel).to(self.device), torch.from_numpy(mask).to(self.device),
                predict_events=True, event_threshold=event_threshold, max_pred_events=max_events)
        tensors = {key: value.float().cpu().numpy() for key, value in out.items()}
        standardized = tensors["scores"]
        raw = standardized * self.std + self.mean
        probabilities = torch.from_numpy(tensors["event_class"]).softmax(-1).numpy()
        frames = torch.from_numpy(tensors["frame"]).sigmoid().numpy()
        onsets = torch.from_numpy(tensors["onset"]).sigmoid().numpy()
        results = []
        for i, wave in enumerate(arrays):
            duration = len(wave) / 16000
            nframes = min(int((mask[i].sum() + 1) // 2), frames.shape[1])
            proposals = []
            for j in np.flatnonzero(tensors["predicted_event_valid"][i]):
                a = int(tensors["predicted_event_starts"][i, j])
                b = int(tensors["predicted_event_ends"][i, j])
                start, end = a * .02, min(b * .02, duration)
                if start >= end:
                    continue
                top = np.argsort(probabilities[i, j])[-3:][::-1]
                proposals.append({"start_s": round(start, 6), "end_s": min(duration, round(end, 6)),
                    "onset_probability": float(onsets[i, a]),
                    "top3_classes": [{"label": self.classes[int(k)],
                        "softmax_probability": float(probabilities[i, j, k])} for k in top]})
            r = {"scores_raw": dict(zip(self.names, map(float, raw[i]))),
                 "scores_training_z": dict(zip(self.names, map(float, standardized[i]))),
                 "duration_s": duration, "burst_event_proposals": proposals,
                 "characters_per_second": float(tensors["cps"][i] * self.norm["cps_std"] + self.norm["cps_mean"]),
                 "estimated_burst_count": float(np.expm1(np.clip(raw[i, 130], 0, 8))),
                 "orange_timbre_128d": tensors["timbre"][i].tolist(),
                 "orange_identity_250d": tensors["identity"][i].tolist(),
                 "provenance": {"model_repo": MODEL_REPO, "revision": MODEL_REVISION,
                     "variant": "medium", "namespace": NAMESPACE, "precision": "bfloat16" if str(self.device).startswith("cuda") else "float32"}}
            if include_frames:
                r["burst_probability_per_20ms_frame"] = frames[i, :nframes].tolist()
            results.append(r)
        return results

    def predict_waveform(self, waveform, *, include_frames=True):
        """Full audio coverage, cross-faded frame track and duplicate-safe events.

        Overlapping windows are used only to provide context. Each event belongs
        to the window whose centre is nearest its midpoint, so two windows do not
        write the same event. Scores use partition-of-unity duration weights.
        """
        import numpy as np
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
        starts = window_starts(len(wave))
        if len(starts) == 1:
            result = self.predict_batch([wave], include_frames=True)[0]
            result["n_windows"] = 1
            if not include_frames:
                result.pop("burst_probability_per_20ms_frame", None)
            return result
        results = []
        for i in range(0, len(starts), self.batch_size):
            results.extend(self.predict_batch([wave[s:s+480000] for s in starts[i:i+self.batch_size]], include_frames=True))
        return self.combine_window_predictions(len(wave), starts, results, include_frames=include_frames)

    def combine_window_predictions(self, samples, starts, results, *, include_frames=True):
        """Combine already batched windows; used by the corpus annotator."""
        import numpy as np
        if len(starts) != len(results) or not results:
            raise ValueError("Window address/prediction mismatch")
        if len(starts) == 1:
            result = dict(results[0], n_windows=1)
            if not include_frames:
                result.pop("burst_probability_per_20ms_frame", None)
            return result
        duration = samples / 16000
        nframes = math.ceil(samples / 320)
        accumulated, weights = np.zeros(nframes), np.zeros(nframes)
        window_weights = []
        centres = [(s + min(480000, samples - s) / 2) / 16000 for s in starts]
        # Adjacent centre midpoints partition the global timeline exactly.
        bounds = [0.] + [(a+b)/2 for a,b in zip(centres, centres[1:])] + [duration]
        events = []
        for i, (s, r) in enumerate(zip(starts, results)):
            offset = s / 16000
            p = np.asarray(r["burst_probability_per_20ms_frame"])
            # The end-anchored window can start between global 20-ms frames.
            # Rounding that offset down would leave the last partial frame
            # uncovered. Align by timestamp and interpolate within the window.
            f0 = math.ceil(s / 320)
            fend = min(nframes, math.ceil((s + round(r["duration_s"] * 16000)) / 320))
            count = fend - f0
            t = (np.arange(count) + f0) * .02
            # Smooth triangular context weights; no uncovered or zero-weight frames.
            context = np.maximum(.001, 1. - np.abs((t-offset) / max(r["duration_s"], .001) * 2 - 1))
            aligned = np.interp(t, offset + np.arange(len(p)) * .02, p)
            accumulated[f0:f0+count] += aligned * context
            weights[f0:f0+count] += context
            window_weights.append(bounds[i+1] - bounds[i])
            for event in r["burst_event_proposals"]:
                e = dict(event, start_s=min(duration, event["start_s"]+offset),
                         end_s=min(duration, event["end_s"]+offset))
                midpoint = (e["start_s"]+e["end_s"]) / 2
                if bounds[i] <= midpoint < bounds[i+1]:
                    events.append(e)
        if np.any(weights <= 0):
            raise RuntimeError("Uncovered Humaneness Ears frame")
        mixture = np.asarray(window_weights) / duration
        raw = sum(w * np.asarray([r["scores_raw"][n] for n in self.names]) for w,r in zip(mixture, results))
        result = dict(results[0])
        result.update(duration_s=duration, scores_raw=dict(zip(self.names,map(float,raw))),
            scores_training_z=dict(zip(self.names,map(float,(raw-self.mean)/self.std))),
            characters_per_second=float(sum(w*r["characters_per_second"] for w,r in zip(mixture,results))),
            burst_event_proposals=sorted(events,key=lambda e:e["start_s"]), n_windows=len(starts),
            estimated_burst_count=len(events), burst_probability_per_20ms_frame=(accumulated/weights).tolist())
        for key in ["orange_timbre_128d", "orange_identity_250d"]:
            embedding = sum(w*np.asarray(r[key]) for w,r in zip(mixture,results))
            norm = np.linalg.norm(embedding)
            result[key] = (embedding / max(float(norm),1e-12)).tolist()
        result["long_audio_aggregation"] = {"scores": "duration-weighted partition by nearest window centre",
            "count": "number of owned onset/duration proposals", "window_s": 30, "overlap_s": 5,
            "frame_track": "normalised triangular context weights",
            "event_ownership": "nearest window centre to event midpoint; class scores uncalibrated"}
        if not include_frames:
            result.pop("burst_probability_per_20ms_frame",None)
        return result
