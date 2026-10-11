"""Procedural captions from the Medium encoder, with Parakeet v3 ASR only."""
from __future__ import annotations

import os
import copy

from caption import caption_detail, render_caption, stable_hash
from humaneness_backend import NAMESPACE, caption_scores
from humaneness_caption_settings import load_settings,filter_events,caption_view

PARAKEET_LANGUAGES = set("bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split())


def add_reference_z(prediction, baseline):
    """Keep training-standardized outputs separate from measured caption z-scores."""
    stats = baseline.get("_raw_targets", {})
    values = dict(prediction["scores_raw"])
    values.update({key: prediction[key] for key in ["characters_per_second", "estimated_burst_count"]})
    normalized = {}
    for name, value in values.items():
        stat = stats.get(name)
        normalized[name] = ((value-stat["mean"])/stat["std"]
            if stat and stat.get("z_defined") else None)
    prediction["scores_reference_z"] = {name: normalized[name] for name in prediction["scores_raw"]}
    prediction["auxiliary_reference_z"] = {name: normalized[name]
        for name in ["characters_per_second", "estimated_burst_count"]}
    return prediction


def process_humaneness(cap, path, cid=None, mp3_out=None, language=None, precomputed_prediction=None):
    from burst_captions import (SR, PROC_TEMPLATE, BC_TIMED, PAUSE_THR, WORD_END,
        SILENCE_FLOOR, decode_16k, write_mp3, build_timed_script, verify_timed_script)
    cid = cid or os.path.splitext(os.path.basename(path))[0]
    wav = decode_16k(path)
    duration = len(wav) / SR
    seed = stable_hash(cid) % (1 << 30)
    asr_supported = language is None or language.lower() in PARAKEET_LANGUAGES
    transcript, words, sentences = cap.parakeet(wav) if asr_supported else ("", [], [])
    settings=getattr(cap,'medium_caption_settings',None) or load_settings()
    global_prediction = add_reference_z(copy.deepcopy(precomputed_prediction) if precomputed_prediction is not None else cap.he.predict_waveform(wav), cap.baseline)
    selected_events=filter_events(global_prediction['burst_event_proposals'],settings)
    global_prediction['selected_burst_event_proposals']=selected_events
    global_prediction['caption_settings']=settings
    scores = caption_scores(global_prediction)
    detail = caption_detail(caption_view(global_prediction,cap.baseline,settings), cap.baseline,
        k_voicenet=settings['kv'],k_emonet=settings['ke'],always_on=['TEMP'],synonym_seed=seed,template=settings['template'])
    events = []
    for event in selected_events:
        top = event["top3_classes"]
        if not top:
            continue
        p = top[0]["softmax_probability"]
        # No no_burst head exists in this 53-class checkpoint. Onsets determine
        # detection. Low-confidence names are retained in raw results but hedged.
        written = top[0]["label"] if p >= .5 else "Vocal Burst"
        events.append({"start": event["start_s"], "end": event["end_s"],
            "label": top[0]["label"], "written": written, "prob": p, "kept": True,
            "onset_probability": event["onset_probability"], "top3_classes": top,
            "vocab": "humaneness-medium-53", "tier": "named" if p >= .5 else "generic"})
    sent_out = []
    for i, sent in enumerate(sentences):
        a, b = sent.get("start"), sent.get("end")
        if a is None or b is None or b <= a:
            prediction = global_prediction
        else:
            cut = wav[max(0,int((a-.1)*SR)):min(len(wav),int((b+.1)*SR))]
            if len(cut) < 1600:
                prediction = global_prediction
            else:
                prediction = add_reference_z(cap.he.predict_waveform(cut, include_frames=False), cap.baseline)
        sentence_scores = caption_scores(prediction)
        selected=caption_detail(caption_view(prediction,cap.baseline,settings),cap.baseline,
            k_voicenet=min(3,settings['kv']),k_emonet=settings['ke'],always_on=[],synonym_seed=seed+i+1,template=settings['template'])
        cue=render_caption(selected)
        sent_out.append(dict(sent, caption=cue, scores=sentence_scores,
            caption_detail=selected, **{NAMESPACE: prediction}))
    # Existing word alignment, pause rendering and verification are reused.
    script_lines = cap._assign_bursts(sent_out, words, events)
    pairs = [(e["start"],e["end"],e["written"],e["prob"]) for e in events]
    inline = cap.variant_a(words,pairs)
    timed, timed_ok = None, None
    if BC_TIMED and words:
        try:
            timed = build_timed_script(sent_out,words,[dict(e,label=e["written"]) for e in events],duration)
            timed_ok = not verify_timed_script("\n".join(timed),
                " ".join(w["w"] for w in words if w.get("start") is not None).strip(),duration)
            if not timed_ok:
                timed = None
        except ValueError:
            timed = None
            timed_ok = False
    if mp3_out:
        write_mp3(path,mp3_out,os.environ.get("BC_MP3_BITRATE","128k"))
    result = {"id":cid,"dur":duration,"transcript":transcript,"words":words,
        "scores":scores,"score_backend":"humaneness-medium",NAMESPACE:global_prediction,
        "normalization_provenance":cap.baseline.get("_provenance"),
        "genu":scores["genu"],"blend":scores["blend"],"content_enjoyment":scores["content_enjoyment"],
        "global_caption":render_caption(detail)+('; vocal-burst candidates are marked in the timed event list' if selected_events else ''),"global_caption_detail":detail,
        "caption_settings":settings,
        "sentences":sent_out,"script_lines":script_lines,"script_timed":timed,"timed_ok":timed_ok,
        "variant_a_bursts":events,"variant_a_inline":inline,"n_spans":len(events),
        "n_words_ts":len(words),"n_locator_windows":global_prediction["n_windows"],
        "locator":"laion/humaneness-ears-base-medium:medium", "classifier":"Humaneness Ears Medium event-local 53-class head",
        "x2":None,"emonet":True,"ei_gender":None,"gender_gated":False,
        "asr":{"model":"nvidia/parakeet-tdt-0.6b-v3","known_language":language,
            "supported_language":asr_supported,"status":"computed" if asr_supported else "not run: unsupported language",
            "word_timestamps":"Parakeet v3, unchanged RMS end trimming" if asr_supported else None},
        "pause":{"threshold":PAUSE_THR,"word_end":WORD_END,"silence_floor":SILENCE_FLOOR}}
    if asr_supported:
        from prompt_format import format_record
        result['procedural_caption'] = format_record(result, duration_style=settings.get('duration_style','square'))
    else:
        result['timing_unavailable_reason'] = 'ASR is not supported for this language; no timed spoken script is invented.'
    return result
