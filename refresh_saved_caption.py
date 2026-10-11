"""Regenerate Medium captions from saved scores/ASR, without model inference."""
from caption import caption_detail, render_caption, stable_hash
from humaneness_captions import add_reference_z
from humaneness_caption_settings import caption_view, filter_events, load_settings
from prompt_format import format_record, _DURATION


def refresh_saved_caption(record, baseline, settings=None, *, variation_index=0):
    settings = dict(settings or load_settings())
    seed = stable_hash(record['id']) % (1 << 30) + variation_index
    prediction = add_reference_z(record['humaneness_ears_medium'], baseline)
    selected = filter_events(prediction['burst_event_proposals'], settings)
    prediction['selected_burst_event_proposals'] = selected
    prediction['caption_settings'] = settings
    record['humaneness_ears_medium'] = prediction
    detail = caption_detail(caption_view(prediction, baseline, settings), baseline,
                            k_voicenet=settings['kv'], k_emonet=settings['ke'], always_on=['TEMP'],
                            synonym_seed=seed, template=settings['template'])
    general = render_caption(detail)
    if selected:
        general += '; vocal-burst candidates are marked in the timed event list'
    record.update(global_caption=general, global_caption_detail=detail, caption_settings=settings,
                  normalization_provenance=baseline['_provenance'], caption_normalization_scope='speech_reference_80265')
    for i, sentence in enumerate(record.get('sentences', [])):
        raw = add_reference_z(sentence['humaneness_ears_medium'], baseline)
        description = caption_detail(caption_view(raw, baseline, settings), baseline,
                                     k_voicenet=min(3, settings['kv']), k_emonet=settings['ke'], always_on=[],
                                     synonym_seed=seed + i + 1, template=settings['template'])
        sentence.update(caption=render_caption(description), caption_detail=description)
    events = []
    for event in selected:
        top = event['top3_classes']
        confidence = top[0]['softmax_probability']
        written = top[0]['label'] if confidence >= .5 else 'Vocal Burst'
        events.append({'start':event['start_s'], 'end':event['end_s'], 'label':top[0]['label'],
                       'written':written, 'prob':confidence, 'kept':True,
                       'onset_probability':event['onset_probability'], 'top3_classes':top,
                       'vocab':'humaneness-medium-53', 'tier':'named' if confidence >= .5 else 'generic'})
    record.update(variant_a_bursts=events, n_spans=len(events))
    record['script_lines'] = [{'cue':s['caption'], 'text':s['text'], 'start':s['start'], 'end':s['end']}
                              for s in record.get('sentences', [])]
    record['procedural_caption'] = format_record(record, duration_style=settings.get('duration_style', 'square'))
    record['variant_a_inline'] = '\n'.join(_DURATION.sub('', line).strip() for line in record['script_timed'])
    return record
