"""Selectable MOSS CAPTION/TRANSCRIPT surfaces for Humaneness Voice training."""
from prompt_format import render_timed, _DURATION


def _quote(text):
    # Preserve real newlines in the training surface; escape literal quotation
    # marks/backslashes so the boundary of the transcript remains unambiguous.
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def moss_training_formats(procedural, moss):
    """Six complete, reproducible choices; no model invocation or text invention."""
    whole = moss['voice_caption']['prediction']
    bursts = []
    descriptions = []
    for i, event in enumerate(moss['burst_captions']):
        start, end = event['start_s'], event['end_s']
        top = event['top3_classes'][0]
        written = top['label'] if top['softmax_probability'] >= .5 else 'Vocal Burst'
        bursts.append({'start_s':start, 'end_s':end, 'label':written})
        descriptions.append(f"Event {i+1}, {start:.2f}–{end:.2f} s ({end-start:.2f} s): {event['caption']['prediction']}")
    appended = whole + ('\nVocal events:\n' + '\n'.join(descriptions) if descriptions else '\nNo vocal-event intervals selected by this locator configuration.')
    lines, proof = render_timed([], procedural['words'], bursts, moss['audio_duration_s'], include_cues=False)
    transcripts = {
        'plain':procedural['transcript'],
        'inline_bursts':'\n'.join(_DURATION.sub('', line).strip() for line in lines),
        'timed':'\n'.join(lines),
    }
    variants = {}
    for kind, caption in [('whole_audio_only', whole), ('with_burst_descriptions', appended)]:
        for surface, transcript in transcripts.items():
            name = kind + '__' + surface
            variants[name] = {
                'caption':caption, 'transcript':transcript,
                'prompt':'CAPTION: ' + caption + '\nTRANSCRIPT: ' + _quote(transcript),
                'caption_source':'MOSS whole-audio adapter' + (' + interval-conditioned burst adapter' if kind=='with_burst_descriptions' else ''),
                'transcript_source':'Parakeet v3 ASR' if surface=='plain' else 'Parakeet v3 word sequence + Humaneness Ears event intervals',
                'appended_burst_descriptions':kind=='with_burst_descriptions',
                'burst_tags_in_transcript':surface!='plain',
                'segment_durations_in_transcript':surface=='timed',
                'locator_settings':moss['locator_settings'],
            }
    return {'schema':'humaneness-moss-caption-transcript-variants-v1', 'variants':variants,
            'timed_transcript_verification':proof,
            'plain_transcript_verbatim':True,
            'whole_audio_only_note':'Original whole-audio prose is unchanged and can itself mention vocal events; only appended event captions are omitted.',
            'inline_bursts_note':'Speech-segment duration tags are omitted; measured burst durations and pause markers are retained.',
            'pairing_note':'All MOSS transcript event tags use the same locator intervals as the MOSS burst-caption list, independently of the procedural locator threshold.'}
