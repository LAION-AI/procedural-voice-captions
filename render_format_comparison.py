#!/usr/bin/env python3
"""Render the frozen paired MOSS/procedural audio review and training surfaces."""
from pathlib import Path
import argparse,base64,copy,hashlib,html,json,sys
import numpy as np
from refresh_saved_caption import refresh_saved_caption
from training_formats import moss_training_formats
from humaneness_caption_settings import load_settings

PAGE=(Path(__file__).parent/'templates'/'format_comparison.html').read_text()


def table(rows,keys):
    return '<div class="table"><table'+(' class="results"' if len(keys)>4 else '')+'><thead><tr>'+''.join('<th>'+html.escape(x[1])+'</th>' for x in keys)+'</tr></thead><tbody>'+''.join('<tr'+(' class="moss-result"' if row['name'].startswith('MOSS') else '')+'>'+''.join('<td'+(' class="num"' if isinstance(row[k],(int,float)) else '')+'>'+ (f'{row[k]:.3f}' if isinstance(row[k],float) else html.escape(str(row[k])))+'</td>' for k,_ in keys)+'</tr>' for row in rows)+'</tbody></table></div>'


def render(args):
    root=args.input;baseline=json.loads((Path(__file__).parent/'baseline_stats_humaneness_medium_speech.json').read_text())
    manifest=[json.loads(x) for x in (root/'comparison_samples.jsonl').open()]
    refs={r['comparison_id']:r for r in map(json.loads,(root/'reference_samples.jsonl').open())}
    sample_events={r['id']:r['events'] for r in map(json.loads,(root/'samples.jsonl').open())}
    predictions={(r['sample_id'],r['task'],r['event_index']):r for p in root.glob('predictions-r*.jsonl') for r in map(json.loads,p.open())}
    ratings={r['id']:{x['config_id']:x for x in r['answer']['ratings']} for phase in ['selection','confirmation'] for p in root.glob(phase+'-*.json') if p.name!=phase+'-summary.json' for r in [json.loads(p.read_text())]}
    selection=json.loads((root/'selection-summary.json').read_text());confirmation=json.loads((root/'confirmation-summary.json').read_text())
    keys=[('name','Pipeline'),('weighted_score','Overall composite'),('caption_fidelity','Whole-voice description'),('burst_description_fidelity','Burst description'),('timing_accuracy','Event timing'),('event_precision','False-event avoidance'),('event_recall','Event coverage'),('readability','Readability')]
    titles={'guide-concise':'Procedural · compact tags','reviewed-prose':'Procedural · previous prose','moss':'MOSS · greedy, uncapped'}
    all_results=[]
    for config in titles:
        row={'name':titles[config],'config_id':config,'n':50}
        for k,_ in keys[1:]:row[k]=float(np.mean([r[config][k] for r in ratings.values()]))
        all_results.append(row)
    rng=np.random.default_rng(20261011)
    ci=[]
    for partition in ['all','confirmation']:
        ids=[r['id'] for r in manifest if partition=='all' or r['partition']==partition]
        delta=np.array([ratings[s]['moss']['weighted_score']-ratings[s]['guide-concise']['weighted_score'] for s in ids])
        b=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
        ci.append({'partition':partition,'n':len(ids),'mean_moss_minus_procedural':float(delta.mean()),'paired_bootstrap_95_ci':np.quantile(b,[.025,.975]).tolist()})
    cards=[];variant_data={};proc_events=0;moss_events=0
    for sample_number,sample in enumerate(sorted(manifest,key=lambda x:(x['language']!='en',x['id'])),start=1):
        sid=sample['id'];options={o['config_id']:o for o in sample['options']};proc=options['guide-concise'];moss=options['moss']
        proc_events+=len(proc['events']);moss_events+=len(moss['events'])
        fresh=refresh_saved_caption(copy.deepcopy(refs[sid]),baseline,load_settings());assert fresh['procedural_caption']==proc['training_prompt']
        record={'voice_caption':{'prediction':moss['general_caption']},'audio_duration_s':sample['duration_s'],'locator_settings':{'threshold':.6,'min_duration_s':.08,'nms_overlap_fraction':.5,'event_cap':None},
                'burst_captions':[dict(e,caption={'prediction':predictions[(sid,'burst_caption',e['event_index'])]['prediction']}) for e in sample_events[sid]]}
        formats=moss_training_formats(fresh,record);variant_data[sid]=formats['variants']
        e=html.escape
        def events(option,color):
            rects=''.join(f'<rect x="{x["start_s"]/sample["duration_s"]*1000:.2f}" y="15" width="{max(1,(x["end_s"]-x["start_s"])/sample["duration_s"]*1000):.2f}" height="20" rx="3" fill="{color}"/>' for x in option['events'])
            timeline=f'<svg class="timeline" viewBox="0 0 1000 65" role="img" aria-label="Predicted vocal event intervals from zero to the full recording duration">{rects}<text x="5" y="57" fill="#b2c0d8" font-size="13">0 s</text><text x="900" y="57" fill="#b2c0d8" font-size="13">{sample["duration_s"]:.1f} s</text></svg>'
            return timeline+(''.join(f'<div class="event"><button type="button" data-seek="{x["start_s"]}" aria-label="Jump to event at {x["start_s"]:.2f} seconds">{x["start_s"]:.2f}–{x["end_s"]:.2f} s ▶</button> <span class="event-duration">Duration: {x["duration_s"]:.2f} s</span><br>{e(x["caption"])}</div>' for x in option['events']) if option['events'] else '<p class="review">No interval selected by this configuration.</p>')
        caption_labels={'whole_audio_only':'Whole-recording caption only','with_burst_descriptions':'Whole-recording caption + event descriptions'}
        transcript_labels={'plain':'plain ASR transcript','inline_bursts':'transcript with inline event labels','timed':'timed transcript with event labels'}
        selectors=''.join(f'<option value="{e(k)}">{e(caption_labels[k.split("__")[0]]+" / "+transcript_labels[k.split("__")[1]])}</option>' for k in formats['variants'])
        blob=base64.b64encode(Path(sample['audio_path']).read_bytes()).decode()
        score=ratings[sid]
        language_name={'en':'English','de':'German'}[sample['language']]
        partition_name={'selection':'Settings-selection recording','confirmation':'Confirmation recording'}[sample['partition']]
        cards.append(f'''<article data-language="{sample["language"]}" id="recording-{sample_number}">
<div class="eyebrow">{language_name} · {sample["duration_s"]:.1f} seconds · {partition_name}</div>
<h2 class="sample-heading">Recording {sample_number:02d} · {language_name}</h2><div class="sample-id">Source recording ID: {e(sid)}</div>
<audio controls preload="none" aria-label="Original audio for recording {sample_number}" src="data:audio/mpeg;base64,{blob}"></audio>
<div class="cols"><section><div class="column-heading"><h3><span class="dot proc-dot"></span>Procedural captions</h3><span class="event-count">{len(proc["events"])} selected intervals</span></div>
<p class="review">Rule-generated GENERAL / SCRIPT · automated overall rating {score["guide-concise"]["weighted_score"]:.2f}/5</p><pre class="script">{e(proc["training_prompt"])}</pre>
<h3>Proposed intervals and event labels</h3>{events(proc,"#c4b4ff")}</section>
<section><div class="column-heading"><h3><span class="dot moss-dot"></span>MOSS captions</h3><span class="event-count">{len(moss["events"])} selected intervals</span></div>
<p class="review">Whole-voice and interval-conditioned models · automated overall rating {score["moss"]["weighted_score"]:.2f}/5</p><div class="caption">{e(moss["general_caption"])}</div>
<h3>Proposed intervals and MOSS descriptions</h3>{events(moss,"#86e5da")}</section></div>
<details><summary>Inspect the six existing CAPTION / TRANSCRIPT exports</summary><p class="review">Choose the caption and transcript surface. Inline transcript markers currently use event type labels; detailed MOSS burst prose is appended only in the caption + event-descriptions option. Plain text preserves the ASR words. These are saved annotations for training, not alternative audio generations.</p>
<label for="choice-{e(sid)}">Training text variant</label><br><select id="choice-{e(sid)}" data-format="{e(sid)}">{selectors}</select><pre id="surface-{e(sid)}" class="script">{e(next(iter(formats["variants"].values()))["prompt"])}</pre></details>
<details><summary>Read the automated reviewer's notes for this recording</summary><h3>Procedural captions</h3><p>{e(score["guide-concise"]["note"])}</p><h3>MOSS captions</h3><p>{e(score["moss"]["note"])}</p></details>
</article>''')
    conf=[dict(confirmation['results'][k],name=titles[k]) for k in titles]
    search_names={'guide-concise':'Compact default: 2 voice properties, up to 1 emotion', 'guide-tags':'Longer tags: 3 voice properties, up to 2 emotions', 'reviewed-prose':'Previous prose: 3 voice properties, up to 2 emotions', 'guide-precision':'Stricter threshold 0.8: longer tags'}
    search=[dict(selection['results'][k],name=search_names[k]) for k in selection['procedural_ranking']]
    intervals='; '.join(f'{c["partition"]}: MOSS minus compact procedural = {c["mean_moss_minus_procedural"]:.3f}, paired bootstrap 95% interval [{c["paired_bootstrap_95_ci"][0]:.3f}, {c["paired_bootstrap_95_ci"][1]:.3f}]' for c in ci)
    page=PAGE.replace('__RESULT_TABLE__',table(all_results,keys)).replace('__CONFIRM_TABLE__',table(conf,keys)).replace('__SEARCH_TABLE__',table(search,keys)).replace('__INTERVALS__',intervals).replace('__PROC_EVENTS__',str(proc_events)).replace('__MOSS_EVENTS__',str(moss_events)).replace('__CARDS__',''.join(cards)).replace('__VARIANTS__',json.dumps(variant_data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(page)
    summary={'schema':'paired-caption-format-review-v1','n':50,'languages':{'en':45,'de':5},'selection_n':25,'confirmation_n':25,'results_all50':all_results,'confirmation':confirmation,'selection':selection,'bootstrap':ci,'selected_config':'guide-concise','selected_before_confirmation':True,'moss_events':moss_events,'procedural_events':proc_events,'all_50_timed_formats_verified':True,'training_variants_per_clip':6,'model_version':'gemini-3.8-flash','corpus_format_check':json.loads((root/'CORPUS_FORMAT_CHECK.json').read_text()),'automated_review_not_human_ground_truth':True,'rubric':selection['rubric']}
    args.summary.write_text(json.dumps(summary,indent=2)+'\n')
    assert page.count('data:audio/mpeg;base64,')==50
    print('PAIRED_FORMAT_REPORT_READY',len(cards),args.output.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);render(p.parse_args())
