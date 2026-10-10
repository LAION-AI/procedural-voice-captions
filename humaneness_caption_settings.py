"""Reviewed default settings for captions; raw acoustic scores stay unchanged."""
from pathlib import Path
import copy,json
from humaneness_backend import caption_scores

def load_settings():
 return json.loads((Path(__file__).parent/'humaneness_caption_config.json').read_text())

def filter_events(events,settings):
 selected=[]
 for event in sorted(events,key=lambda e:-e['onset_probability']):
  a,b=event['start_s'],event['end_s']
  if event['onset_probability']<settings['threshold'] or b-a<settings['min_duration_s']:continue
  if settings.get('nms',True) and any(max(0,min(b,e['end_s'])-max(a,e['start_s']))>settings['nms_overlap_fraction']*min(b-a,e['end_s']-e['start_s']) for e in selected):continue
  selected.append(event)
 return sorted(selected,key=lambda e:e['start_s'])

def caption_view(prediction,baseline,settings):
 return caption_view_from_scores(caption_scores(prediction),baseline,settings)

def caption_view_from_scores(scores,baseline,settings):
 p=copy.deepcopy(scores);p['blend']=None
 if settings.get('omit_genuineness_wording'):p['genu']=None
 if settings.get('omit_demographic_wording'):p['dims']={k:v for k,v in p['dims'].items() if k not in {'AGEV','GEND','REGS'}}
 minimum=settings.get('min_emotion_z')
 if minimum is not None:
  p['emo']={k:v for k,v in p['emo'].items() if k in baseline and baseline[k].get('std',0)>1e-6 and (v-baseline[k]['mean'])/baseline[k]['std']>=minimum}
 return p
