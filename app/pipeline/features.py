"""Small temporal feature vectors, shared exactly between training and inference."""
import json
import numpy as np
from .. import versions

KEYS = ['money','bag','docs','screen','safe_open','key','personal','drinking','door_open','id_shown',
        'two_visitors','cash_in','cash_out','machine','band','cash_safe','mixed','visitor_seated','occupied','cup_in_hand','banknotes_in_hand','hand_contact']
NAMES = [f'{key}_{stat}' for key in KEYS for stat in ('max','mean','seen')] + ['people_mean','attended','safe_near','service']


def make_windows(duration, trace, contexts, tl):
    result = []
    for start in np.arange(0, max(duration, .1), 4):
        end = min(start+8, duration)
        row = []
        for key in KEYS:
            values = [p for t,k,p in trace if k == key and start <= t <= end]
            row.extend([max(values), float(np.mean(values)), 1.] if values else [0.,0.,0.])
        mask = (tl['t'] >= start) & (tl['t'] <= end)
        row += [float(np.mean(tl[k][mask])) if mask.any() else 0. for k in ('n','attended','beside_safe')]
        row += [float(any(c['context']=='service' and c['start'] <= end and c['end'] >= start for c in contexts))]
        result.append({'start': round(float(start),1), 'end': round(float(end),1), 'x': row})
    return result


def predict(artifact, windows):
    if not windows:
        return np.empty((0, len(artifact['types'])))
    x = np.asarray([w['x'] for w in windows], dtype=float)
    x = (x-np.asarray(artifact['mean'])) / np.asarray(artifact['scale'])
    z = x @ np.asarray(artifact['weights']) + np.asarray(artifact['bias'])
    return 1 / (1 + np.exp(-np.clip(z,-35,35)))


def decisions(artifact, windows, base):
    """The same refinement policy is used in production and held-out evaluation."""
    scores=predict(artifact,windows)
    positions={tid:i for i,tid in enumerate(artifact['types'])}
    keep=[]; suppressed=[]; additions=[]; review=[]
    for i,event in enumerate(base):
        tid=event['type']; j=positions.get(tid)
        enough=artifact.get('positive_groups',{}).get(tid,0)>=3 and artifact.get('negative_groups',{}).get(tid,0)>=5
        local=[float(scores[k,j]) for k,w in enumerate(windows) if j is not None and w['start']<=event['t']<=w['end']]
        if enough and local and max(local)<artifact.get('suppress_threshold',.1):
            suppressed.append({**event,'classifier_score':max(local)})
        else:keep.append(i)
    for w,ps in zip(windows,scores):
        for tid,p in zip(artifact['types'],ps):
            if p<artifact.get('threshold',.85):continue
            if artifact.get('positive_groups',{}).get(tid,0)<3:
                review.append({'type':tid,'start':w['start'],'end':w['end']})
            else:additions.append({'type':tid,'t':(w['start']+w['end'])/2,'conf':float(p)})
    return {'keep':keep,'suppressed':suppressed,'additions':additions,'review':review}


def apply_classifier(windows, vid, clip, findings):
    if not vid:
        return findings, []
    v = versions.get(vid)
    if not v or not v.get('artifact'):
        return findings, []
    from pathlib import Path
    artifact = json.loads(Path(v['artifact']).read_text(encoding='utf-8'))
    if artifact['feature_names'] != NAMES:
        raise ValueError('Classifier feature schema mismatch')
    from .analyze import Finding
    from .video import nearest_index
    from .rules import TYPE_NAMES
    d=decisions(artifact,windows,[{'type':f.type_id,'t':f.t,'conf':f.conf} for f in findings])
    out=[findings[i] for i in d['keep']]
    issues=[f"{r['start']}–{r['end']} с: гипотеза «{TYPE_NAMES[r['type']]}»; слишком мало независимых обучающих примеров" for r in d['review']]
    for r in d['additions']:
        tid,t=r['type'],r['t']
        out.append(Finding(tid,t,r['conf'],f"Временной классификатор выделил признаки события «{TYPE_NAMES[tid]}»; требуется сверка с кадрами",
                           [(nearest_index(clip.times,t),None)]))
    if d['suppressed']:
        from ..telemetry import emit
        emit('inference','classifier_rejected_events',version=vid,events=d['suppressed'])
    return out, issues
