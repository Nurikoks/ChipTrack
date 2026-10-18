"""Event matching is one-to-one, same type and inside the labelled interval (+2 seconds)."""
from collections import Counter


def evaluate(rows, prediction_key='preds'):
    tp=fp=fn=tn=review=event_tp=event_fp=event_fn=0
    types = {}
    for r in rows:
        preds = r.get(prediction_key, [])
        truth = r.get('events', [])
        positive, detected = bool(truth), bool(preds)
        tp += int(positive and detected)
        fp += int(not positive and detected)
        fn += int(positive and not detected)
        unresolved = r.get('verdict') in ('needs_review','error') or bool(r.get('partial'))
        tn += int(not positive and not detected and not unresolved)
        review += int(unresolved)
        # Maximum cardinality bipartite matching; prevents double-counting one prediction.
        adjacent = []
        for e in truth:
            lo, hi = e.get('start',e.get('t',0))-2, e.get('end',e.get('t',0))+2
            adjacent.append([i for i,p in enumerate(preds) if p['type']==e['type'] and lo <= p['t'] <= hi])
        assigned = {}
        def visit(k, seen):
            for i in adjacent[k]:
                if i in seen: continue
                seen.add(i)
                if i not in assigned or visit(assigned[i], seen):
                    assigned[i] = k
                    return True
            return False
        for k in range(len(truth)): visit(k, set())
        matched = set(assigned.values())
        event_tp += len(matched)
        event_fn += len(truth)-len(matched)
        event_fp += len(preds)-len(matched)
        for k,e in enumerate(truth):
            d=types.setdefault(e['type'], {'events':0,'matched':0})
            d['events']+=1
            d['matched']+=int(k in matched)
    ratio=lambda a,b: round(a/b,4) if b else None
    return {'videos':len(rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'needs_review':review,
            'recall':ratio(tp,tp+fn),'precision':ratio(tp,tp+fp),
            'event_tp':event_tp,'event_fp':event_fp,'event_fn':event_fn,
            'event_recall':ratio(event_tp,event_tp+event_fn),'event_precision':ratio(event_tp,event_tp+event_fp),
            'coverage':ratio(len(rows)-review,len(rows)), 'per_type':types}
