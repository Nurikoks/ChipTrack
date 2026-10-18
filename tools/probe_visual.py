"""Diagnostic questions on labelled development clips; never used by production inference."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config
from app.pipeline.models import VLM,crop

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
labels={r['video_id']:r for r in csv.DictReader((config.ROOT/'data/labels_enriched.csv').open(encoding='utf-8-sig'))}
baseline={r['video_id']:r for r in map(json.loads,(config.WORK_DIR/'reports/baseline/results.jsonl').read_text(encoding='utf-8').splitlines())}
vlm=VLM(args.model)
rows=[]
def frame(vid,t):
    cap=cv2.VideoCapture(str(config.ROOT/'videos'/labels[vid]['file']))
    t=min(t,cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(1,cap.get(cv2.CAP_PROP_FPS))-.1)
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
    ok,f=cap.read(); cap.release()
    if not ok:raise ValueError((vid,t))
    return f
def check(vid,t,anchor,key,question):
    f=frame(vid,t)
    box=baseline[vid]['debug']['scene'].get(anchor) if anchor else None
    im=crop(f,box,margin=.3,min_side=640) if box else VLM.to_pil(f)
    start=time.time(); score=vlm.p_yes(im,question)
    r={'video':vid,'t':t,'crop':anchor,'question':key,'score':round(score,5),'seconds':round(time.time()-start,2)}
    rows.append(r); print(json.dumps(r),flush=True)
    Path(args.output).write_text(json.dumps(rows,indent=2),encoding='utf-8')

for vid,ts in [('X-1',[20,22,24]),('X-46',[9,11,13])]:
    for t in ts:
        for anchor in [None,'desk']:
            check(vid,t,anchor,'cup','Is the cashier actually holding a drinking cup or mug, with a visible rim or handle? Do not confuse folded banknotes with a cup.')
            check(vid,t,anchor,'banknotes','Is the cashier holding paper banknotes in her hands?')
for vid in ['X-22','X-23']:
    for t in [17,20]:
        check(vid,t,'safe','key','Is a physical metal key inserted into the lock of the safe? An empty dark keyhole is not a key.')
        check(vid,t,'safe','empty_lock','Is the keyhole of the safe empty, without any inserted key?')
for vid in ['X-2','X-3']:
    for t in [3,5,9]:
        check(vid,t,'desk','identity','Is the cashier holding or inspecting a small identity card or passport? Exclude banknotes, receipts and ordinary paper.')
    for t in [8,9,10,11,12]:
        check(vid,t,'desk','id_inspected','Is the cashier herself holding and looking closely at a small plastic identity card with a portrait photo, or at a passport? A stack of paper banknotes is not an identity document.')
for vid in ['X-1','X-3']:
    for times in [[0,2,4,6],[6,8,10,12],[12,14,16,18],[18,20,22,24]]:
        frames=[frame(vid,t) for t in times if t<float(labels[vid].get('duration_sec') or 1000)]
        box=baseline[vid]['debug']['scene']['desk']
        images=[crop(f,box,margin=.65,min_side=640) for f in frames]
        for key,q in [('cash_in','Is the client passing paper banknotes to the cashier?'),('cash_out','Is the cashier handing paper banknotes back to the client?')]:
            score=vlm.p_sequence(images,q)
            r={'video':vid,'times':times,'question':key,'score':round(score,5)}
            rows.append(r);print(json.dumps(r),flush=True)
            Path(args.output).write_text(json.dumps(rows,indent=2),encoding='utf-8')
