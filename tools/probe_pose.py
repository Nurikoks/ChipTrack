"""Inspect real wrist landmarks on development clips; no VLM needed."""
import csv,json,sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
labels={r['video_id']:r for r in csv.DictReader((ROOT/'data/labels_enriched.csv').open(encoding='utf-8-sig'))}
baseline={r['video_id']:r for r in map(json.loads,(ROOT/'runtime/reports/baseline/results.jsonl').read_text(encoding='utf-8').splitlines())}
options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/pose/pose_landmarker_full.task')),num_poses=4,min_pose_detection_confidence=.3,min_pose_presence_confidence=.3)
det=mp.tasks.vision.PoseLandmarker.create_from_options(options)
all_rows=[]
for vid,start in [('X-1',18),('X-4',30),('X-3',12),('X-5',30)]:
    scene=baseline[vid]['debug']['scene']; desk=np.array(scene['desk']); laptop=np.array(scene['laptop']); lp=(laptop[:2]+laptop[2:])/2; dc=(desk[:2]+desk[2:])/2; scale=np.linalg.norm(desk[2:]-desk[:2]); rows=[]
    cap=cv2.VideoCapture(str(ROOT/'videos'/labels[vid]['file']))
    sheet=Image.new('RGB',(7*250,400),'white');draw=ImageDraw.Draw(sheet)
    for k,t in enumerate(range(start,start+7)):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
        if not ok:continue
        h,w=frame.shape[:2];poses=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))).pose_landmarks
        ps=[np.array([[lm.x*w,lm.y*h,lm.visibility,lm.presence] for lm in p]) for p in poses]
        centers=[p[[11,12],:2].mean(axis=0) for p in ps]
        row={'t':t,'poses':len(ps)}
        if len(ps)>=2:
            worker=min(range(len(ps)),key=lambda i:np.linalg.norm(centers[i]-lp))
            visitor=min([i for i in range(len(ps)) if i!=worker],key=lambda i:np.linalg.norm(centers[i]-dc))
            row.update(visitor_wrists=ps[visitor][[15,16]].round(3).tolist(),visitor_center=centers[visitor].round(3).tolist(),worker_center=centers[worker].round(3).tolist())
            for index,color in [(worker,(0,210,0)),(visitor,(0,0,255))]:
                for j in [11,12,13,14,15,16]:
                    if ps[index][j,2]>=.3: cv2.circle(frame,tuple(ps[index][j,:2].astype(int)),4,color,-1)
        rows.append(row)
        x1,y1,x2,y2=desk.astype(int);dw=x2-x1;dh=y2-y1
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im=im.crop((max(0,int(x1-dw*.25)),max(0,int(y1-dh*.8)),min(w,int(x2+dw*.25)),min(h,int(y2+dh*.6))));im.thumbnail((250,370));sheet.paste(im,(k*250,25));draw.text((k*250+5,5),f'{vid} {t}s',fill='black')
    cap.release()
    motion=[]
    for hand in [0,1]:
        points=[np.array(r['visitor_wrists'][hand][:2])-np.array(r['visitor_center']) for r in rows if 'visitor_wrists' in r and min(r['visitor_wrists'][hand][2:])>=.5]
        motion.append(max((np.linalg.norm(a-b)/scale for a in points for b in points),default=0.))
    result={'video':vid,'motion':motion,'rows':rows};all_rows.append(result);print(json.dumps(result),flush=True)
    sheet.save(ROOT/f'runtime/reports/pose-{vid}.jpg')
    (ROOT/'runtime/reports/pose-probe.json').write_text(json.dumps(all_rows,indent=2),encoding='utf-8')
det.close()
