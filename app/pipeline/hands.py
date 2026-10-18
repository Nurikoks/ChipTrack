"""Independent wrist geometry for candidate handovers. No video IDs or fixed camera zones."""
import cv2
import numpy as np


class HandContact:
    def __init__(self,path):
        import mediapipe as mp
        self.mp=mp
        options=mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),num_poses=1,
            min_pose_detection_confidence=.3,min_pose_presence_confidence=.3)
        self.model=mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def iou(a,b):
        lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=float(np.prod(np.maximum(0,hi-lo)))
        return inter/max(1.,float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-inter)

    def pose(self,frame,box):
        h,w=frame.shape[:2];bw,bh=box[2:]-box[:2]
        x1,y1=max(0,int(box[0]-.1*bw)),max(0,int(box[1]-.1*bh))
        x2,y2=min(w,int(box[2]+.1*bw)),min(h,int(box[3]+.1*bh))
        roi=frame[y1:y2,x1:x2]
        if min(roi.shape[:2])<20:return None
        result=self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)))
        if not result.pose_landmarks:return None
        p=np.array([[lm.x*(x2-x1)+x1,lm.y*(y2-y1)+y1,lm.visibility,lm.presence] for lm in result.pose_landmarks[0]])
        # Reject a second person's pose that intruded into an overlapping crop.
        if not (box[0]-.1*bw <= p[0,0] <= box[2]+.1*bw and box[1]-.1*bh <= p[0,1] <= box[1]+.55*bh):
            return None
        return p

    def check(self,clip,start,end,people,scene):
        if 'desk' not in scene or 'laptop' not in scene:return {'status':'unknown','reason':'missing_anchors'}
        desk=np.array(scene['desk']);lap=np.array(scene['laptop']);dc=(desk[:2]+desk[2:])/2;lc=(lap[:2]+lap[2:])/2
        scale=max(1.,float(np.linalg.norm(desk[2:]-desk[:2])))
        ids=sorted(set(int(np.argmin(abs(np.asarray(clip.times)-t))) for t in np.arange(start,end+.1,.5)))
        previous=None;rows=[]
        for i in ids:
            boxes=[np.array(p['box'],dtype=float) for p in people[i]]
            if len(boxes)<2:continue
            if previous is None:
                heads=[np.array([(b[0]+b[2])/2,b[1]+.15*(b[3]-b[1])]) for b in boxes]
                worker=min(range(len(boxes)),key=lambda j:np.linalg.norm(heads[j]-lc))
                visitor=min([j for j in range(len(boxes)) if j!=worker],key=lambda j:np.linalg.norm(heads[j]-dc))
            else:
                visitor=max(range(len(boxes)),key=lambda j:self.iou(boxes[j],previous[1]))
                others=[j for j in range(len(boxes)) if j!=visitor]
                worker=max(others,key=lambda j:self.iou(boxes[j],previous[0]))
                if self.iou(boxes[visitor],previous[1])<.2 or self.iou(boxes[worker],previous[0])<.2:continue
            previous=(boxes[worker],boxes[visitor])
            wp=self.pose(clip.frames[i],boxes[worker]);vp=self.pose(clip.frames[i],boxes[visitor])
            if wp is None or vp is None:continue
            if np.linalg.norm(wp[[11,12],:2].mean(axis=0)-vp[[11,12],:2].mean(axis=0))<.08*scale:continue
            wh=[p[:2] for p in wp[[15,16]] if min(p[2:])>=.4]
            vh=[p[:2] for p in vp[[15,16]] if min(p[2:])>=.4]
            gap=min((float(np.linalg.norm(a-b)/scale) for a in wh for b in vh),default=None)
            if gap is not None:rows.append({'index':i,'t':round(clip.times[i],2),'gap':round(gap,4)})
        near=[r for r in rows if r['gap']<=.12]
        for a,b in zip(near,near[1:]):
            if b['t']-a['t']<=1.1 and any(r['t']<a['t'] and r['gap']>=.17 for r in rows):
                best=min((a,b),key=lambda r:r['gap'])
                return {'status':'contact','index':best['index'],'time':best['t'],'gap':best['gap'],'observations':rows}
        return {'status':'separate' if len(rows)>=4 else 'unknown','observations':rows}
