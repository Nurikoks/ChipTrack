"""Real local API audit. Reads the key from a file; never logs credentials."""
import argparse
import base64
import concurrent.futures
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from PIL import Image


def validate(r):
    errors=[]
    for key in ('job_id','status','verdict','violations'):
        if key not in r:errors.append('missing '+key)
    if r.get('status')!='done':errors.append('not done')
    if r.get('verdict') not in ('violation','no_violation'):errors.append('invalid verdict')
    if (r.get('verdict')=='violation')!=bool(r.get('violations')):errors.append('verdict/list mismatch')
    for v in r.get('violations',[]):
        if not isinstance(v.get('type'),str) or not v['type']:errors.append('invalid type')
        if not isinstance(v.get('description'),str) or not v['description']:errors.append('invalid description')
        t=v.get('timecode_sec')
        if not isinstance(t,(int,float)) or not 0<=t<=r.get('video_duration_sec',float('inf')):errors.append('invalid timecode')
        frames=v.get('evidence_frames',[])
        if not 1<=len(frames)<=3:errors.append('evidence count')
        for f in frames:
            if not isinstance(f.get('timestamp_sec'),(int,float)):errors.append('frame time')
            try:
                with Image.open(io.BytesIO(base64.b64decode(f['image_base64'],validate=True))) as img:
                    if img.format!='JPEG':errors.append('not JPEG')
                    img.verify()
            except Exception as exc:errors.append('invalid JPEG: '+type(exc).__name__)
    return errors


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='http://127.0.0.1:8000')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();root=args.root
    key=(root/'runtime/api-key.txt').read_text(encoding='utf-8').strip()
    out=root/'runtime/reports/api-audit';out.mkdir(parents=True,exist_ok=True)
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'base':args.base,'checks':[],'videos':[]}
    def record(name,response,expected):
        report['checks'].append({'name':name,'http':response.status_code,'expected':expected,'passed':response.status_code==expected})
    def upload_and_wait(name,data):
        start=time.monotonic()
        with httpx.Client(base_url=args.base,headers={'X-API-Key':key},timeout=30) as c:
            response=c.post('/api/v1/analyze',files={'file':(name,data,'video/quicktime')})
            response.raise_for_status();jid=response.json()['job_id'];states=[]
            while time.monotonic()-start<330:
                response=c.get('/api/v1/jobs/'+jid);response.raise_for_status();r=response.json()
                if not states or states[-1]!=r['status']:states.append(r['status'])
                if r['status']!='processing':break
                time.sleep(2)
            else:r={'job_id':jid,'status':'timeout'}
        (out/(jid+'.json')).write_text(json.dumps(r,ensure_ascii=False),encoding='utf-8')
        row={'file':name,'job_id':jid,'states':states,'seconds_from_upload':round(time.monotonic()-start,2),'status':r['status'],
             'verdict':r.get('verdict'),'error':r.get('error'),'model_version':r.get('model_version'),
             'schema_errors':validate(r) if r['status']=='done' else [],
             'events':[{'type':v['type_id'],'time':v['timecode_sec'],'frames':len(v['evidence_frames'])} for v in r.get('violations',[])]}
        row['within_5_minutes']=row['seconds_from_upload']<=300
        print(json.dumps(row,ensure_ascii=False),flush=True)
        return row
    with httpx.Client(base_url=args.base,timeout=15) as c:
        report['health']=c.get('/health').json()
        record('POST without key',c.post('/api/v1/analyze',files={'file':('a.mov',b'test')}),401)
        record('POST wrong key',c.post('/api/v1/analyze',headers={'X-API-Key':'wrong'},files={'file':('a.mov',b'test')}),401)
        record('GET wrong key',c.get('/api/v1/jobs/unknown',headers={'X-API-Key':'wrong'}),401)
        record('unknown job',c.get('/api/v1/jobs/unknown',headers={'X-API-Key':key}),404)
        record('unsupported file',c.post('/api/v1/analyze',headers={'X-API-Key':key},files={'file':('a.txt',b'test')}),415)
        record('empty video',c.post('/api/v1/analyze',headers={'X-API-Key':key},files={'file':('a.mov',b'')}),400)
        record('missing file',c.post('/api/v1/analyze',headers={'X-API-Key':key}),422)
    report['corrupt_video']=upload_and_wait('audit-corrupt.mov',b'not a valid movie')
    # Source filenames mix Latin X and Cyrillic Х. Resolve all inputs before submitting.
    available={p.name.replace('Х','X'):p for p in (root/'videos').glob('*.mov')}
    paths=[available[name] for name in ['X-1.mov','X-3.mov','X-22.mov','X-23.mov','X-46.mov']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures=[executor.submit(upload_and_wait,path.name,path.read_bytes()) for path in paths]
        report['videos']=[f.result() for f in futures]
    report['passed']=all(r['passed'] for r in report['checks']) and report['corrupt_video']['status']=='failed' and all(r['status']=='done' and not r['schema_errors'] and r['within_5_minutes'] for r in report['videos'])
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AUDIT PASSED:',report['passed'],flush=True)


if __name__=='__main__':main()
