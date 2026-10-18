"""Check the deployed HTTPS route without printing keys, passwords or cookies."""
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
import cv2
import numpy as np
from jury_api_audit import validate

root=Path(__file__).resolve().parents[1]
runtime=root/'runtime'
base=(runtime/'public-url.txt').read_text(encoding='utf-8-sig').strip()
assert base.startswith('https://') and base.endswith('.trycloudflare.com')
report={'base':base,'input':'artificial test pattern, no bank footage','started_utc':datetime.now(timezone.utc).isoformat(),'checks':[]}
def check(name,passed,**details):
    report['checks'].append({'name':name,'passed':bool(passed),**details})
    print(name, 'PASS' if passed else 'FAIL',flush=True)
    if not passed: raise RuntimeError('Failed check: '+name)

with httpx.Client(base_url=base,timeout=40) as client:
    response=client.get('/health');response.raise_for_status()
    check('public health',response.json().get('status')=='ready',health=response.json())
    response=client.get('/')
    check('public HTML',response.status_code==200 and 'ChipTRick' in response.text)
    for route in ['/ui/jobs','/ui/admin/versions','/ui/admin/feedback','/ui/admin/logs/all']:
        check('anonymous blocked '+route,client.get(route).status_code==401)
    for route in ['/runtime/api-key.txt','/runtime/admin-login.txt','/.git/config']:
        check('private file blocked '+route,client.get(route).status_code==404)
    check('API requires key',client.post('/api/v1/analyze',files={'file':('test.mov',b'test')}).status_code==401)
    credentials_path=runtime/'jury-web-login.json'
    if credentials_path.exists():
        credentials=json.loads(credentials_path.read_text(encoding='utf-8'))
    else:
        credentials={'username':'jury_demo_'+secrets.token_hex(3),'password':secrets.token_urlsafe(24)}
        response=client.post('/auth/register',json=credentials,headers={'Origin':base})
        response.raise_for_status()
        credentials_path.write_text(json.dumps(credentials),encoding='utf-8')
    (runtime/'jury-web-login.txt').write_text('ChipTRick ordinary demo user\nURL: '+base+'\nLogin: '+credentials['username']+'\nPassword: '+credentials['password']+'\n',encoding='utf-8')
    response=client.post('/auth/login',json=credentials,headers={'Origin':base})
    check('ordinary login',response.status_code==200)
    cookie=response.headers.get('set-cookie','').lower()
    check('HTTPS cookie security','secure' in cookie and 'httponly' in cookie and 'samesite=lax' in cookie)
    check('ordinary role',client.get('/auth/me').json().get('role')=='user')
    check('own history',client.get('/ui/jobs').status_code==200)
    for route in ['/ui/admin/versions','/ui/admin/feedback','/ui/admin/api-jobs','/ui/admin/logs/all']:
        check('ordinary user blocked '+route,client.get(route).status_code==403)
    check('ordinary user cannot train',client.post('/ui/admin/train',json={},headers={'Origin':base}).status_code==403)
    check('cross-origin blocked',client.post('/auth/logout',headers={'Origin':'https://unrelated.invalid'}).status_code==403)
    key=(runtime/'api-key.txt').read_text(encoding='utf-8').strip()
    started=time.monotonic()
    synthetic=runtime/'reports'/'synthetic-public-test.mov'
    writer=cv2.VideoWriter(str(synthetic),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,360))
    if not writer.isOpened():raise RuntimeError('Cannot create synthetic video')
    for frame in range(90):
        canvas=np.full((360,640,3),235,dtype=np.uint8)
        cv2.putText(canvas,'ChipTRick synthetic API test',(35,80),cv2.FONT_HERSHEY_SIMPLEX,.8,(30,30,30),2)
        cv2.putText(canvas,'No bank footage or personal data',(35,130),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),2)
        cv2.circle(canvas,(60+frame*5,240),20,(90,80,180),-1)
        writer.write(canvas)
    writer.release()
    with synthetic.open('rb') as video:
        response=client.post('/api/v1/analyze',headers={'X-API-Key':key},files={'file':('synthetic-public-test.mov',video,'video/quicktime')})
    response.raise_for_status();jid=response.json()['job_id']
    while time.monotonic()-started<300:
        response=client.get('/api/v1/jobs/'+jid,headers={'X-API-Key':key})
        response.raise_for_status();result=response.json()
        if result['status']!='processing':break
        time.sleep(2)
    elapsed=round(time.monotonic()-started,2)
    terminal=(result['status']=='done' and not validate(result)) or (result['status']=='failed' and result.get('error')=='insufficient_evidence')
    check('public API synthetic video',terminal and elapsed<300,
          job_id=jid,seconds=elapsed,verdict=result.get('verdict'),model_version=result.get('model_version'),
          status=result['status'],error=result.get('error'),
          events=[{'type':v['type_id'],'time':v['timecode_sec']} for v in result.get('violations',[])])
    check('API job private from ordinary user',client.get('/ui/jobs/'+jid).status_code==404)
    (runtime/'reports'/'public-api-response.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    client.post('/auth/logout',headers={'Origin':base})
report['passed']=all(r['passed'] for r in report['checks'])
report['finished_utc']=datetime.now(timezone.utc).isoformat()
(runtime/'reports'/'public-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PUBLIC SMOKE PASSED:',report['passed'],flush=True)
