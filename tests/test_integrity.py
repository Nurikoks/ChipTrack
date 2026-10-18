import json
import sqlite3
import time
import pytest
from fastapi.testclient import TestClient
from app import config, store, versions


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'WORK_DIR',tmp_path)
    monkeypatch.setenv('CHIPTRICK_TEST_MODE','1')
    from app.main import app
    with TestClient(app) as c:
        yield c


def account(c,name):
    payload={'username':name,'password':'a-real-test-password'}
    assert c.post('/auth/register',json=payload).status_code==200
    assert c.post('/auth/login',json=payload).status_code==200
    return c.get('/auth/me').json()


def job(owner,path='sample.mp4'):
    result={'verdict':'no_violation','violations':[],'video_duration_sec':30,'_windows':[]}
    store.execute("INSERT INTO jobs(id,owner,filename,path,sha256,created,status,stage,result,version) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  ('job1',owner,path,path,'abc',time.time(),'done','done',json.dumps(result),versions.BASE_ID))


def test_feedback_once_and_ownership(client):
    alice=account(client,'alice'); job(alice['id'])
    body={'verdict':'violation','events':[{'type_id':'T04_KEY_LEFT_IN_SAFE','start':10,'end':15}],'note':'Ключ остался'}
    assert client.post('/ui/jobs/job1/feedback',json=body).status_code==200
    assert client.post('/ui/jobs/job1/feedback',json=body).status_code==409
    row=client.get('/ui/jobs/job1').json()
    assert row['result']['verdict']=='no_violation'
    assert row['feedback']['verdict']=='violation'
    with pytest.raises(sqlite3.IntegrityError):
        store.execute("UPDATE feedback SET verdict='no_violation' WHERE job_id='job1'")
    account(client,'bob')
    assert client.get('/ui/jobs/job1').status_code==404
    assert client.get('/ui/jobs/job1/video').status_code==404
    assert client.post('/ui/jobs/job1/feedback',json=body).status_code==404
    assert client.get('/ui/admin/versions').status_code==403


def test_service_account_cannot_be_registered_or_used_for_web_login(client):
    from app.auth import create_user
    for name in ['jury-api',' ADMIN ']:
        assert client.post('/auth/register',json={'username':name,'password':'safe-test-password'}).status_code==400
    create_user('jury-api','safe-test-password')
    assert client.post('/auth/login',json={'username':'jury-api','password':'safe-test-password'}).status_code==401


def test_invalid_intervals_and_csrf(client):
    a=account(client,'alice'); job(a['id'])
    for start,end in ((20,10),(1,31),(-1,3)):
        r=client.post('/ui/jobs/job1/feedback',json={'verdict':'violation','events':[{'type_id':'T04_KEY_LEFT_IN_SAFE','start':start,'end':end}]})
        assert r.status_code in (400,422)
    assert client.post('/auth/logout',json={},headers={'Origin':'https://attacker.invalid'}).status_code==403
    assert client.get('/ui/jobs/job1').status_code==200


def test_release_requires_passing_report(client):
    versions.initialize()
    manifest=json.dumps(versions.active()['manifest'])
    store.execute('INSERT INTO versions VALUES(?,?,?,?,?,?,?)',('bad',time.time(),versions.BASE_ID,'candidate',manifest,json.dumps({'release_eligible':False,'gate_reasons':['worse']}),None))
    with pytest.raises(ValueError,match='worse'):versions.release('bad','test','do not deploy')
    assert versions.active()['id']==versions.BASE_ID


def test_metrics_match_types_and_times_one_to_one():
    from app.metrics import evaluate
    r={'events':[{'type':'A','start':10,'end':12},{'type':'A','start':10,'end':12}],
       'preds':[{'type':'A','t':11},{'type':'A','t':28},{'type':'B','t':11}], 'verdict':'violation'}
    m=evaluate([r])
    assert (m['event_tp'],m['event_fn'],m['event_fp'])==(1,1,2)
    m=evaluate([{'events':[{'type':'A','start':1,'end':2}],'preds':[],'verdict':'needs_review'}])
    assert m['fn']==1 and m['recall']==0 and m['needs_review']==1


def test_training_reduces_loss_and_roundtrips():
    import numpy as np
    from app.training import fit
    from app.pipeline.features import predict,NAMES
    rng=np.random.default_rng(9)
    x=rng.normal(size=(80,len(NAMES)))
    y=(x[:,0]>0).astype(float)[:,None]
    losses=[]
    artifact=fit(x,y,['test'],callback=lambda epoch,loss:losses.append(loss))
    assert losses[-1]<losses[0]*.6
    a=predict(artifact,[{'x':r.tolist()} for r in x])
    b=predict(json.loads(json.dumps(artifact)),[{'x':r.tolist()} for r in x])
    assert np.allclose(a,b)


def test_training_choice_is_separate_from_feedback(client):
    alice=account(client,'alice'); job(alice['id'])
    assert client.get('/ui/jobs/job1').json()['training_allowed']==0
    assert client.post('/ui/jobs/job1/feedback',json={'verdict':'no_violation','events':[]}).status_code==200
    for allowed in (True,False,True):
        assert client.post('/ui/jobs/job1/training',json={'allowed':allowed}).status_code==200
        assert bool(client.get('/ui/jobs/job1').json()['training_allowed'])==allowed
    assert client.post('/ui/jobs/job1/feedback',json={'verdict':'no_violation','events':[]}).status_code==409
    account(client,'bob')
    assert client.post('/ui/jobs/job1/training',json={'allowed':True}).status_code==404


def test_jury_feedback_can_train_only_with_opt_in(client):
    from app.auth import create_user
    from app.pipeline.features import NAMES
    from app.training import dataset
    admin=account(client,'reviewer')
    store.execute("UPDATE users SET role='admin' WHERE id=?",(admin['id'],))
    jury=create_user('jury-api','some-generated-api-secret')
    job(jury['id'])
    result={'verdict':'no_violation','violations':[],'video_duration_sec':30,'_windows':[{'start':0,'end':8,'x':[0]*len(NAMES)}],
            '_extractor':versions.engine_fingerprint(),'feature_schema':'events-v1','partial':False}
    store.execute('UPDATE jobs SET result=? WHERE id=?',(json.dumps(result),'job1'))
    assert client.post('/ui/admin/api-jobs/job1/feedback',json={'verdict':'no_violation','events':[]}).status_code==200
    assert client.post('/ui/admin/feedback/job1/review',json={'status':'approved'}).status_code==200
    assert dataset()[0]==[]
    assert client.post('/ui/admin/api-jobs/job1/training',json={'allowed':True}).status_code==200
    rows,_=dataset()
    assert len(rows)==1 and rows[0]['source']=='jury_feedback'
    client.post('/ui/admin/api-jobs/job1/training',json={'allowed':False})
    assert dataset()[0]==[]


def test_candidate_cannot_release_after_consent_withdrawn(client):
    alice=account(client,'alice'); job(alice['id'])
    client.post('/ui/jobs/job1/feedback',json={'verdict':'no_violation','events':[]})
    store.execute('UPDATE jobs SET training_allowed=1 WHERE id=?',('job1',))
    store.execute('INSERT INTO reviews(job_id,status,reviewer,note,created) VALUES(?,?,?,?,?)',('job1','approved',alice['id'],'',time.time()))
    report={'release_eligible':True,'feedback_jobs':['job1']}
    store.execute('INSERT INTO versions VALUES(?,?,?,?,?,?,?)',('candidate-good',time.time(),versions.BASE_ID,'candidate',json.dumps(versions.active()['manifest']),json.dumps(report),None))
    client.post('/ui/jobs/job1/training',json={'allowed':False})
    with pytest.raises(ValueError,match='разрешение'):
        versions.release('candidate-good','test','Should remain blocked')
    assert versions.active()['id']==versions.BASE_ID
    client.post('/ui/jobs/job1/training',json={'allowed':True})
    versions.release('candidate-good','test','Verified isolated release test')
    assert versions.active()['id']=='candidate-good'
    assert store.one("SELECT version FROM jobs WHERE id='job1'")['version']==versions.BASE_ID
    versions.release(versions.BASE_ID,'test','Verified isolated rollback test',rollback=True)
    assert versions.active()['id']==versions.BASE_ID
    assert len(store.many('SELECT * FROM releases'))==2
