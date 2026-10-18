"""CPU temporal logistic heads; frozen visual models. Real epochs and grouped evaluation."""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from . import config, store, versions
from .telemetry import emit
from .pipeline.features import NAMES, predict, decisions
from .metrics import evaluate


def dataset():
    grouped = {}
    seed = config.WORK_DIR / 'dataset' / 'seed.jsonl'
    if seed.exists():
        for line in seed.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r.get('windows') and not r.get('partial') and r.get('extractor')==versions.engine_fingerprint():
                grouped[r['sha256']]=r
    approved = store.many("SELECT j.*,u.username AS source_username,f.verdict AS corrected,f.events AS corrections,f.created AS feedback_created FROM jobs j JOIN feedback f ON f.job_id=j.id JOIN reviews r ON r.job_id=j.id JOIN users u ON u.id=j.owner WHERE r.status='approved' AND j.training_allowed=1 ORDER BY f.created")
    # Conflicting feedback for identical videos is excluded, never silently voted by uploads.
    conflicts=set()
    seen={}
    for r in approved:
        signature=json.dumps(json.loads(r['corrections']),sort_keys=True)
        if r['sha256'] in seen and seen[r['sha256']] != signature:
            conflicts.add(r['sha256'])
        seen[r['sha256']]=signature
        result=json.loads(r['result'] or '{}')
        if not result.get('_windows') or result.get('partial') or result.get('feature_schema')!='events-v1' or result.get('_extractor')!=versions.engine_fingerprint(): continue
        events=[{'type':e['type_id'],'start':e['start'],'end':e['end'],'t':(e['start']+e['end'])/2} for e in json.loads(r['corrections'])]
        grouped[r['sha256']]={'video_id':r['id'],'sha256':r['sha256'],'source':'jury_feedback' if r['source_username']=='jury-api' else 'feedback','events':events,
            'windows':result['_windows'],'preds':[{'type':v['type_id'],'t':v['timecode_sec'],'conf':v['confidence']} for v in result['violations']],
            'verdict':result['verdict'],'partial':result.get('partial',False),
            'base_preds':result.get('_base_predictions',[]),'base_verdict':result.get('_base_verdict','needs_review'),
            'base_review_required':result.get('_base_review_required',bool(result.get('review_reasons')))}
    for h in conflicts: grouped.pop(h,None)
    return list(grouped.values()), sorted(conflicts)


def matrices(rows, types):
    x,y,groups=[],[],[]
    for r in rows:
        for w in r['windows']:
            x.append(w['x'])
            center=(w['start']+w['end'])/2
            y.append([float(any(e['type']==tid and e['start']-2 <= center <= e['end']+2 for e in r['events'])) for tid in types])
            groups.append(r['sha256'])
    return np.asarray(x,float),np.asarray(y,float),np.asarray(groups)


def fit(x,y,types,epochs=180,callback=None):
    mean=x.mean(axis=0)
    scale=x.std(axis=0)
    scale[scale<.05]=1
    z=(x-mean)/scale
    w=np.zeros((x.shape[1],len(types)))
    b=np.full(len(types),-2.)
    weights=np.minimum(10,(len(y)-y.sum(axis=0))/np.maximum(1,y.sum(axis=0)))
    for epoch in range(1,epochs+1):
        p=1/(1+np.exp(-np.clip(z@w+b,-35,35)))
        per_sample=np.where(y>0,weights,1.)
        error=(p-y)*per_sample
        w-=.04*(z.T@error/len(z)+.025*w)
        b-=.04*error.mean(axis=0)
        loss=float(np.mean(-(y*np.log(p+1e-8)+(1-y)*np.log(1-p+1e-8))*per_sample))
        if callback and (epoch==1 or epoch%10==0 or epoch==epochs): callback(epoch,loss)
    return {'feature_names':NAMES,'types':types,'mean':mean.tolist(),'scale':scale.tolist(),
            'weights':w.tolist(),'bias':b.tolist(),'threshold':.85,'suppress_threshold':.1,'feature_schema':'events-v1'}


def predicted_row(artifact,row):
    base=row.get('base_preds',row['preds'])
    d=decisions(artifact,row['windows'],base) if artifact else {'keep':list(range(len(base))),'additions':[],'review':[]}
    result=[]
    for p in sorted([base[i] for i in d['keep']]+d['additions'],key=lambda e:(e['type'],e['t'])):
        if result and result[-1]['type']==p['type'] and p['t']-result[-1]['t']<12: continue
        result.append(p)
    review=row.get('base_review_required',row.get('base_verdict',row['verdict']) in ('needs_review','error')) or bool(d['review'])
    return {**row,'preds':result,'verdict':'violation' if result else 'needs_review' if review else 'no_violation'}


def add_support(artifact,rows,types):
    artifact['positive_groups']={t:sum(any(e['type']==t for e in r['events']) for r in rows) for t in types}
    artifact['negative_groups']={t:len(rows)-artifact['positive_groups'][t] for t in types}


def run(run_id):
    try:
        rows,conflicts=dataset()
        types=sorted({e['type'] for r in rows for e in r['events']})
        if len(rows)<4 or not types or not any(not r['events'] for r in rows):
            raise ValueError('Нужно минимум 4 уникальных размеченных видео с полным анализом, включая нормальные и нарушения')
        signature=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        folder=config.WORK_DIR/'models'/run_id
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'dataset.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
        x,y,groups=matrices(rows,types)
        parent=versions.active()['id']
        parent_version=versions.get(parent)
        parent_artifact=json.loads(Path(parent_version['artifact']).read_text(encoding='utf-8')) if parent_version.get('artifact') else None
        for row in rows:
            # Recompute both versions from the same frozen visual features. A previous
            # uploaded verdict must not silently become an input to the next classifier.
            row.update(predicted_row(parent_artifact,row))
        emit('training','dataset_ready',run_id=run_id,parent_version=parent,candidate_version='candidate-'+run_id[:12],unique_videos=len(rows),windows=len(x),types=types,sha256=signature,excluded_conflicts=conflicts,device='CPU',frozen_models=[config.DETECTOR_ID,config.VLM_ID])
        store.execute("UPDATE training_runs SET status='training',epochs=180 WHERE id=?",(run_id,))
        def progress(epoch,loss):
            store.execute('UPDATE training_runs SET epoch=?,loss=? WHERE id=?',(epoch,loss,run_id))
            emit('training','epoch',run_id=run_id,epoch=epoch,epochs=180,loss=round(loss,6))
        artifact=fit(x,y,types,callback=progress)
        store.execute("UPDATE training_runs SET status='evaluating' WHERE id=?",(run_id,))
        add_support(artifact,rows,types)
        # Held-out groups never appear in training folds. Existing rules are already in-sample.
        ordered=sorted({r['sha256'] for r in rows},key=lambda h:hashlib.sha256(('chiptrick-split-1'+h).encode()).hexdigest())
        fold_map={h:i%min(4,len(ordered)) for i,h in enumerate(ordered)}
        oof=[]
        for fold in sorted(set(fold_map.values())):
            test=[r for r in rows if fold_map[r['sha256']]==fold]
            train=[r for r in rows if fold_map[r['sha256']]!=fold]
            xx,yy,_=matrices(train,types)
            a=fit(xx,yy,types)
            add_support(a,train,types)
            for r in test:
                prediction=predicted_row(a,r)
                oof.append({**r,'candidate_preds':prediction['preds'],'candidate_verdict':prediction['verdict']})
            emit('training','fold_evaluated',run_id=run_id,fold=fold+1,train_videos=len(train),test_videos=len(test))
        baseline=evaluate(oof)
        candidate=evaluate([{**r,'preds':r['candidate_preds'],'verdict':r['candidate_verdict']} for r in oof])
        reasons=[]
        if not (candidate['event_tp']>baseline['event_tp'] or candidate['event_fp']<baseline['event_fp'] or candidate['needs_review']<baseline['needs_review']):
            reasons.append('Нет улучшения правильных событий, ложных событий или числа ручных просмотров')
        if candidate['event_tp']<baseline['event_tp']: reasons.append('Стало меньше правильных типов в размеченных интервалах')
        if candidate['event_fp']>baseline['event_fp']: reasons.append('Стало больше ложных событий')
        if candidate['fp']>baseline['fp']: reasons.append('Стало больше ложных тревог на нормальных видео')
        if candidate['fn']>baseline['fn']: reasons.append('Стало больше пропусков')
        if candidate['needs_review']>baseline['needs_review']: reasons.append('Стало больше случаев для ручного просмотра')
        report={'evaluation':'grouped_out_of_fold_development','release_eligible':not reasons,'gate_reasons':reasons,
                'limitations':['Исходные правила настраивались на этих видео. Это оценка разработки, а не независимый тест новой камеры.',
                               'Типы с менее чем 3 положительными обучающими видео дают гипотезы для просмотра.'],
                'active_metrics':baseline,'candidate_metrics':candidate,'positive_groups':artifact['positive_groups'],
                'feedback_jobs':[r['video_id'] for r in rows if r.get('source') in ('feedback','jury_feedback')],
                'conflicting_hashes':conflicts,'dataset_sha256':signature,'rows':[{k:v for k,v in r.items() if k!='windows'} for r in oof]}
        artifact_path=folder/'classifier.json'
        artifact_path.write_text(json.dumps(artifact),encoding='utf-8')
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        vid='candidate-'+run_id[:12]
        manifest={**parent_version['manifest'],'classifier':'temporal-logistic','dataset_sha256':signature,'training_run':run_id}
        store.execute('INSERT INTO versions VALUES(?,?,?,?,?,?,?)',(vid,time.time(),parent,'candidate',json.dumps(manifest,ensure_ascii=False),json.dumps(report,ensure_ascii=False),str(artifact_path)))
        store.execute("UPDATE training_runs SET status='completed',version=? WHERE id=?",(vid,run_id))
        store.execute("INSERT INTO settings VALUES('last_training_signature',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(signature,))
        keys=[hashlib.sha256(json.dumps([r['sha256'],r['events']],sort_keys=True).encode()).hexdigest() for r in rows if r.get('source') in ('feedback','jury_feedback')]
        store.execute("INSERT INTO settings VALUES('trained_feedback_keys',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(json.dumps(keys),))
        emit('training','completed',run_id=run_id,version=vid,metrics=candidate,release_eligible=not reasons)
        emit('versions','candidate_created',version=vid,parent=parent,report=str(folder/'report.json'),release_eligible=not reasons)
    except Exception as exc:
        store.execute("UPDATE training_runs SET status='failed',error=? WHERE id=?",(str(exc),run_id))
        emit('training','failed',run_id=run_id,error=str(exc))
        raise


def launch():
    recover_runs()
    rows,_=dataset()
    if len(rows)<4 or not any(r['events'] for r in rows) or not any(not r['events'] for r in rows):
        raise ValueError('Нужно минимум 4 готовых уникальных видео, включая нарушения и нормальные случаи. Сначала выполните полный анализ набора или одобрите исправления')
    rid=uuid.uuid4().hex
    with store.db() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute("SELECT id FROM training_runs WHERE status IN ('queued','training','evaluating')").fetchone():
            raise ValueError('Обучение уже идёт')
        c.execute('INSERT INTO training_runs(id,created,status) VALUES(?,?,?)',(rid,time.time(),'queued'))
    try:
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        log=config.WORK_DIR/'logs'/'training-process.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('ab') as f:
            child=subprocess.Popen([sys.executable,'-m','app.cli','train-worker',rid],cwd=config.ROOT,stdout=f,stderr=f,env=os.environ.copy(),**kwargs)
        store.execute('UPDATE training_runs SET pid=? WHERE id=?',(child.pid,rid))
    except Exception as exc:
        store.execute("UPDATE training_runs SET status='failed',error=? WHERE id=?",(str(exc),rid))
        raise
    emit('training','queued',run_id=rid)
    return rid


def recover_runs():
    import psutil
    for r in store.many("SELECT * FROM training_runs WHERE status IN ('queued','training','evaluating')"):
        if time.time()-r['created']<30: continue
        try:
            alive=bool(r['pid']) and r['id'] in psutil.Process(r['pid']).cmdline()
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            alive=False
        if not alive:
            store.execute("UPDATE training_runs SET status='failed',error='Процесс обучения прерван; можно запустить снова' WHERE id=?",(r['id'],))
            emit('training','interrupted',run_id=r['id'])


def start_scheduler():
    def loop():
        while True:
            time.sleep(60)
            try:
                recover_runs()
                rows,_=dataset()
                old=store.one("SELECT value FROM settings WHERE key='trained_feedback_keys'")
                previous=set(json.loads(old['value'])) if old else set()
                current={hashlib.sha256(json.dumps([r['sha256'],r['events']],sort_keys=True).encode()).hexdigest() for r in rows if r.get('source') in ('feedback','jury_feedback')}
                if len(current-previous)<int(os.environ.get('CHIPTRICK_TRAIN_MIN_NEW','5')): continue
                sig=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
                last=store.one("SELECT value FROM settings WHERE key='last_training_signature'")
                if not last or last['value']!=sig: launch()
            except ValueError:
                pass
            except Exception as exc:
                emit('training','scheduler_error',error=str(exc))
    threading.Thread(target=loop,daemon=True,name='training-scheduler').start()
