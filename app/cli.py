r"""Local operator commands. Run: .venv\Scripts\python -m app.cli --help"""
import argparse
import csv
import getpass
import json
import logging
import os
import secrets
import sys
import time
from collections import deque
from pathlib import Path
from . import config, store, versions
from .telemetry import emit


def show(value):
    print(json.dumps(value,ensure_ascii=False,indent=2,default=str),flush=True)


def tail(stream,follow=False,lines=30):
    path=config.WORK_DIR/'logs'/f'{stream}.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.touch(exist_ok=True)
    def render(line):
        try:
            d=json.loads(line)
            stamp=d.pop('time'); channel=d.pop('stream'); event=d.pop('event'); d.pop('pid',None)
            print(f"{stamp[:19].replace('T',' ')} UTC | {channel:9} | {event:24} | "+json.dumps(d,ensure_ascii=False),flush=True)
        except json.JSONDecodeError:
            print(line.rstrip(),flush=True)
    with path.open(encoding='utf-8') as f:
        for line in deque(f,maxlen=lines): render(line)
        while follow:
            pos=f.tell(); line=f.readline()
            if line.endswith('\n'): render(line)
            else:
                f.seek(pos); time.sleep(.3)


def evaluate_set(only=None):
    from types import SimpleNamespace
    from .pipeline.analyze import Analyzer
    from .metrics import evaluate
    labels=list(csv.DictReader((config.ROOT/'data'/'labels_enriched.csv').open(encoding='utf-8-sig')))
    groups={}
    for r in labels:
        path=config.ROOT/'videos'/r['file']
        if not path.exists(): raise ValueError(f'Видео не найдено: {path}')
        sha=store.digest_file(path)
        g=groups.setdefault(sha,{'video_id':r['video_id'],'aliases':[],'file':r['file'],'sha256':sha,'events':[]})
        g['aliases'].append(r['video_id'])
        if r['label']=='violation':
            g['events'].append({'type':r['type_candidate'],'t':float(r['timestamp_sec']),
                               'start':float(r['event_start'] or r['timestamp_sec']),'end':float(r['event_end'] or r['timestamp_sec'])})
    out=config.WORK_DIR/'reports'/'current'
    out.mkdir(parents=True,exist_ok=True)
    (out/'inventory.json').write_text(json.dumps(list(groups.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    analyzer=Analyzer()
    rows=[]
    for g in groups.values():
        if only and not any(a in only for a in g['aliases']): continue
        vid=versions.active()['id']
        job=SimpleNamespace(id=g['video_id'],deadline=time.time()+config.JOB_DEADLINE_SEC,stage='',progress=0,version=vid)
        emit('inference','evaluation_started',video=g['video_id'],version=vid)
        try:
            saved=out/(g['video_id']+'.json')
            cached=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else None
            if cached and cached.get('_extractor')==versions.engine_fingerprint() and cached.get('model_version')==vid and not cached.get('partial'):
                result=cached
                emit('inference','evaluation_cache_hit',video=g['video_id'],version=vid)
            else:
                result=analyzer.analyze(config.ROOT/'videos'/g['file'],job)
                result['model_version']=vid
            row={**g,'verdict':result['verdict'],'partial':result['partial'],'windows':result['_windows'],
                 'preds':[{'t':v['timecode_sec'],'type':v['type_id'],'conf':v['confidence']} for v in result['violations']],
                 'seconds':result['processing_time_sec'],'review_reasons':result['review_reasons'], 'extractor':result['_extractor'],
                 'base_preds':result['_base_predictions'],'base_verdict':result['_base_verdict'],
                 'base_review_required':result['_base_review_required']}
            (out/(g['video_id']+'.json')).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        except Exception as exc:
            row={**g,'verdict':'error','partial':True,'windows':[],'preds':[],'error':str(exc)}
            logging.exception('evaluation failed')
        rows.append(row)
        emit('inference','evaluation_result',video=g['video_id'],verdict=row['verdict'],preds=row['preds'],completed=len(rows),total=len(groups))
        (out/'results.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    metrics=evaluate(rows)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    if not only:
        folder=config.WORK_DIR/'dataset'
        folder.mkdir(exist_ok=True)
        (folder/'seed.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    show(metrics)


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description='ChipTRick: модель, обучение, версии и журналы')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('status'); sub.add_parser('versions'); sub.add_parser('changes'); sub.add_parser('train')
    a=sub.add_parser('logs'); a.add_argument('stream',choices=['inference','training','versions','all']); a.add_argument('--follow',action='store_true'); a.add_argument('--lines',type=int,default=30)
    a=sub.add_parser('report'); a.add_argument('version',nargs='?')
    a=sub.add_parser('release'); a.add_argument('version'); a.add_argument('--reason',required=True)
    a=sub.add_parser('rollback'); a.add_argument('version'); a.add_argument('--reason',required=True)
    a=sub.add_parser('create-user'); a.add_argument('username'); a.add_argument('--admin',action='store_true')
    sub.add_parser('bootstrap')
    a=sub.add_parser('train-worker'); a.add_argument('run_id')
    a=sub.add_parser('evaluate'); a.add_argument('--only',nargs='*')
    args=p.parse_args(); versions.initialize()
    if args.command=='status':
        show({'active':versions.active(),'jobs':store.many('SELECT status,COUNT(*) AS count FROM jobs GROUP BY status'),
              'training':store.many('SELECT * FROM training_runs ORDER BY created DESC LIMIT 3'),'logs':str(config.WORK_DIR/'logs')})
    elif args.command=='versions':
        show([store.unpack(r) for r in store.many('SELECT id,created,parent,status,manifest FROM versions ORDER BY created DESC')])
    elif args.command=='changes': show(store.many('SELECT * FROM releases ORDER BY created DESC'))
    elif args.command=='logs': tail(args.stream,args.follow,args.lines)
    elif args.command=='report':
        v=versions.get(args.version) if args.version else store.unpack(store.one('SELECT * FROM versions WHERE report IS NOT NULL ORDER BY created DESC LIMIT 1'))
        show(v['report'] if v else {'message':'Обученных кандидатов пока нет'})
    elif args.command in ('release','rollback'):
        show(versions.release(args.version,getpass.getuser(),args.reason,rollback=args.command=='rollback'))
    elif args.command=='create-user':
        from .auth import create_user
        show(create_user(args.username,getpass.getpass('Пароль (минимум 10 символов): '),'admin' if args.admin else 'user'))
    elif args.command=='bootstrap':
        from .auth import create_user
        key_file=config.WORK_DIR/'api-key.txt'
        if not key_file.exists(): key_file.write_text(secrets.token_urlsafe(32),encoding='utf-8')
        if not store.one("SELECT id FROM users WHERE role='admin'"):
            password=secrets.token_urlsafe(18)
            create_user('admin',password,'admin')
            f=config.WORK_DIR/'admin-login.txt'
            f.write_text('ChipTRick local administrator\nURL: http://127.0.0.1:8000\nLogin: admin\nPassword: '+password+'\n',encoding='utf-8')
            print('Данные входа сохранены локально: '+str(f))
        else: print('Администратор уже создан')
    elif args.command=='train':
        from .training import launch
        show({'run_id':launch()})
    elif args.command=='train-worker':
        from .training import run
        run(args.run_id)
    elif args.command=='evaluate': evaluate_set(args.only)

if __name__=='__main__':
    main()
