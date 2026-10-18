import json
import hashlib
import time
from pathlib import Path
from . import config, store
from .telemetry import emit

def engine_fingerprint():
    h = hashlib.sha256()
    for path in sorted((config.ROOT / 'app' / 'pipeline').glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    h.update(config.RULES_PATH.read_bytes())
    h.update(str(config.SAMPLE_FPS).encode())
    h.update(config.DETECTOR_ID.encode()); h.update(config.VLM_ID.encode())
    h.update(str((config.VLM_4BIT,config.EXPERIMENTAL)).encode())
    for model_path in (config.DETECTOR_ID,config.VLM_ID):
        source = Path(model_path) / 'chiptrick-source.json'
        if source.exists(): h.update(source.read_bytes())
    h.update(str(config.POSE_PATH).encode())
    pose_source=config.POSE_PATH.parent/'chiptrick-source.json'
    if pose_source.exists():h.update(pose_source.read_bytes())
    return h.hexdigest()

BASE_ID = "rules-2.0-" + engine_fingerprint()[:8]


def initialize():
    store.init()
    manifest = {"engine": "temporal-2", "detector": config.DETECTOR_ID, "vlm": config.VLM_ID,"pose":str(config.POSE_PATH),
                "classifier": None, "confidence": "uncalibrated", "feature_schema": "events-v1", "engine_fingerprint": engine_fingerprint(),
                "description": "Исправленные правила, временные признаки, явный неполный анализ"}
    changed=None
    with store.db() as c:
        inserted=c.execute("INSERT OR IGNORE INTO versions(id,created,status,manifest) VALUES(?,?,?,?)",
                  (BASE_ID, time.time(), "released", json.dumps(manifest, ensure_ascii=False)))
        created=inserted.rowcount>0
        c.execute("INSERT OR IGNORE INTO settings VALUES('active_version',?)", (BASE_ID,))
        previous = c.execute("SELECT v.id,v.manifest FROM versions v JOIN settings s ON s.value=v.id WHERE s.key='active_version'").fetchone()
        if previous and json.loads(previous['manifest']).get('engine_fingerprint') != manifest['engine_fingerprint']:
            changed=previous['id']
            c.execute("UPDATE settings SET value=? WHERE key='active_version'", (BASE_ID,))
            c.execute("INSERT INTO releases(created,actor,previous,version,reason) VALUES(?,?,?,?,?)", (time.time(),'local-code-update',previous['id'],BASE_ID,'Изменился код извлечения признаков; старый классификатор несовместим'))
    if created:
        emit('versions','base_registered',version=BASE_ID,engine_fingerprint=manifest['engine_fingerprint'])
    if changed:
        emit('versions','code_updated',previous=changed,version=BASE_ID)


def active():
    row = store.one("SELECT value FROM settings WHERE key='active_version'")
    return get(row["value"] if row else BASE_ID)


def get(vid):
    return store.unpack(store.one("SELECT * FROM versions WHERE id=?", (vid,)))


def release(vid, actor, reason, rollback=False):
    with store.db() as c:
        c.execute('BEGIN IMMEDIATE')
        v = c.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        if not v or v["status"] not in ("candidate", "released"):
            raise ValueError("Версия не готова к выпуску")
        report = json.loads(v["report"] or "{}")
        if json.loads(v['manifest']).get('engine_fingerprint') != engine_fingerprint():
            raise ValueError('Версия относится к другому коду извлечения признаков. Нужна соответствующая копия приложения')
        if v["status"] == "candidate" and not report.get("release_eligible"):
            raise ValueError("Кандидат не прошёл контроль качества: " + "; ".join(report.get("gate_reasons", [])))
        if v['status']=='candidate':
            current=c.execute("SELECT value FROM settings WHERE key='active_version'").fetchone()[0]
            if v['parent']!=current:
                raise ValueError('Действующая версия изменилась после обучения. Нужно повторить сравнение с ней')
            for jid in report.get('feedback_jobs',[]):
                consent=c.execute("SELECT j.training_allowed,r.status FROM jobs j JOIN reviews r ON r.job_id=j.id WHERE j.id=?",(jid,)).fetchone()
                if not consent or not consent['training_allowed'] or consent['status']!='approved':
                    raise ValueError('После обучения изменилось разрешение или проверка разметки. Создайте нового кандидата')
        old = c.execute("SELECT value FROM settings WHERE key='active_version'").fetchone()[0]
        c.execute("UPDATE settings SET value=? WHERE key='active_version'", (vid,))
        c.execute("UPDATE versions SET status='released' WHERE id=?", (vid,))
        c.execute("INSERT INTO releases(created,actor,previous,version,reason) VALUES(?,?,?,?,?)",
                  (time.time(), actor, old, vid, reason))
    emit("versions", "rollback" if rollback else "released", previous=old, version=vid, actor=actor, reason=reason)
    return get(vid)
