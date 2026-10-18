"""ChipTRick: authenticated local review application and jury API."""
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from . import auth, config, store, versions
from .jobs import JobManager, make_preview
from .telemetry import emit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
jobs = None
started = time.time()

@asynccontextmanager
async def lifespan(app):
    global jobs
    versions.initialize()
    if os.environ.get("CHIPTRICK_TEST_MODE") != "1":
        jobs = JobManager()
        from .training import start_scheduler
        start_scheduler()
    yield

app = FastAPI(title="ChipTRick", version="2.0", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.middleware("http")
async def same_origin(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Недопустимый источник запроса"}, 403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith(("/ui/", "/auth/")):
        response.headers["Cache-Control"] = "no-store"
    return response

class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=1, max_length=200)

class Event(BaseModel):
    type_id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    @model_validator(mode="after")
    def valid(self):
        from .pipeline.rules import TYPE_NAMES
        if self.type_id not in TYPE_NAMES or self.end < self.start:
            raise ValueError("Неверный тип или интервал")
        return self

class Correction(BaseModel):
    verdict: str
    events: list[Event] = Field(default_factory=list, max_length=30)
    note: str = Field(default="", max_length=2000)
    @model_validator(mode="after")
    def valid(self):
        if self.verdict not in ("violation", "no_violation"):
            raise ValueError("Укажите итоговый вердикт")
        if (self.verdict == "violation") != bool(self.events):
            raise ValueError("Для нарушения нужны тип и интервал; для отсутствия нарушений список должен быть пустым")
        return self

class Review(BaseModel):
    status: str
    note: str = Field(default="", max_length=2000)

class Release(BaseModel):
    reason: str = Field(min_length=3, max_length=1000)

class TrainingChoice(BaseModel):
    allowed: bool

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

@app.get("/health")
def health():
    return {"service":"ChipTRick","status": "ready" if jobs and jobs.ready else "loading" if jobs and not jobs.load_error else "unavailable", "model_version": versions.active()["id"], "uptime_sec": round(time.time()-started), "queue": jobs._q.qsize() if jobs else 0, "error": jobs.load_error if jobs else None}

@app.post("/auth/register")
def register(body: Credentials):
    if body.username.strip().lower() in ('jury-api','admin'):
        raise HTTPException(400,'Этот логин зарезервирован системой')
    try:
        return auth.create_user(body.username, body.password)
    except (ValueError, sqlite3.IntegrityError) as e:
        raise HTTPException(400, "Логин занят" if isinstance(e, sqlite3.IntegrityError) else str(e))

_attempts = {}

@app.post("/auth/login")
def login(body: Credentials, request: Request):
    ip = request.client.host if request.client else "local"
    now = time.monotonic()
    attempts = [t for t in _attempts.get(ip, []) if now-t < 60]
    if len(attempts) >= 15:
        raise HTTPException(429, "Слишком много попыток. Подождите минуту")
    _attempts[ip] = attempts + [now]
    token = auth.login(body.username, body.password)
    r = JSONResponse({"ok": True})
    r.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax", secure=request.url.scheme == "https", max_age=86400)
    return r

@app.post("/auth/logout")
def logout(request: Request):
    store.execute("DELETE FROM sessions WHERE token=?", (hashlib.sha256(request.cookies.get(auth.COOKIE, "").encode()).hexdigest(),))
    r = JSONResponse({"ok": True})
    r.delete_cookie(auth.COOKIE)
    return r

@app.get("/auth/me")
def me(user=Depends(auth.current_user)):
    return user

def owned(jid, user):
    row = store.unpack(store.one("SELECT * FROM jobs WHERE id=? AND owner=?", (jid, user["id"])))
    if not row:
        raise HTTPException(404, "Видео не найдено")
    return row

def public_job(row):
    for key in ("path", "owner", "sha256"):
        row.pop(key, None)
    row["feedback"] = store.unpack(store.one("SELECT * FROM feedback WHERE job_id=?", (row["id"],)))
    if row.get("result"):
        row["result"] = {k: v for k, v in row["result"].items() if not k.startswith("_") and k != "debug"}
    return row

@app.get("/ui/jobs")
def history(user=Depends(auth.current_user)):
    return [public_job(store.unpack(r)) for r in store.many("SELECT id,filename,created,finished,status,stage,progress,version,error,training_allowed,json_extract(result,'$.verdict') AS verdict FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 200", (user["id"],))]

async def upload(file, user):
    if not jobs or jobs.load_error:
        raise HTTPException(503, "Модели ещё не готовы: проверьте /health и журнал inference")
    name = Path(file.filename or "video").name
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(415, "Загрузите MOV, MP4, M4V, AVI или MKV")
    folder = config.WORK_DIR / "media"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / (uuid.uuid4().hex + ext)
    size = 0
    h = hashlib.sha256()
    try:
        with dst.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_MB * 2**20:
                    raise HTTPException(413, "Видео слишком большое")
                f.write(chunk)
                h.update(chunk)
        if not size:
            raise HTTPException(400, "Файл пуст")
        return {"job_id": jobs.submit(dst, name, user["id"], h.hexdigest())}
    except BaseException:
        dst.unlink(missing_ok=True)
        raise

@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...), user=Depends(auth.current_user)):
    return await upload(file, user)

@app.get("/ui/jobs/{jid}")
def detail(jid: str, user=Depends(auth.current_user)):
    return public_job(owned(jid, user))

def set_training_choice(row, body, user):
    store.execute('UPDATE jobs SET training_allowed=? WHERE id=?',(int(body.allowed),row['id']))
    emit('versions','training_choice_changed',job_id=row['id'],allowed=body.allowed,actor=user['username'])
    return {'allowed':body.allowed}

@app.post('/ui/jobs/{jid}/training')
def training_choice(jid: str, body: TrainingChoice, user=Depends(auth.current_user)):
    return set_training_choice(owned(jid,user),body,user)

@app.get("/ui/jobs/{jid}/video")
def video(jid: str, user=Depends(auth.current_user)):
    path = Path(owned(jid, user)["path"])
    return FileResponse(make_preview(path),media_type='video/mp4')

@app.post("/ui/jobs/{jid}/feedback")
def feedback(jid: str, body: Correction, user=Depends(auth.current_user)):
    row = owned(jid, user)
    return save_feedback(row, body, user)

def save_feedback(row, body, user):
    jid = row['id']
    if row["status"] != "done":
        raise HTTPException(409, "Дождитесь завершения анализа")
    if any(e.end > row["result"]["video_duration_sec"] + 0.1 for e in body.events):
        raise HTTPException(400, "Интервал выходит за длительность видео")
    try:
        store.execute("INSERT INTO feedback VALUES(?,?,?,?,?,?)", (jid, user["id"], time.time(), body.verdict, json.dumps([e.model_dump() for e in body.events]), body.note))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Вердикт уже исправлен. Повторное исправление невозможно")
    emit("versions", "feedback_submitted", job_id=jid, verdict=body.verdict)
    return {"ok": True}

def api_owned_by_system(jid):
    row=store.unpack(store.one("SELECT j.* FROM jobs j JOIN users u ON u.id=j.owner WHERE j.id=? AND u.username='jury-api'",(jid,)))
    if not row: raise HTTPException(404, 'Загрузка API не найдена')
    return row

@app.get('/ui/admin/api-jobs')
def api_history(user=Depends(auth.admin)):
    return [public_job(store.unpack(r)) for r in store.many("SELECT j.id,j.filename,j.created,j.finished,j.status,j.stage,j.progress,j.version,j.error,j.training_allowed,json_extract(j.result,'$.verdict') AS verdict FROM jobs j JOIN users u ON u.id=j.owner WHERE u.username='jury-api' ORDER BY j.created DESC LIMIT 200")]

@app.get('/ui/admin/api-jobs/{jid}')
def api_detail(jid: str, user=Depends(auth.admin)):
    return public_job(api_owned_by_system(jid))

@app.post('/ui/admin/api-jobs/{jid}/training')
def api_training_choice(jid: str, body: TrainingChoice, user=Depends(auth.admin)):
    return set_training_choice(api_owned_by_system(jid),body,user)

@app.get('/ui/admin/api-jobs/{jid}/video')
def api_video(jid: str, user=Depends(auth.admin)):
    p=Path(api_owned_by_system(jid)['path'])
    return FileResponse(make_preview(p),media_type='video/mp4')

@app.post('/ui/admin/api-jobs/{jid}/feedback')
def api_feedback(jid: str, body: Correction, user=Depends(auth.admin)):
    return save_feedback(api_owned_by_system(jid),body,user)

@app.get("/ui/types")
def types(user=Depends(auth.current_user)):
    from .pipeline.rules import TYPE_NAMES
    return TYPE_NAMES

@app.get("/ui/admin/feedback")
def pending(user=Depends(auth.admin)):
    return [store.unpack(r) for r in store.many("SELECT f.*,j.filename,j.training_allowed,r.status AS review_status,r.note AS review_note FROM feedback f JOIN jobs j ON j.id=f.job_id LEFT JOIN reviews r ON r.job_id=f.job_id ORDER BY f.created DESC")]

@app.get("/ui/admin/feedback/{jid}/video")
def admin_video(jid: str, user=Depends(auth.admin)):
    row = store.one("SELECT j.path FROM jobs j JOIN feedback f ON j.id=f.job_id WHERE j.id=?", (jid,))
    if not row:
        raise HTTPException(404)
    p = Path(row["path"])
    return FileResponse(make_preview(p),media_type='video/mp4')

@app.post("/ui/admin/feedback/{jid}/review")
def review(jid: str, body: Review, user=Depends(auth.admin)):
    if body.status not in ("approved", "rejected") or not store.one("SELECT job_id FROM feedback WHERE job_id=?", (jid,)):
        raise HTTPException(400, "Неверный статус или пример")
    store.execute("INSERT INTO reviews VALUES(?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET reviewer=excluded.reviewer,status=excluded.status,note=excluded.note,created=excluded.created", (jid, user["id"], body.status, body.note, time.time()))
    emit("versions", "feedback_reviewed", job_id=jid, status=body.status, reviewer=user["username"])
    return {"ok": True}

@app.get("/ui/admin/versions")
def model_versions(user=Depends(auth.admin)):
    return {"active": versions.active()["id"], "versions": [store.unpack(r) for r in store.many("SELECT * FROM versions ORDER BY created DESC")], "training": store.many("SELECT * FROM training_runs ORDER BY created DESC LIMIT 10")}

@app.get("/ui/admin/dataset/{video_id}/video")
def dataset_video(video_id: str, user=Depends(auth.admin)):
    import csv
    from .jobs import make_preview
    labels=list(csv.DictReader((config.ROOT/'data'/'labels_enriched.csv').open(encoding='utf-8-sig')))
    row=next((r for r in labels if r['video_id']==video_id),None)
    if row:
        original=config.ROOT/'videos'/row['file']
        # Transcoded previews live in runtime, never beside original dataset files.
        folder=config.WORK_DIR/'dataset-previews'
        folder.mkdir(parents=True,exist_ok=True)
        return FileResponse(make_preview(original,folder/(video_id+'.preview.mp4')),media_type='video/mp4')
    j=store.one("SELECT path FROM jobs WHERE id=?",(video_id,))
    if not j: raise HTTPException(404)
    return FileResponse(make_preview(Path(j['path'])),media_type='video/mp4')

@app.post("/ui/admin/train")
def train(user=Depends(auth.admin)):
    from .training import launch
    try:
        return {"run_id": launch()}
    except ValueError as e:
        raise HTTPException(409, str(e))

@app.post("/ui/admin/versions/{vid}/release")
def release(vid: str, body: Release, user=Depends(auth.admin)):
    try:
        return versions.release(vid, user["username"], body.reason)
    except ValueError as e:
        raise HTTPException(409, str(e))

@app.get("/ui/admin/logs/{stream}")
def logs(stream: str, user=Depends(auth.admin)):
    if stream not in ("all", "inference", "training", "versions"):
        raise HTTPException(404)
    path = config.WORK_DIR / "logs" / f"{stream}.jsonl"
    if not path.exists():
        return []
    from collections import deque
    with path.open(encoding="utf-8") as f:
        rows=[]
        for line in deque(f,maxlen=150):
            if not line.endswith('\n'):continue
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue
        return rows

def api_user(request: Request):
    if not config.API_KEY or not secrets.compare_digest(request.headers.get("x-api-key", ""), config.API_KEY):
        raise HTTPException(401, "Неверный X-API-Key")
    u = store.one("SELECT id,username,role FROM users WHERE username='jury-api'")
    if not u:
        try:
            u = auth.create_user("jury-api", secrets.token_urlsafe(32))
        except sqlite3.IntegrityError:
            u = store.one("SELECT id,username,role FROM users WHERE username='jury-api'")
    return u

@app.post("/api/v1/analyze")
async def api_analyze(file: UploadFile = File(...), user=Depends(api_user)):
    return await upload(file, user)

@app.get("/api/v1/jobs/{jid}")
def api_job(jid: str, user=Depends(api_user)):
    row = owned(jid, user)
    result={k: v for k, v in (row["result"] or {}).items() if not k.startswith("_")}
    status='processing' if row['status']=='queued' else row['status']
    if result.get('verdict')=='needs_review':
        # The jury contract has no third final verdict. Do not disguise uncertainty as a clean clip.
        return {'job_id':jid,'status':'failed','error':'insufficient_evidence','review_reasons':result.get('review_reasons',[]),'model_version':row['version']}
    return {"job_id": jid, "status": status, "error": row["error"], **result}
