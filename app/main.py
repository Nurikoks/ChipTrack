"""FastAPI service: jury API (/api/v1, X-API-Key) + reviewer web UI (/, /ui/*)."""
import logging
import secrets
import shutil
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .jobs import JobManager
from .pipeline import rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("forte.api")

if not config.API_KEY:
    raise SystemExit("FORTE_API_KEY environment variable is not set")

config.WORK_DIR.mkdir(parents=True, exist_ok=True)
for stale in config.WORK_DIR.glob("upload_*"):  # leftovers from a crash: never keep videos
    stale.unlink(missing_ok=True)

rules.validate()                      # a broken rules registry stops startup instead of the first violating job

if config.DUMMY:
    from .pipeline.dummy import analyze
else:
    from .pipeline.analyze import Analyzer
    _analyzer = Analyzer()            # loads all models once, before serving
    _analyzer.warmup()
    analyze = _analyzer.analyze

jobs = JobManager(analyze)
app = FastAPI(title="ForteBank cash-desk video review", version="0.1")
STATIC = Path(__file__).parent / "static"
STARTED = time.time()


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """Reject /api/* without a valid key. The request body is drained first: some reverse proxies
    (ngrok with Expect: 100-continue) otherwise rewrite an early 401 into 200."""
    if request.url.path.startswith("/api/"):
        key = (request.headers.get("x-api-key") or "").encode("utf-8", "ignore")
        if not secrets.compare_digest(key, config.API_KEY.encode()):
            async for _ in request.stream():
                pass
            return JSONResponse(status_code=401, content={"detail": "invalid or missing X-API-Key"})
    return await call_next(request)


def require_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _store_upload(file: UploadFile) -> Path:
    name = file.filename or "video"
    if Path(name).suffix.lower() not in config.ALLOWED_EXT:
        raise HTTPException(415, f"unsupported file type; expected one of {sorted(config.ALLOWED_EXT)}")
    dst = config.WORK_DIR / f"upload_{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    limit = config.MAX_UPLOAD_MB * 2**20
    size = 0
    with dst.open("wb") as out:
        while chunk := await file.read(2**20):
            size += len(chunk)
            if size > limit:
                out.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "empty file")
    return dst


@app.get("/health")
def health():
    alive = jobs.worker_alive()
    return {"status": "ok" if alive else "degraded", "worker_alive": alive,
            "uptime_sec": round(time.time() - STARTED), "dummy": config.DUMMY, "queue": jobs._q.qsize()}


@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def api_analyze(file: UploadFile = File(...)):
    path = await _store_upload(file)
    return {"job_id": jobs.submit(path, file.filename or "").id}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.public()


# ---- Reviewer UI (no auth by requirement; same pipeline, same queue) ----
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    path = await _store_upload(file)
    return {"job_id": jobs.submit(path, file.filename or "").id}


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    out = job.public()
    if job.status == "processing":
        out["queue_position"] = jobs.queue_position(job_id)
        out["elapsed_sec"] = round(time.time() - job.created)
    return out


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
