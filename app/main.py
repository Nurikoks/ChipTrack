"""FastAPI service: jury API (/api/v1, X-API-Key) + reviewer web UI (/, /ui/*)."""
import logging
import secrets
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .jobs import JobManager
from .pipeline import rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("forte.api")

if not config.API_KEY:
    raise SystemExit("FORTE_API_KEY environment variable is not set")
if not config.API_KEY.isascii():
    raise SystemExit("FORTE_API_KEY must be ASCII (HTTP headers cannot carry it otherwise)")

config.WORK_DIR.mkdir(parents=True, exist_ok=True)
for stale in config.WORK_DIR.glob("upload_*"):  # leftovers from a crash: never keep videos
    stale.unlink(missing_ok=True)
# Starlette spools multipart uploads to a temp file before the handler runs: keep that copy next to our
# own uploads (same disk, cleaned on start) instead of the system TEMP.
TMP_DIR = config.WORK_DIR / "tmp"
shutil.rmtree(TMP_DIR, ignore_errors=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TMP_DIR)

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
UPLOAD_PATHS = ("/api/v1/analyze", "/ui/analyze")
UI_MAX_PENDING = 3                    # unauthenticated uploads waiting or running at once

# container signatures: the extension alone lets any file reach FFmpeg's demuxers
_ISO_BMFF = (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")   # .mov/.mp4/.m4v


def _looks_like_video(head: bytes, ext: str) -> bool:
    if ext in (".mov", ".mp4", ".m4v"):
        return head[4:8] in _ISO_BMFF
    if ext == ".avi":
        return head[:4] == b"RIFF" and head[8:12] == b"AVI "
    if ext == ".mkv":
        return head[:4] == b"\x1a\x45\xdf\xa3"
    return False


async def _reject(request: Request, status: int, detail: str) -> JSONResponse:
    """The request body is drained first: some reverse proxies (ngrok with Expect: 100-continue)
    otherwise rewrite an early error response into 200."""
    async for _ in request.stream():
        pass
    return JSONResponse(status_code=status, content={"detail": detail})


@app.middleware("http")
async def guard(request: Request, call_next):
    """API key for /api/*, upload size and UI queue limits — all checked before the body is received.
    Also stamps the request start: the job deadline counts from here, upload time included."""
    request.state.t0 = time.time()
    path = request.url.path
    if path.startswith("/api/"):
        key = (request.headers.get("x-api-key") or "").encode("utf-8", "ignore")
        if not secrets.compare_digest(key, config.API_KEY.encode()):
            return await _reject(request, 401, "invalid or missing X-API-Key")
    if request.method == "POST" and path in UPLOAD_PATHS:
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > config.MAX_UPLOAD_MB * 2**20 + 2**20:   # + multipart overhead
            return await _reject(request, 413, f"file larger than {config.MAX_UPLOAD_MB} MB")
        if path == "/ui/analyze" and jobs.pending("ui") >= UI_MAX_PENDING:
            return await _reject(request, 429, "Очередь проверки занята, повторите через минуту.")
    return await call_next(request)


def require_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key.encode("utf-8", "ignore"), config.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _store_upload(file: UploadFile) -> Path:
    name = file.filename or "video"
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(415, f"unsupported file type; expected one of {sorted(config.ALLOWED_EXT)}")
    dst = config.WORK_DIR / f"upload_{uuid.uuid4().hex}{ext}"
    limit = config.MAX_UPLOAD_MB * 2**20
    size = 0
    try:
        with dst.open("wb") as out:
            while chunk := await file.read(2**20):
                if size == 0 and not _looks_like_video(chunk[:12], ext):
                    raise HTTPException(415, "file content is not a supported video container")
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_MB} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty file")
    except BaseException:   # rejected, client gone, disk full: never leave a partial video behind
        dst.unlink(missing_ok=True)
        raise
    return dst


async def _accept(request: Request, file: UploadFile, via: str) -> dict:
    path = await _store_upload(file)
    try:
        job = jobs.submit(path, file.filename or "", via=via, created=request.state.t0)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"job_id": job.id}


@app.get("/health")
def health():
    alive, stuck = jobs.worker_alive(), jobs.worker_stuck()
    ok = alive and not stuck
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ok" if ok else "degraded", "worker_alive": alive,
                                 "worker_stuck": stuck, "uptime_sec": round(time.time() - STARTED),
                                 "dummy": config.DUMMY})


@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def api_analyze(request: Request, file: UploadFile = File(...)):
    return await _accept(request, file, "api")


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.public()


# ---- Reviewer UI (no auth by requirement; same pipeline, lower priority than the jury API) ----
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/ui/analyze")
async def ui_analyze(request: Request, file: UploadFile = File(...)):
    return await _accept(request, file, "ui")


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.via != "ui":   # jury results (evidence frames of people) need the key
        raise HTTPException(404, "job not found")
    out = job.public()
    if job.status == "processing":
        out["queue_position"] = jobs.queue_position(job_id)
        out["elapsed_sec"] = round(time.time() - job.created)
    return out


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    # a plain string, like every other error here (the UI shows `detail` as text)
    return JSONResponse(status_code=422, content={"detail": "expected multipart/form-data with a 'file' field"})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
