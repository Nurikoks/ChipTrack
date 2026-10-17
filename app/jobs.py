"""In-memory job store + single GPU worker with per-job deadlines.

Guarantees: every job ends in `done` or `failed`; the uploaded file is deleted as soon as
processing ends; a job never runs past its deadline (the pipeline returns its best partial
result instead).
"""
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config

log = logging.getLogger("forte.jobs")


@dataclass
class Job:
    id: str
    path: Path
    filename: str
    created: float = field(default_factory=time.time)
    status: str = "processing"           # processing | done | failed
    stage: str = "queued"                # human-readable progress stage
    progress: float = 0.0                # 0..1
    result: dict | None = None
    error: str | None = None
    finished: float | None = None

    @property
    def deadline(self) -> float:
        return self.created + config.JOB_DEADLINE_SEC

    def public(self) -> dict:
        out = {"job_id": self.id, "status": self.status}
        if self.status == "done" and self.result is not None:
            out.update(self.result)
        elif self.status == "failed":
            out["error"] = self.error or "processing failed"
        else:
            out["stage"] = self.stage
            out["progress"] = round(self.progress, 3)
        return out


class JobManager:
    def __init__(self, analyze_fn):
        self._analyze = analyze_fn       # (path, job) -> result dict
        self._jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="gpu-worker").start()
        threading.Thread(target=self._janitor, daemon=True, name="janitor").start()

    def submit(self, path: Path, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], path=path, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job.id)
        log.info("job %s queued (%s), queue=%d", job.id, filename, self._q.qsize())
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> int:
        return list(self._q.queue).index(job_id) + 1 if job_id in self._q.queue else 0

    def _worker(self):
        while True:
            jid = self._q.get()
            job = self.get(jid)
            if job is None:
                continue
            t0 = time.time()
            try:
                if time.time() > job.deadline - 5:
                    raise TimeoutError("job waited in queue past its deadline")
                job.stage = "analyzing"
                job.result = self._analyze(job.path, job)
                job.status = "done"
            except Exception as e:  # never leave a job hanging
                log.error("job %s failed: %s\n%s", jid, e, traceback.format_exc())
                job.error = f"{type(e).__name__}: {e}"
                job.status = "failed"
            finally:
                job.finished = time.time()
                job.progress = 1.0
                try:
                    job.path.unlink(missing_ok=True)   # never keep uploaded videos
                except OSError:
                    log.warning("could not delete %s", job.path)
                log.info("job %s %s in %.1fs (total since upload %.1fs)", jid, job.status,
                         job.finished - t0, job.finished - job.created)

    def _janitor(self):
        while True:
            time.sleep(300)
            now = time.time()
            with self._lock:
                old = [k for k, j in self._jobs.items() if j.finished and now - j.finished > config.RESULT_TTL_SEC]
                for k in old:
                    del self._jobs[k]
