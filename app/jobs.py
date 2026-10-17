"""In-memory job store + single GPU worker with per-job deadlines.

Guarantees: every job ends in `done` or `failed`; the uploaded file is deleted as soon as
processing ends; the pipeline returns its best partial result at the deadline, and a watchdog
fails any job that is still running shortly after it (a hung decode or CUDA call).

Jury (API-key) jobs run before reviewer-UI jobs: the UI is unauthenticated, so anyone with the
public URL could otherwise push the jury's jobs past their deadline.
"""
import itertools
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .pipeline.video import VideoError

log = logging.getLogger("forte.jobs")

WATCHDOG_GRACE_SEC = 20.0      # a job still running this long after its deadline is failed
# Decode + detection + scene grounding take ~15-25 s before any check can run: a job with less time left
# would only return an empty "partial" result (or overrun into the watchdog), so it fails honestly instead.
MIN_START_SEC = 30.0
PRIORITY = {"api": 0, "ui": 1}


@dataclass
class Job:
    id: str
    path: Path
    filename: str
    via: str = "api"                      # api | ui
    created: float = field(default_factory=time.time)   # start of the upload request
    status: str = "processing"           # processing | done | failed
    stage: str = "queued"                # human-readable progress stage
    progress: float = 0.0                # 0..1
    result: dict | None = None
    error: str | None = None
    finished: float | None = None
    discarded: bool = False              # removed by the reviewer while running: drop it once it ends

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


def _public_error(e: Exception) -> str:
    """Short message for the client; details (paths, CUDA internals) stay in the log."""
    if isinstance(e, (VideoError, TimeoutError)):
        return str(e)
    return f"processing failed ({type(e).__name__})"


class JobManager:
    def __init__(self, analyze_fn):
        self._analyze = analyze_fn       # (path, job) -> result dict
        self._jobs: dict[str, Job] = {}
        self._q: queue.PriorityQueue[tuple[int, int, str]] = queue.PriorityQueue()
        self._seq = itertools.count()    # FIFO within a priority
        self._lock = threading.Lock()
        self._running: Job | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="gpu-worker")
        self._thread.start()
        threading.Thread(target=self._janitor, daemon=True, name="janitor").start()

    def submit(self, path: Path, filename: str, via: str = "api", created: float | None = None) -> Job:
        # full 128-bit id: /ui/jobs is unauthenticated, so the id is all that guards a reviewer's result
        job = Job(id=uuid.uuid4().hex, path=path, filename=filename, via=via,
                  created=created or time.time())
        with self._lock:
            self._jobs[job.id] = job
        self._q.put((PRIORITY[via], next(self._seq), job.id))
        log.info("job %s queued via %s (%r), queue=%d", job.id, via, filename[:80], self._q.qsize())
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def discard(self, job_id: str, via: str) -> bool:
        """Forget a `via` job and its result now. A queued job never runs; the running one is dropped
        when it ends (the GPU call cannot be interrupted). False if there is no such job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.via != via:
                return False
            if job is self._running and job.status == "processing":
                job.discarded = True
                log.info("job %s discarded while running: dropped when it ends", job_id)
                return True
            del self._jobs[job_id]
        if job.status == "processing":         # still queued: _run skips a job it cannot find
            job.error, job.status, job.finished = "cancelled", "failed", time.time()
            try:
                job.path.unlink(missing_ok=True)
            except OSError:                    # the worker picked it up just now: _run deletes it
                log.warning("could not delete %s", job.path)
        job.result = None
        log.info("job %s discarded", job_id)
        return True

    def queue_position(self, job_id: str) -> int:
        with self._q.mutex:              # the worker mutates the heap concurrently
            pending = [jid for _, _, jid in sorted(self._q.queue)]
        return pending.index(job_id) + 1 if job_id in pending else 0

    def pending(self, via: str) -> int:
        """Unfinished jobs (queued or running) submitted through `via`."""
        with self._lock:
            return sum(j.via == via and j.status == "processing" for j in self._jobs.values())

    def worker_alive(self) -> bool:
        return self._thread.is_alive()

    def worker_stuck(self) -> bool:
        job = self._running
        return job is not None and time.time() > job.deadline + WATCHDOG_GRACE_SEC

    def _worker(self):
        while True:
            _, _, jid = self._q.get()
            try:
                self._run(jid)
            except BaseException:  # the worker must never die: every later job would stay 'processing'
                log.critical("worker error on job %s\n%s", jid, traceback.format_exc())
                job = self.get(jid)
                if job is not None and job.status == "processing":
                    job.status, job.error = "failed", "internal worker error"
            finally:
                self._running = None

    def _run(self, jid: str):
        job = self.get(jid)
        if job is None:
            return
        if job.status != "processing":         # already failed by the watchdog while queued
            job.path.unlink(missing_ok=True)
            return
        self._running = job
        t0 = time.time()
        try:
            if time.time() > job.deadline - MIN_START_SEC:
                raise TimeoutError("job waited in queue past its deadline")
            job.stage = "analyzing"
            result = self._analyze(job.path, job)
            if result is None:
                raise RuntimeError("pipeline returned no result")
            if job.status == "processing":     # the watchdog may have answered already
                job.result, job.status = result, "done"
        except Exception as e:  # never leave a job hanging
            log.error("job %s failed: %s\n%s", jid, e, traceback.format_exc())
            if job.status == "processing":
                job.error, job.status = _public_error(e), "failed"
        finally:
            job.finished = job.finished or time.time()
            job.progress = 1.0
            try:
                job.path.unlink(missing_ok=True)   # never keep uploaded videos
            except OSError:
                log.warning("could not delete %s", job.path)
            log.info("job %s %s in %.1fs (total since request %.1fs)", jid, job.status,
                     time.time() - t0, time.time() - job.created)
            if job.discarded:
                with self._lock:
                    self._jobs.pop(jid, None)
                job.result = None

    def _watchdog(self, now: float):
        """The deadline is cooperative; a hung decode/CUDA call would otherwise leave the running job
        and everything queued behind it 'processing' forever."""
        with self._lock:
            late = [j for j in self._jobs.values()
                    if j.status == "processing" and now > j.deadline + WATCHDOG_GRACE_SEC]
        for j in late:
            log.critical("job %s still processing %.0f s after its deadline: failing it", j.id, now - j.deadline)
            j.error, j.status, j.finished = "processing did not finish in time", "failed", now
            if j is not self._running:
                j.path.unlink(missing_ok=True)

    def _janitor(self):
        last_ttl = 0.0
        while True:
            time.sleep(2)
            now = time.time()
            try:
                self._watchdog(now)
                if now - last_ttl > 300:
                    last_ttl = now
                    with self._lock:
                        old = [k for k, j in self._jobs.items()
                               if j.finished and now - j.finished > config.RESULT_TTL_SEC]
                        for k in old:
                            del self._jobs[k]
            except Exception:
                log.exception("janitor error")
