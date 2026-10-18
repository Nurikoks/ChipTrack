"""Durable jobs and one shared GPU worker. Queue waiting does not consume analysis budget."""
import json
import logging
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from . import config, store, versions
from .telemetry import emit


@dataclass
class Job:
    id: str
    started: float
    version: str
    stage: str = "analyzing"
    progress: float = 0.0

    @property
    def deadline(self):
        return self.started + config.JOB_DEADLINE_SEC


class JobManager:
    def __init__(self, analyze_fn=None):
        self._analyze = analyze_fn
        self._q = queue.Queue()
        self.ready = analyze_fn is not None
        self.load_error = None
        self.current = None
        store.execute("UPDATE jobs SET status='failed', error='Анализ прерван перезапуском сервера', finished=? WHERE status='processing'", (time.time(),))
        for r in store.many("SELECT id FROM jobs WHERE status='queued' ORDER BY created"):
            self._q.put(r["id"])
        threading.Thread(target=self._worker, daemon=True, name="gpu-worker").start()
        threading.Thread(target=self._progress, daemon=True).start()

    def submit(self, path, filename, owner, sha256):
        jid = uuid.uuid4().hex
        store.execute("INSERT INTO jobs(id,owner,filename,path,sha256,created,status,stage) VALUES(?,?,?,?,?,?,?,?)",
                      (jid, owner, filename, str(path), sha256, time.time(), "queued", "queued"))
        self._q.put(jid)
        emit("inference", "queued", job_id=jid, queue=self._q.qsize())
        return jid

    def _progress(self):
        while True:
            time.sleep(1)
            j = self.current
            if j:
                store.execute("UPDATE jobs SET stage=?,progress=? WHERE id=? AND status='processing'", (j.stage, float(j.progress), j.id))

    def _worker(self):
        if self._analyze is None:
            try:
                emit("inference", "model_loading", version=versions.active()["id"], detector=config.DETECTOR_ID, vlm=config.VLM_ID)
                if config.DUMMY:
                    raise RuntimeError("FORTE_DUMMY запрещён для рабочего сервера")
                from .pipeline.analyze import Analyzer
                self._analyze = Analyzer().analyze
                self.ready = True
                emit("inference", "model_ready", version=versions.active()["id"])
            except Exception as e:
                self.load_error = str(e)
                emit("inference", "model_load_failed", error=str(e))
                logging.exception("model load failed")
                return
        while True:
            jid = self._q.get()
            row = store.one("SELECT * FROM jobs WHERE id=?", (jid,))
            if not row:
                self._q.task_done()
                continue
            j = Job(jid, time.time(), versions.active()["id"])
            self.current = j
            store.execute("UPDATE jobs SET status='processing',started=?,version=? WHERE id=?", (j.started, j.version, jid))
            try:
                emit("inference", "analysis_started", job_id=jid, version=j.version, waited_sec=round(j.started-row["created"], 2))
                result = self._analyze(Path(row["path"]), j)
                result["model_version"] = j.version
                store.execute("UPDATE jobs SET status='done',stage='done',progress=1,result=?,finished=? WHERE id=?", (json.dumps(result, ensure_ascii=False), time.time(), jid))
                emit("inference", "analysis_finished", job_id=jid, version=j.version, verdict=result["verdict"], partial=result.get("partial"), seconds=round(time.time()-j.started, 2))
                try:
                    make_preview(Path(row["path"]))
                except Exception as e:
                    emit("inference", "preview_failed", job_id=jid, error=str(e))
            except Exception as e:
                logging.exception("analysis failed")
                store.execute("UPDATE jobs SET status='failed',error=?,finished=? WHERE id=?", (str(e), time.time(), jid))
                emit("inference", "analysis_failed", job_id=jid, version=j.version, error=str(e))
            finally:
                self.current = None
                self._q.task_done()


_preview_locks={}
_preview_guard=threading.Lock()


def make_preview(path, destination=None):
    import imageio_ffmpeg
    dst = destination or path.with_suffix(".preview.mp4")
    with _preview_guard:
        lock=_preview_locks.setdefault(str(dst),threading.Lock())
    with lock:
        if dst.exists():return dst
        dst.parent.mkdir(parents=True,exist_ok=True)
        temp = dst.with_suffix(".encoding.mp4")
        try:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-i", str(path), '-map','0:v:0','-map','0:a:0?', "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", '-c:a','aac','-b:a','128k', "-movflags", "+faststart", str(temp)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
            temp.replace(dst)
        finally:
            temp.unlink(missing_ok=True)
    return dst
