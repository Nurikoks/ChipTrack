"""Three real event streams, plus a combined stream, shared by CLI and server."""
import json
import logging
import os
from datetime import datetime, timezone
from . import config


def emit(stream, event, **fields):
    record = {"time": datetime.now(timezone.utc).isoformat(), "stream": stream,
              "event": event, "pid": os.getpid(), **fields}
    folder = config.WORK_DIR / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    for name in (stream, "all"):
        fd = os.open(str(folder / f"{name}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    logging.getLogger(f"chiptrick.{stream}").info("%s %s", event, fields)
    return record
