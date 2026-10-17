"""Model-free stand-in used with FORTE_DUMMY=1 (API/UI smoke tests). Never used for real results."""
import time

from . import video
from .rules import build_violation


def analyze(path, job):
    clip = video.sample(path, target_fps=1.0)
    job.progress = 0.5
    time.sleep(1.0)
    i = len(clip.frames) // 2
    t = clip.times[i]
    frame = {"timestamp_sec": round(t, 1), "image_base64": video.to_b64_jpeg(clip.frames[i])}
    v = build_violation("T01_UNATTENDED_VALUABLES", t, "ТЕСТОВЫЙ РЕЖИМ: на кадре показан пример вывода",
                        [frame], 0.0)
    return {"verdict": "violation", "violations": [v], "dummy": True,
            "video_duration_sec": round(clip.duration, 1)}
