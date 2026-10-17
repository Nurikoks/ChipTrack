"""Video decoding / sampling / evidence-frame encoding."""
import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


class VideoError(ValueError):
    pass


MAX_PIXELS = 4096 * 2304          # beyond 4K a single decoded frame costs tens of MB
# grey 64x64 JPEG for a finding whose frames could not be encoded (the API requires 1-3 frames);
# built at import, so a broken JPEG encoder stops startup instead of the first report
PLACEHOLDER_B64 = base64.b64encode(cv2.imencode(".jpg", np.full((64, 64, 3), 128, np.uint8))[1].tobytes()).decode("ascii")


@dataclass
class Clip:
    times: list[float]            # timestamp (s) of each sampled frame
    frames: list[np.ndarray]      # BGR frames (native resolution)
    fps: float
    duration: float
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)   # decode problems (see sample)


def probe(path) -> tuple[float, int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("файл не удаётся открыть как видео (повреждён или неподдерживаемый формат)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or fps <= 0 or w == 0:
        raise VideoError("в файле нет декодируемых видеокадров")
    if w * h > MAX_PIXELS:
        raise VideoError(f"слишком большое разрешение видео ({w}×{h}); поддерживается до 4K")
    return fps, n, (n / fps if n > 0 else 0.0), w, h


def sample(path, target_fps: float = 2.0, max_frames: int = 1200, max_side: int = 960,
           deadline: float | None = None) -> Clip:
    """Sequential decode keeping every k-th frame. Large videos are downscaled to max_side.
    Stops at `deadline` (wall clock) so a pathological file cannot hold the single GPU worker."""
    fps, n, dur, w, h = probe(path)
    step = max(1, round(fps / target_fps))
    scale = min(1.0, max_side / max(w, h))
    cap = cv2.VideoCapture(str(path))
    times, frames, i, bad, late = [], [], 0, 0, False
    while len(frames) < max_frames:
        if deadline is not None and frames and i % 64 == 0 and time.time() > deadline:
            late = True
            break
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            bad += not ok
            if ok:
                if scale < 1.0:
                    f = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                times.append(ts if ts > 0 or i == 0 else i / fps)
                frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise VideoError("в файле нет декодируемых видеокадров")
    # A cut or damaged file decodes only its first part; without this the result would look complete.
    warnings = []
    if len(frames) >= max_frames:
        warnings.append("frames_capped")
    elif late or (n > 0 and i < 0.9 * n) or bad > 0.1 * (len(frames) + bad):
        warnings.append("decode_truncated")
    dur = max(dur, times[-1])
    return Clip(times, frames, fps, dur, frames[0].shape[1], frames[0].shape[0], warnings)


def to_b64_jpeg(frame: np.ndarray, quality: int = 85, box=None, label: str | None = None) -> str:
    img = frame.copy()
    if box is not None:  # highlight the region the finding refers to
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if label:
            cv2.putText(img, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def nearest_index(times: list[float], t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times) - t)))
