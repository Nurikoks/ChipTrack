"""Contact-sheet / frame extraction helper for visual data review.

  python sheets.py frame  <video> <t_sec> <out.jpg>
  python sheets.py window <video> <t0> <t1> <step> <cols> <thumb_w> <out.jpg>
  python sheets.py crop   <video> <t0> <t1> <step> <cols> <thumb_w> <x0> <y0> <x1> <y1> <out.jpg>
      (crop box in relative coords 0..1 of the full frame)
"""
import sys

import cv2
import numpy as np


def grab(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    return f if ok else None


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(video, t0, t1, step, cols, tw, out, box=None):
    cap = cv2.VideoCapture(video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    ts = np.arange(max(0, t0), min(t1, dur - 0.05) + 1e-6, step)
    tiles = []
    for t in ts:
        f = grab(cap, t)
        if f is None:
            continue
        if box:
            h, w = f.shape[:2]
            f = f[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
        th = int(f.shape[0] * tw / f.shape[1])
        tiles.append(label(cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA), f"{t:.1f}s"))
    if not tiles:
        raise SystemExit("no frames")
    th = tiles[0].shape[0]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, len(ts), "frames", f"dur={dur:.1f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "frame":
        cap = cv2.VideoCapture(sys.argv[2])
        cv2.imwrite(sys.argv[4], grab(cap, float(sys.argv[3])))
    elif mode == "window":
        v, t0, t1, st, c, tw, out = sys.argv[2:9]
        sheet(v, float(t0), float(t1), float(st), int(c), int(tw), out)
    elif mode == "crop":
        v, t0, t1, st, c, tw, x0, y0, x1, y1, out = sys.argv[2:13]
        sheet(v, float(t0), float(t1), float(st), int(c), int(tw), out,
              box=tuple(map(float, (x0, y0, x1, y1))))
