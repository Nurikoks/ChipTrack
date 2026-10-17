"""Pseudo-new-angle validation set: geometric + photometric warps of training clips.

Each clip gets one deterministic transform (chosen by its id) from:
  persp_l / persp_r  - perspective tilt (simulates camera moved sideways)
  rot                - rotation +-8 deg with zoom-in (camera roll)
  flip               - horizontal mirror (room layout mirrored)
  crop_zoom          - 80% crop off-centre, rescaled (narrower field of view)
  landscape          - re-framed to 4:3 landscape with padding/crop (different aspect)
plus brightness/contrast jitter. Output is .mp4 (H.264 via OpenCV mp4v fallback).

Usage: python make_pseudo_angles.py <videos_dir> <out_dir> [video_id ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRANSFORMS = ["persp_l", "persp_r", "rot", "flip", "crop_zoom", "landscape"]


def make_fn(kind, w, h, seed):
    rng = np.random.default_rng(seed)
    alpha, beta = rng.uniform(0.8, 1.2), rng.uniform(-25, 25)
    if kind in ("persp_l", "persp_r"):
        d = 0.12 * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[d, 0.06 * h], [w, 0], [w, h], [d, 0.94 * h]]) if kind == "persp_l" else \
            np.float32([[0, 0], [w - d, 0.06 * h], [w - d, 0.94 * h], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        geo = lambda f: cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        size = (w, h)
    elif kind == "rot":
        ang = rng.choice([-8, 8])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.15)
        geo = lambda f: cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        size = (w, h)
    elif kind == "flip":
        geo = lambda f: cv2.flip(f, 1)
        size = (w, h)
    elif kind == "crop_zoom":
        cw, ch = int(0.8 * w), int(0.8 * h)
        x0, y0 = int(rng.uniform(0, w - cw)), int(rng.uniform(0, 0.1 * h))
        geo = lambda f: cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h))
        size = (w, h)
    else:  # landscape 4:3 from the upper part of the portrait frame (where the scene is)
        ch = int(w * 3 / 4)
        y0 = int(0.12 * h)
        ow, oh = 640, 480
        geo = lambda f: cv2.resize(f[y0:y0 + ch, :], (ow, oh))
        size = (ow, oh)
    return (lambda f: cv2.convertScaleAbs(geo(f), alpha=alpha, beta=beta)), size


def main():
    vdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / "data" / "labels_enriched.csv")
    lab = lab[lab["duplicate_of"].isna() | ~lab["video_id"].isin(["X-47", "X-48"])]
    only = set(sys.argv[3:])
    rows = []
    for k, r in enumerate(lab.itertuples()):
        if only and r.video_id not in only:
            continue
        kind = TRANSFORMS[k % len(TRANSFORMS)]
        cap = cv2.VideoCapture(str(vdir / r.file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fn, size = make_fn(kind, w, h, seed=k)
        dst = out / f"{r.video_id}__{kind}.mp4"
        wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            wr.write(fn(f))
        wr.release()
        rows.append({"video_id": r.video_id, "file": dst.name, "transform": kind, "label": r.label,
                     "timestamp_sec": r.timestamp_sec, "type_candidate": r.type_candidate})
        print(dst.name)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)


if __name__ == "__main__":
    main()
