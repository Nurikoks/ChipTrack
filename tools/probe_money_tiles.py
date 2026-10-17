"""Probe 3: banknotes on desk with tiled desk crops (each tile upscaled), both prompts, latency.

Usage: python probe_money_tiles.py <model_id>
"""
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.pipeline.models import VLM, crop  # noqa: E402

# (file, t, desk box from grounding, expected)
CASES = [("Х-26.mov", 11.5, (87, 370, 345, 574), True), ("Х-4.mov", 55.0, (24, 396, 299, 546), True),
         ("Х-38.mov", 30.0, (64, 362, 324, 542), True), ("Х-46.mov", 8.0, (40, 390, 305, 565), True),
         ("Х-14.mov", 32.6, (48, 394, 315, 546), True),
         ("Х-25.mov", 31.0, (91, 367, 345, 525), False), ("Х-27.mov", 25.9, (91, 367, 345, 525), False),
         ("Х-19.mov", 47.0, (46, 356, 303, 497), False), ("Х-5.mov", 62.0, (23, 404, 296, 545), False),
         ("Х-24.mov", 12.0, (94, 377, 343, 530), False), ("Х-40.mov", 22.0, (60, 380, 330, 560), False)]
QS = ["Are there banknotes (paper money) lying on this part of the table?",
      "Is there a stack of money or a pile of banknotes here?"]

vlm = VLM(sys.argv[1], four_bit=True)
lat = []
for f, t, (x1, y1, x2, y2), exp in CASES:
    cap = cv2.VideoCapture(str(ROOT / "videos" / f))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    w = x2 - x1
    tiles = [(x1, y1, x1 + 0.6 * w, y2), (x1 + 0.4 * w, y1, x2, y2)]   # two overlapping halves
    best_full = max(vlm.p_yes(crop(fr, (x1, y1, x2, y2), 0.1), q) for q in QS)
    ps = []
    for tb in tiles:
        for q in QS:
            s = time.time()
            ps.append(vlm.p_yes(crop(fr, tb, 0.1, min_side=512), q))
            torch.cuda.synchronize()
            lat.append(time.time() - s)
    print(f"{f[2:-4]:5s}@{t:5.1f} exp={str(exp):5s} full={best_full:.3f} tiles_max={max(ps):.3f}")
print(f"median latency per question: {sorted(lat)[len(lat) // 2]:.2f}s")
