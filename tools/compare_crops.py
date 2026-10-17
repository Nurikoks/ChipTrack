"""Tile the same crop region from several videos at given times.

  python compare_crops.py <videos_dir> <out.jpg> <x0> <y0> <x1> <y1> <tile_w> <cols> FILE@T[:TAG] ...
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from sheets import grab, label

vdir, out = Path(sys.argv[1]), sys.argv[2]
x0, y0, x1, y1 = map(float, sys.argv[3:7])
tw, cols = int(sys.argv[7]), int(sys.argv[8])
tiles = []
for spec in sys.argv[9:]:
    name, rest = spec.split("@")
    t, _, tag = rest.partition(":")
    cap = cv2.VideoCapture(str(vdir / name))
    f = grab(cap, float(t))
    h, w = f.shape[:2]
    f = f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    f = cv2.resize(f, (tw, int(f.shape[0] * tw / f.shape[1])))
    tiles.append(label(f, f"{Path(name).stem} {float(t):.1f}s {tag}"))
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
print(out)
