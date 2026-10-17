"""Build one example-frame sheet per candidate violation type (Phase 1 evidence).

Each row = one clip: 3 frames (before / at label / after) + optional zoom tile.
Usage: python make_taxonomy_frames.py <videos_dir> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from sheets import grab

VDIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SCENE = (0.0, 0.02, 1.0, 0.62)  # relative crop that contains the whole cash-desk scene
TW = 260

# type_id -> rows of (file, [t...], row caption, optional zoom (t, box))
SPEC = {
    "T01_UNATTENDED_VALUABLES": [
        ("Х-26.mov", [8.0, 9.5, 12.0], "X-26 V@9: cash put on desk, cashier walks away", None),
        ("Х-15.mov", [6.0, 15.0, 19.0], "X-15 V@15: cash-collection bag left on desk", None),
        ("Х-24.mov", [4.0, 8.0, 12.0], "X-24 V@8: folder/items left on desk", None),
        ("Х-25.mov", [2.0, 20.0, 31.0], "X-25 NORMAL: desk cleared before leaving", None),
    ],
    "T02_PC_UNLOCKED_UNATTENDED": [
        ("Х-28.mov", [5.0, 7.5, 10.0], "X-28 V@7: leaves, spreadsheet open on screen", (9.5, (0.45, 0.26, 0.8, 0.46))),
        ("Х-11.mov", [10.0, 18.0, 24.0], "X-11 V@18: lights off, screen still unlocked", (15.0, (0.45, 0.26, 0.8, 0.46))),
        ("Х-29.mov", [5.0, 9.0, 15.0], "X-29 NORMAL: leaves with lock screen", (15.0, (0.45, 0.26, 0.8, 0.46))),
    ],
    "T03_SAFE_LEFT_OPEN": [
        ("Х-20.mov", [2.0, 5.0, 7.5], "X-20 V@5: safe door left open, cashier at desk", None),
        ("Х-21.mov", [3.0, 7.0, 10.0], "X-21 NORMAL: safe closed before sitting", None),
    ],
    "T04_KEY_LEFT_IN_SAFE": [
        ("Х-22.mov", [10.0, 14.0, 20.0], "X-22 V@14: leaves, key stays in safe lock", (20.5, (0.33, 0.24, 0.53, 0.42))),
        ("Х-13.mov", [44.0, 47.0, 61.0], "X-13 V@47: key in lock before/after, cashier leaves", (61.0, (0.36, 0.2, 0.62, 0.46))),
        ("Х-23.mov", [6.0, 10.0, 17.7], "X-23 NORMAL: no key in lock after leaving", (17.7, (0.33, 0.24, 0.53, 0.42))),
    ],
    "T05_CASH_EXPOSED_NEXT_CLIENT": [
        ("Х-4.mov", [53.0, 57.0, 59.0], "X-4 V@57: banknotes on desk as next client enters", (55.0, (0.0, 0.25, 0.75, 0.55))),
        ("Х-38.mov", [29.0, 33.0, 35.0], "X-38 V@33: banknotes on desk as next person enters", (31.0, (0.0, 0.25, 0.75, 0.55))),
        ("Х-19.mov", [45.0, 47.0, 50.0], "X-19 NORMAL: desk clear at client change", None),
    ],
    "T06_UNAUTHORIZED_PERSON_CASH_ZONE": [
        ("Х-30.mov", [11.0, 13.0, 15.0], "X-30 V@13: person stands beside cashier during counting", None),
        ("Х-18.mov", [16.5, 20.0, 21.0], "X-18 V@20: third person enters, item handed to him", None),
    ],
    "T07_CASH_ROOM_DOOR_NOT_CLOSED": [
        ("Х-32.mov", [3.0, 5.0, 8.0], "X-32 V@5: enters, walks to desk, door not closed", None),
        ("Х-33.mov", [4.5, 6.0, 11.0], "X-33 V@6: door left open, man in doorway at 11s", None),
        ("Х-31.mov", [7.0, 9.0, 10.0], "X-31 NORMAL: cashier turns back and closes door", None),
    ],
    "T08_COUNT_WITHOUT_CLIENT": [
        ("Х-34.mov", [7.0, 11.5, 13.5], "X-34 V@13: client leaves, cash loaded into counter", None),
        ("Х-39.mov", [11.0, 14.8, 18.5], "X-39 NORMAL: client stays seated during count", None),
    ],
    "T09_COLLECTOR_NO_DOC_CHECK": [
        ("Х-17.mov", [5.0, 7.0, 11.0], "X-17 V@7: bag put on desk directly, collector leaves", None),
        ("Х-16.mov", [9.0, 12.0, 16.0], "X-16 NORMAL: document shown+checked, then bag", None),
    ],
    "T10_PERSONAL_ITEMS_AT_DESK": [
        ("Х-40.mov", [13.0, 17.0, 19.5], "X-40 V@17: coat and handbag put on cash desk", None),
    ],
    "T11_FOOD_DRINK_AT_DESK": [
        ("Х-46.mov", [8.0, 11.0, 13.0], "X-46 V@11: drinks from cup, banknotes on desk", None),
        ("Х-25.mov", [1.0, 3.0, 5.0], "X-25 NORMAL: cup on desk, not drinking", None),
    ],
    "T99_UNRESOLVED": [
        ("X-1.mov", [19.5, 22.0, 23.0], "X-1 V@22: notes from counter, hand count, to client", None),
        ("X-2.mov", [9.0, 10.0, 13.0], "X-2 V@10: client lays item on desk, cashier handles", None),
        ("Х-44.mov", [13.0, 16.0, 19.0], "X-44 V@16: notes taken from counter, hand count", None),
        ("Х-47.mov", [64.0, 68.0, 70.0], "X-47 V@68: 2nd client's cash, old stack still on desk", None),
    ],
}


def tile(frame, box, tw, text):
    h, w = frame.shape[:2]
    f = frame[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
    th = int(round(f.shape[0] * tw / f.shape[1]))
    f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.rectangle(f, (0, 0), (tw, 20), (0, 0, 0), -1)
    cv2.putText(f, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return f


def fit_h(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


for type_id, rows in SPEC.items():
    out_rows = []
    for fname, ts, caption, zoom in rows:
        cap = cv2.VideoCapture(str(VDIR / fname))
        tiles = [tile(grab(cap, t), SCENE, TW, f"{t:.1f}s") for t in ts]
        rh = tiles[0].shape[0]
        if zoom:
            zt, zbox = zoom
            tiles.append(fit_h(tile(grab(cap, zt), zbox, TW, f"zoom {zt:.1f}s"), rh))
        row = np.hstack(tiles)
        bar = np.full((26, row.shape[1], 3), 40, np.uint8)
        color = (120, 220, 120) if "NORMAL" in caption else (90, 160, 255)
        cv2.putText(bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        out_rows.append(np.vstack([bar, row]))
    width = max(r.shape[1] for r in out_rows)
    out_rows = [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)]) for r in out_rows]
    path = OUT / f"{type_id}.jpg"
    cv2.imwrite(str(path), np.vstack(out_rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(path)
