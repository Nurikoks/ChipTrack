"""Ground-truth probe set of single frames for key visual cues (from the Phase 1 review).

Each item: (cue, file, t_sec, expected_answer). Used to measure detector / VLM capability
before committing to an architecture. Frames are saved to <out_dir>/<cue>__<file>@<t>.jpg
Usage: python probe_set.py <videos_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2

PROBES = [
    # banknotes lying on the desk
    ("money_on_desk", "Х-26.mov", 11.5, True), ("money_on_desk", "Х-4.mov", 55.0, True),
    ("money_on_desk", "Х-38.mov", 30.0, True), ("money_on_desk", "Х-46.mov", 8.0, True),
    ("money_on_desk", "Х-25.mov", 31.0, False), ("money_on_desk", "Х-27.mov", 25.9, False),
    ("money_on_desk", "Х-19.mov", 47.0, False), ("money_on_desk", "Х-5.mov", 62.0, False),
    # safe door open
    ("safe_open", "Х-20.mov", 6.5, True), ("safe_open", "Х-13.mov", 49.0, True),
    ("safe_open", "Х-35.mov", 43.2, True), ("safe_open", "Х-39.mov", 33.3, True),
    ("safe_open", "Х-21.mov", 12.0, False), ("safe_open", "Х-23.mov", 17.7, False),
    ("safe_open", "Х-22.mov", 20.5, False), ("safe_open", "Х-14.mov", 50.0, False),
    # key left in the safe lock
    ("key_in_lock", "Х-22.mov", 20.5, True), ("key_in_lock", "Х-13.mov", 61.0, True),
    ("key_in_lock", "Х-22.mov", 5.0, True),
    ("key_in_lock", "Х-23.mov", 17.7, False), ("key_in_lock", "Х-14.mov", 50.0, False),
    ("key_in_lock", "Х-21.mov", 12.0, False),
    # laptop screen shows working content (spreadsheet) vs lock screen / black / closed
    ("screen_working", "Х-28.mov", 9.5, True), ("screen_working", "Х-11.mov", 15.0, True),
    ("screen_working", "Х-29.mov", 15.0, False), ("screen_working", "Х-23.mov", 16.0, False),
    ("screen_working", "Х-25.mov", 29.0, False), ("screen_working", "Х-27.mov", 24.0, False),
    # person drinking
    ("drinking", "Х-46.mov", 11.5, True), ("drinking", "Х-46.mov", 13.0, True),
    ("drinking", "Х-46.mov", 8.0, False), ("drinking", "Х-25.mov", 3.0, False),
    # coat / handbag on the desk
    ("personal_items", "Х-40.mov", 19.0, True), ("personal_items", "Х-40.mov", 24.0, True),
    ("personal_items", "Х-33.mov", 12.0, False), ("personal_items", "Х-43.mov", 10.0, False),
    # cash-room door open
    ("door_open", "Х-33.mov", 8.0, True), ("door_open", "Х-32.mov", 7.0, True),
    ("door_open", "Х-31.mov", 12.0, False), ("door_open", "Х-43.mov", 10.0, False),
    # green cash-collection bag on desk
    ("bag_on_desk", "Х-15.mov", 18.0, True), ("bag_on_desk", "Х-17.mov", 12.0, True),
    ("bag_on_desk", "Х-12.mov", 40.0, False), ("bag_on_desk", "Х-29.mov", 16.0, False),
]

if __name__ == "__main__":
    vdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for cue, f, t, exp in PROBES:
        cap = cv2.VideoCapture(str(vdir / f))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            print("FAIL", f, t)
            continue
        name = f"{cue}__{Path(f).stem.replace('Х', 'X')}@{t}.jpg"
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index.append(dict(cue=cue, file=f, t=t, expected=exp, path=str(out / name)))
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(index), "probe frames")
