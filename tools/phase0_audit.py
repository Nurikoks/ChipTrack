"""Phase 0: probe every video and build a clean labels.csv.

Usage: python phase0_audit.py <project_dir>
"""
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import cv2
import imageio_ffmpeg
import openpyxl
import pandas as pd

ROOT = Path(sys.argv[1])
VIDEOS = ROOT / "videos"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

# Cyrillic look-alikes that appear in file names / labels -> Latin
CYR2LAT = str.maketrans({"Х": "X", "х": "X", "Ч": "X"})


def norm_id(name: str) -> str:
    """'Х-47.mov' / 'X-47' / ' х 47 ' -> 'X-47'."""
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = re.sub(r"\.mov$", "", s, flags=re.I).translate(CYR2LAT).upper()
    m = re.match(r"^X\s*[-_ ]?\s*(\d+)$", s)
    if not m:
        raise ValueError(f"cannot normalize {name!r}")
    return f"X-{int(m.group(1))}"


def parse_ts(v) -> float | None:
    """'22 секунда' -> 22; '1минута 8 секунда' -> 68; numbers pass through."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).lower()
    mins = re.search(r"(\d+(?:[.,]\d+)?)\s*мин", s)
    secs = re.search(r"(\d+(?:[.,]\d+)?)\s*сек", s)
    if not mins and not secs:
        nums = re.findall(r"\d+(?:[.,]\d+)?", s)
        if len(nums) == 1:
            return float(nums[0].replace(",", "."))
        raise ValueError(f"cannot parse timestamp {v!r}")
    total = 0.0
    if mins:
        total += 60 * float(mins.group(1).replace(",", "."))
    if secs:
        total += float(secs.group(1).replace(",", "."))
    return total


def ffprobe_codec(path: Path) -> str:
    # imageio-ffmpeg ships ffmpeg only; parse its stderr banner for the codec line
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Video: ([^\n]+)", r.stderr)
    return m.group(1).strip() if m else "?"


rows = []
for p in sorted(VIDEOS.glob("*.mov")):
    cap = cv2.VideoCapture(str(p))
    ok_open = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok_read, _ = cap.read()
    cap.release()
    rows.append(dict(
        file=p.name, video_id=norm_id(p.name), opens=ok_open and ok_read,
        width=w, height=h, fps=round(fps, 3), frames=n,
        duration_s=round(n / fps, 2) if fps else None,
        size_mb=round(p.stat().st_size / 2**20, 2),
        md5=hashlib.md5(p.read_bytes()).hexdigest(),
        codec=ffprobe_codec(p),
    ))
meta = pd.DataFrame(rows)

wb = openpyxl.load_workbook(ROOT / "open_labels.xlsx")
ws = wb["labels"]
lab = []
for r in list(ws.iter_rows(values_only=True))[1:]:
    if r[0] is None:
        continue
    lab.append(dict(raw_name=r[0], video_id=norm_id(r[0]), raw_timestamp=r[1],
                    timestamp_sec=parse_ts(r[1]),
                    label="violation" if str(r[2]).strip().lower() == "нарушение" else "no_violation"))
labels = pd.DataFrame(lab)

df = labels.merge(meta, on="video_id", how="outer", indicator=True)
dup = df.groupby("md5")["video_id"].transform(lambda s: ",".join(sorted(s)) if len(s) > 1 else "")
df["duplicate_of"] = [",".join(x for x in d.split(",") if x != v) if d else "" for d, v in zip(dup, df["video_id"])]
df["num"] = df["video_id"].str.extract(r"(\d+)").astype(int)
df = df.sort_values("num").drop(columns="num")

df[["video_id", "file", "label", "timestamp_sec", "raw_timestamp", "duplicate_of"]].to_csv(
    OUT / "labels.csv", index=False, encoding="utf-8")
df.to_csv(OUT / "video_meta.csv", index=False, encoding="utf-8")

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 100)
print(df[["video_id", "file", "label", "timestamp_sec", "width", "height", "fps", "duration_s",
          "size_mb", "opens", "duplicate_of", "_merge"]].to_string(index=False))
print("\ncodecs:", df["codec"].value_counts().to_dict())
print("labels:", df["label"].value_counts().to_dict())
print("ts beyond duration:", df[df.timestamp_sec > df.duration_s][["video_id", "timestamp_sec", "duration_s"]].to_dict("records"))
print("duration stats:", df["duration_s"].describe().round(1).to_dict())
