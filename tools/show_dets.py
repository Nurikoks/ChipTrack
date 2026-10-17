"""Draw RT-DETR person boxes (score, height ratio) on frames. Usage: python show_dets.py <out.jpg> FILE@T ..."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.pipeline.models import Detector  # noqa: E402

det = Detector("PekingU/rtdetr_v2_r18vd")
tiles = []
for spec in sys.argv[2:]:
    f, t = spec.split("@")
    cap = cv2.VideoCapture(str(ROOT / "videos" / f))
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
    ok, fr = cap.read()
    for d in det([fr], threshold=0.3)[0]:
        if d["label"] != "person":
            continue
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(fr, f"{d['score']:.2f} h={y2 - y1}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(fr, spec, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    tiles.append(cv2.resize(fr, (232, 424)))
cv2.imwrite(sys.argv[1], np.hstack(tiles))
