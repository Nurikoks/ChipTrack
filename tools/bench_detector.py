"""Benchmark decode + RT-DETRv2 (Apache-2.0, COCO) on a full video at a given sampling fps.

Usage: python bench_detector.py <video> [fps] [model_id]
"""
import sys
import time
from collections import Counter

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

video = sys.argv[1]
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
mid = sys.argv[3] if len(sys.argv) > 3 else "PekingU/rtdetr_v2_r18vd"

t0 = time.time()
proc = AutoImageProcessor.from_pretrained(mid)
model = RTDetrV2ForObjectDetection.from_pretrained(mid, dtype=torch.float16).cuda().eval()
print(f"load {time.time() - t0:.1f}s")

cap = cv2.VideoCapture(video)
vfps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, round(vfps / fps))
t_dec = t_det = 0.0
frames, counts = 0, Counter()
batch, B = [], 8


def run(batch):
    global t_det
    t = time.time()
    inp = proc(images=batch, return_tensors="pt").to("cuda", torch.float16)
    with torch.inference_mode():
        out = model(**inp)
    res = proc.post_process_object_detection(out, threshold=0.4,
                                             target_sizes=[(im.height, im.width) for im in batch])
    torch.cuda.synchronize()
    t_det += time.time() - t
    for r in res:
        for lab in r["labels"].tolist():
            counts[model.config.id2label[lab]] += 1


i = 0
while True:
    t = time.time()
    ok = cap.grab()
    if not ok:
        break
    if i % step == 0:
        ok, f = cap.retrieve()
        t_dec += time.time() - t
        batch.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        frames += 1
        if len(batch) == B:
            run(batch)
            batch = []
    else:
        t_dec += time.time() - t
    i += 1
if batch:
    run(batch)
dur = n / vfps
print(f"video {dur:.1f}s, {frames} frames @ {fps} fps | decode {t_dec:.1f}s | detect {t_det:.1f}s "
      f"({1000 * t_det / frames:.0f} ms/frame) | vram peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
print("detections:", counts.most_common(15))
