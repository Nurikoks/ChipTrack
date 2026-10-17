"""Benchmark RTMPose (rtmlib, Apache-2.0, ONNX/CPU) on detector person boxes.

Usage: python bench_pose.py <video> <t_sec>...
"""
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from rtmlib import RTMPose
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

MID = "PekingU/rtdetr_v2_r18vd"
proc = AutoImageProcessor.from_pretrained(MID)
det = RTDetrV2ForObjectDetection.from_pretrained(MID, dtype=torch.float16).cuda().eval()
t0 = time.time()
pose = RTMPose("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
               model_input_size=(192, 256), backend="onnxruntime", device="cpu")
print(f"pose load {time.time() - t0:.1f}s")

cap = cv2.VideoCapture(sys.argv[1])
times = []
for t in map(float, sys.argv[2:]):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    inp = proc(images=[Image.fromarray(f[:, :, ::-1])], return_tensors="pt").to("cuda", torch.float16)
    with torch.inference_mode():
        r = proc.post_process_object_detection(det(**inp), threshold=0.5, target_sizes=[f.shape[:2]])[0]
    boxes = [b.tolist() for b, l in zip(r["boxes"], r["labels"]) if det.config.id2label[l.item()] == "person"]
    ts = time.time()
    kps, scores = pose(f, bboxes=boxes) if boxes else (np.zeros((0, 17, 2)), np.zeros((0, 17)))
    times.append((time.time() - ts) / max(1, len(boxes)))
    for k, s in zip(kps, scores):
        # wrists (9,10) vs nose (0): hand-to-mouth distance relative to shoulder width
        sw = np.linalg.norm(k[5] - k[6]) + 1e-6
        d = min(np.linalg.norm(k[9] - k[0]), np.linalg.norm(k[10] - k[0])) / sw
        print(f"t={t:5.1f} persons={len(boxes)} wrist-nose/shoulder={d:.2f} mean_kp_score={s.mean():.2f}")
print(f"pose ms/person: {1000 * np.median(times):.0f}")
