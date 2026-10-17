"""Download all model weights once into HF_HOME. After this the service runs offline
(set HF_HUB_OFFLINE=1). Usage: python scripts/download_weights.py [--vlm Qwen/Qwen3-VL-2B-Instruct]
"""
import argparse
import os

from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--vlm", default=os.environ.get("FORTE_VLM", "Qwen/Qwen3-VL-2B-Instruct"))
ap.add_argument("--detector", default=os.environ.get("FORTE_DETECTOR", "PekingU/rtdetr_v2_r18vd"))
a = ap.parse_args()
for repo in (a.detector, a.vlm):
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
    print("ok", repo, "->", path)
print("HF_HOME =", os.environ.get("HF_HOME", "~/.cache/huggingface"))
