"""Probe 2: ground the relevant anchor on each frame, crop+upscale it, then ask the cue question.

Usage: python probe_vlm_crops.py <probe_dir> <model_id> [--4bit]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

CUES = {  # cue -> (anchor to ground, question on the crop)
    "money_on_desk": ("desk", "This is a close-up of an office desk. Are there banknotes (paper money, a stack of bills) lying on it?"),
    "key_in_lock": ("small grey metal safe", "This is a close-up of a small metal safe. Is a key sticking out of its keyhole (key inserted in the lock)?"),
    "screen_working": ("laptop", "This is a close-up of a laptop. Is its screen showing a spreadsheet or document with text and table rows (not a lock screen, wallpaper or black screen)?"),
    "door_open": ("glass door", "This is a close-up of a glass door. Is the door standing open (not closed)?"),
    "safe_open": ("small grey metal safe", "This is a close-up of a small metal safe. Is the safe door open?"),
}

ap = argparse.ArgumentParser()
ap.add_argument("probe_dir")
ap.add_argument("model_id")
ap.add_argument("--4bit", dest="q4", action="store_true")
args = ap.parse_args()
kw = dict(dtype=torch.float16, device_map="cuda")
if args.q4:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kw).eval()
proc = AutoProcessor.from_pretrained(args.model_id, max_pixels=640 * 640)
tok = proc.tokenizer
YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]


def build(img, text):
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(m, tokenize=True, add_generation_prompt=True, return_dict=True,
                                    return_tensors="pt").to(model.device)


@torch.inference_mode()
def p_yes(img, q):
    lg = model(**build(img, q + " Answer with one word: yes or no.")).logits[0, -1].float()
    return torch.sigmoid(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)).item()


@torch.inference_mode()
def ground(img, name):
    inp = build(img, f'Output the bounding box of the {name} as JSON {{"bbox_2d":[x1,y1,x2,y2]}}.')
    out = model.generate(**inp, max_new_tokens=40, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", txt)
    if not m:
        return None
    x1, y1, x2, y2 = (int(v) / 1000 for v in m.groups())  # Qwen3-VL: relative 0..1000
    return x1 * img.width, y1 * img.height, x2 * img.width, y2 * img.height


def crop(img, box, margin=0.25, min_side=448):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - margin * w), max(0, y1 - margin * h)
    x2, y2 = min(img.width, x2 + margin * w), min(img.height, y2 + margin * h)
    c = img.crop((int(x1), int(y1), int(x2), int(y2)))
    s = max(1.0, min_side / max(c.size))
    return c.resize((int(c.width * s), int(c.height * s)), Image.BICUBIC)


out_dir = Path(args.probe_dir) / "crops"
out_dir.mkdir(exist_ok=True)
index = json.loads((Path(args.probe_dir) / "index.json").read_text(encoding="utf-8"))
rows = []
for it in index:
    if it["cue"] not in CUES:
        continue
    anchor, q = CUES[it["cue"]]
    img = Image.open(it["path"]).convert("RGB")
    t = time.time()
    box = ground(img, anchor)
    tg = time.time() - t
    if box is None:
        print(f"{it['cue']:15s} {Path(it['path']).stem:28s} GROUNDING FAILED")
        rows.append((it["cue"], it["expected"], None))
        continue
    c = crop(img, box)
    c.save(out_dir / Path(it["path"]).name)
    p = p_yes(c, q)
    if it["cue"] == "money_on_desk":  # alternative phrasing, keep the max
        p = max(p, p_yes(c, "Is there a stack of money or a pile of banknotes anywhere on the table?"))
    rows.append((it["cue"], it["expected"], p))
    print(f"{it['cue']:15s} {Path(it['path']).stem.split('__')[1]:12s} exp={str(it['expected']):5s} "
          f"p_yes={p:.3f}  ground={tg:.1f}s box={[int(v) for v in box]}")

print()
for cue in CUES:
    r = [x for x in rows if x[0] == cue and x[2] is not None]
    pos = [x[2] for x in r if x[1]]
    neg = [x[2] for x in r if not x[1]]
    if pos and neg:
        print(f"  {cue:15s} pos_min={min(pos):.3f} neg_max={max(neg):.3f} separable={min(pos) > max(neg)}")
