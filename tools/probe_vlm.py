"""Probe a local VLM (Qwen3-VL) on the ground-truth cue frames.

Measures P(yes) for a yes/no question per cue, latency, VRAM, and tries anchor grounding.
Usage: python probe_vlm.py <probe_dir> <model_id> [--4bit] [--max-pixels N]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTIONS = {
    "money_on_desk": "Look at the office desk. Are there banknotes (paper money) lying on the desk surface?",
    "safe_open": "Look at the small grey metal safe standing on a white cabinet. Is the door of the safe open?",
    "key_in_lock": "Look at the small grey metal safe. Is a key inserted in the lock of the safe door?",
    "screen_working": "Look at the laptop screen. Does it show a working document or spreadsheet with text (not a lock screen, picture wallpaper or black screen)?",
    "drinking": "Is the person at the desk drinking from a cup right now?",
    "personal_items": "Is there a coat, jacket or handbag lying on the desk?",
    "door_open": "Look at the glass door of the room. Is the door open?",
    "bag_on_desk": "Is there a green fabric bag lying on the desk?",
}

ap = argparse.ArgumentParser()
ap.add_argument("probe_dir")
ap.add_argument("model_id")
ap.add_argument("--4bit", dest="q4", action="store_true")
ap.add_argument("--max-pixels", type=int, default=464 * 848)
args = ap.parse_args()

kw = dict(dtype=torch.float16, device_map="cuda")
if args.q4:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kw).eval()
proc = AutoProcessor.from_pretrained(args.model_id, max_pixels=args.max_pixels)
print(f"load {time.time() - t0:.1f}s  vram={torch.cuda.memory_allocated() / 2**30:.2f}GB")

tok = proc.tokenizer
YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]


def build(img, text):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt").to(model.device)


@torch.inference_mode()
def p_yes(img, q):
    inp = build(img, q + " Answer with one word: yes or no.")
    logits = model(**inp).logits[0, -1].float()
    y = torch.logsumexp(logits[YES], 0)
    n = torch.logsumexp(logits[NO], 0)
    return torch.sigmoid(y - n).item(), inp["input_ids"].shape[1]


@torch.inference_mode()
def generate(img, text, n=200):
    inp = build(img, text)
    out = model.generate(**inp, max_new_tokens=n, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


index = json.loads((Path(args.probe_dir) / "index.json").read_text(encoding="utf-8"))
rows, lat = [], []
for it in index:
    img = Image.open(it["path"]).convert("RGB")
    t = time.time()
    p, ntok = p_yes(img, QUESTIONS[it["cue"]])
    torch.cuda.synchronize()
    lat.append(time.time() - t)
    rows.append((it["cue"], Path(it["path"]).stem.split("__")[1], it["expected"], p))
    print(f"{it['cue']:15s} {rows[-1][1]:12s} exp={str(it['expected']):5s} p_yes={p:.3f}  {lat[-1]:.2f}s tok={ntok}")

print(f"\nlatency median {sorted(lat)[len(lat) // 2]:.2f}s  peak vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
print("\nper-cue: separation (min p_yes of positives vs max p_yes of negatives), acc@0.5")
for cue in QUESTIONS:
    r = [x for x in rows if x[0] == cue]
    pos = [x[3] for x in r if x[2]]
    neg = [x[3] for x in r if not x[2]]
    acc = sum((x[3] >= 0.5) == x[2] for x in r) / len(r)
    print(f"  {cue:15s} pos_min={min(pos):.3f} neg_max={max(neg):.3f} separable={min(pos) > max(neg)} acc@0.5={acc:.2f}")

# anchor grounding test on two different layouts
for name in ["safe_open__X-21@12.0.jpg", "door_open__X-33@8.0.jpg", "money_on_desk__X-4@55.0.jpg"]:
    img = Image.open(Path(args.probe_dir) / name).convert("RGB")
    t = time.time()
    txt = generate(img, "Detect the following objects and output their bounding boxes as JSON list "
                        "[{\"label\":..., \"bbox_2d\":[x1,y1,x2,y2]}]: safe, door, desk, laptop, "
                        "banknote counting machine, person.")
    print(f"\n[grounding {name}] {time.time() - t:.1f}s\n{txt}")
