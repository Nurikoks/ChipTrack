"""Local model wrappers: RT-DETRv2 person/object detector and Qwen3-VL verifier.

Both run fully offline once weights are in HF_HOME (set HF_HUB_OFFLINE=1 in production).
"""
import logging
import re
import time

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger("forte.models")

COCO_KEEP = {"person", "cup", "laptop", "handbag", "backpack", "suitcase", "cell phone", "book",
             "bottle", "chair", "tvmonitor", "tv", "diningtable", "dining table"}


class Detector:
    def __init__(self, model_id: str):
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
        t = time.time()
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id, dtype=torch.float16).cuda().eval()
        self.labels = self.model.config.id2label
        log.info("detector %s loaded in %.1fs", model_id, time.time() - t)

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray], threshold: float = 0.35, batch: int = 8) -> list[list[dict]]:
        out = []
        for i in range(0, len(frames), batch):
            ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[i:i + batch]]
            inp = self.proc(images=ims, return_tensors="pt").to("cuda", torch.float16)
            res = self.proc.post_process_object_detection(self.model(**inp), threshold=threshold,
                                                          target_sizes=[(im.height, im.width) for im in ims])
            for r in res:
                dets = []
                for s, l, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                    name = self.labels[l]
                    if name in COCO_KEEP:
                        dets.append({"label": name, "score": s, "box": b})
                out.append(dets)
        return out


class VLM:
    """Constrained yes/no verifier + single-object grounding (Qwen3-VL, Apache-2.0)."""

    def __init__(self, model_id: str, four_bit: bool = True, max_pixels: int = 640 * 640):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        t = time.time()
        kw = dict(dtype=torch.float16, device_map="cuda")
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
        self.calls = 0
        log.info("VLM %s loaded in %.1fs (vram %.2f GB)", model_id, time.time() - t,
                 torch.cuda.memory_allocated() / 2**30)

    def _inputs(self, img: Image.Image, text: str):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                             return_dict=True, return_tensors="pt").to(self.model.device)

    @staticmethod
    def to_pil(frame: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @torch.inference_mode()
    def p_yes(self, img: Image.Image, question: str) -> float:
        self.calls += 1
        lg = self.model(**self._inputs(img, question + " Answer with one word: yes or no.")).logits[0, -1].float()
        return torch.sigmoid(torch.logsumexp(lg[self.yes], 0) - torch.logsumexp(lg[self.no], 0)).item()

    @torch.inference_mode()
    def ground(self, img: Image.Image, name: str) -> tuple[float, float, float, float] | None:
        """Returns a box in pixel coords of `img`, or None."""
        self.calls += 1
        inp = self._inputs(img, f'Output the bounding box of the {name} as JSON {{"bbox_2d":[x1,y1,x2,y2]}}. '
                                f'If there is no {name}, output {{"bbox_2d":[]}}.')
        out = self.model.generate(**inp, max_new_tokens=40, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", txt)
        if not m:
            return None
        x1, y1, x2, y2 = (int(v) / 1000 for v in m.groups())  # Qwen3-VL uses relative 0..1000 coords
        if x2 <= x1 or y2 <= y1:
            return None
        return x1 * img.width, y1 * img.height, x2 * img.width, y2 * img.height


def crop(frame: np.ndarray, box, margin: float = 0.25, min_side: int = 448) -> Image.Image:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - margin * bw)), int(max(0, y1 - margin * bh))
    x2, y2 = int(min(w, x2 + margin * bw)), int(min(h, y2 + margin * bh))
    c = frame[y1:y2, x1:x2]
    s = max(1.0, min_side / max(c.shape[:2]))
    if s > 1.0:
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    return VLM.to_pil(c)
