# Architecture (Phase 3, historical)

> This document describes the original prototype. For the current ChipTRick implementation, accounts, feedback, training, versioning and limitations, see `CHIPTRICK_GUIDE.md`. Hardware and queue semantics below are historical and do not describe the current machine.

## 1. Constraints that shaped the design

| Constraint | Consequence |
|---|---|
| Single laptop GPU: RTX 3050 **4 GB VRAM**, 16 GB RAM | One small detector plus **one** ≤4B VLM in 4-bit. No ensembles, no 7B models |
| ≤ 5 min per video, from upload, for any clip ≤ 145 s; the jury may upload several at once | One GPU worker, so jobs run one at a time. Target is **≤ 60 s per clip** so a queue of 4–5 still meets the limit. Each job carries a deadline (upload + 240 s) and the pipeline drops low-priority checks when time runs out (`partial: true`) |
| Fully local; offline after the weights download | Hugging Face weights are cached in `HF_HOME`, production runs with `HF_HUB_OFFLINE=1`, and no network calls happen at inference |
| Only permissive licenses | Ultralytics YOLO (AGPL-3.0) and Qwen2.5-VL-3B (Qwen research license) are rejected; see §4 |
| Tiny labelled set: 21 positive clips over 11 types (1–3 each), and the test uses **a different camera angle** | No end-to-end training. Each type is a **trigger** (a relation between people and scene objects over time) plus a **constrained yes/no visual check** on a targeted crop. Scene objects are located per video, never from fixed pixel zones |
| Output is a hint; only visible facts may be stated | Every finding is backed by an explicit VLM yes/no answer on the evidence frame. Descriptions use templates that state what is visible. Rules come only from the verified registry |

## 2. Pipeline

```
 upload (.mov) ─► job queue (1 GPU worker, per-job deadline) ─► analyze():

 1  Decode + sample        OpenCV, 2 fps (≤ 290 frames for 145 s)                    ~1–2 s
 2  Perception             RT-DETRv2-r18 (COCO): persons (+cup, laptop, bags)         ~6 s / 145 s clip
                           person filter: drops small, far boxes (people behind or reflected in glass)
 3  Scene grounding        Qwen3-VL grounds: desk, safe, laptop, glass door           ~12 s
                           on the least-occluded frames; boxes are per video, so camera-angle agnostic
 4  Tracking + timeline    greedy IoU tracker; a track counts as "in the room" only if it reaches the
                           workplace or crosses below the door's lower edge (removes people seen through
                           the glass wall); relative geometry only: workplace attended (desk ∪ safe),
                           at/beside safe, person count (smoothed)
 5  Event triggers         entry/exit = room tracks that start/end at the grounded door, last ≥ 5 s and
                           are not a continuation of another track (person standing up);
                           departure = workplace unattended ≥ 2 s (geometry) OR VLM occupancy scan
                           ("is someone at the desk?", every 2 s); client changeover; periodic scan (4 s)
 6  Visual checklist       Qwen3-VL P(yes) on targeted crops (desk, laptop, safe, door) or full frame
                           money / bag / docs / screen content / safe open / key / personal items /
                           drinking / carrying a bag / document handover / counting / extra person
 7  Aggregate              threshold per question, merge per type within 12 s,
                           evidence frames (abandonment frame + state frame with highlighted box)
 8  Rule mapper            type → data/rules_registry.json → description text + "rule" object
 9  Verdict                violation if ≥ 1 confirmed finding; otherwise no_violation
```

### Trigger → check → type

| Trigger (from the timeline) | Visual check (crop) | Type |
|---|---|---|
| Workplace unattended ≥ 2 s | desk (two upscaled half-desk tiles): banknotes / green bag; else documents | T01 |
| same | laptop: working content on screen, gated by "lid open" and "not a lock screen" | T02 |
| same | safe: door open at the moment of leaving **and** at the end of the absence, nobody beside it | T03 |
| same | safe: key in lock (low recall, see limits) | T04 |
| Entry within 40 s after an exit | desk tiles at +1/+2/+3 s: banknotes | T05 |
| Entry with ≥ 2 people present, not a collector | standing person next to the worker at +1.5 s and +3 s; visitor not seated | T06 |
| Entry into an empty room | door crop open at +3 s and +5 s | T07 |
| Exit at the door, only the worker left, visitor chair empty | worker already counting or loading the machine at +1 s | T08 |
| Entry carrying a green bag | document handover before the bag lies alone on the desk (**experimental, off by default**) | T09 |
| Periodic, every 4 s | full frame: coat/handbag on the desk (not the green cash bag) | T10 |
| Periodic, every 4 s | full frame: drinking, confirmed again 1 s later | T11 |
| Periodic, every 4 s | safe open with nobody at it, then checked every 2 s: open ≥ 9 s, or ≥ 5 s and still open at the end | T03 |

**Adding a type** means adding one question, one trigger rule and one registry mapping. No retraining.

**Timecode convention** (agreed in the taxonomy): for state or omission types, the moment the worker walks away from the workplace or the person enters. For action types, the frame of the action.

## 3. Measured on this machine (RTX 3050 Laptop 4 GB, i5-12450H)

| Component | Measurement |
|---|---|
| Decode 142.5 s clip at 2 fps (285 frames) | 1.6 s |
| RT-DETRv2-r18 fp16, batch 8 | **21 ms/frame**, 0.42 GB VRAM |
| Qwen3-VL-2B nf4: yes/no question (~400 visual tokens) | **0.38 s/question**, 1.7 GB VRAM peak |
| Qwen3-VL-2B nf4: single-object grounding | ~3 s/object |
| RTMPose-m (ONNX, CPU) | 21 ms/person (not used, see §5) |
| End-to-end (first 8 clips, MVP) | 22–53 s per clip |

### VLM capability probe (44 frames with known answers; `tools/probe_vlm*.py`)

| Cue | Full frame (2B) | Grounded crop (2B) |
|---|---|---|
| drinking | separable (pos ≥ 0.97, neg ≤ 0.03) | — |
| coat/handbag on desk | separable (1.00 vs ≤ 0.004) | — |
| green bag on desk | separable (≥ 0.995 vs ≤ 0.001) | — |
| safe door open | separable (≥ 0.45 vs ≤ 0.08) | separable (≥ 0.99 vs ≤ 0.06) |
| laptop working screen | **not** separable | separable (≥ 0.98 vs ≤ 0.45) |
| door open | weak (0.73 vs 0.67) | weak (0.76 vs 0.66) |
| small banknote stack on desk | fails (only large piles found) | fails |
| key in safe lock | fails | fails (the key is ~3–5 px) |

The 4B model (nf4, 0.47 s/question) separates screens and safes even better, but is worse on small cash stacks and also fails on the key, while being 1.7× slower. **2B is the default** to protect the 5-minute limit under concurrent uploads; 4B can be selected with `FORTE_VLM`.

## 4. Models and libraries (all permissive; versions pinned in `requirements.txt`)

| Component | Model / library | License | Size | Role |
|---|---|---|---|---|
| Detector | `PekingU/rtdetr_v2_r18vd` (RT-DETRv2, COCO) | Apache-2.0 | 77 MB | persons, cups, laptops |
| VLM | `Qwen/Qwen3-VL-2B-Instruct` (alternative: `Qwen3-VL-4B-Instruct`) | Apache-2.0 | 4.3 GB on disk, 1.6 GB VRAM in nf4 | grounding + yes/no checks |
| Framework | PyTorch 2.11 (cu128) | BSD-3 | | |
| | transformers 5.20, accelerate, huggingface_hub | Apache-2.0 | | |
| | bitsandbytes 0.50 | MIT | | 4-bit quantization |
| | OpenCV (opencv-python-headless) | Apache-2.0 (bundled FFmpeg is LGPL) | | decoding |
| | FastAPI | MIT | | API |
| | uvicorn | BSD-3 | | server |
| | NumPy, pandas | BSD-3 | | |
| | Pillow | MIT-CMU (HPND) | | |
| Dev only (not shipped) | imageio-ffmpeg (FFmpeg binary, GPL build) | BSD-2 wrapper | | Phase 0 probing |
| Dev only (evaluated, not used) | rtmlib + RTMPose ONNX | Apache-2.0 | | |

**Rejected:**
* Ultralytics YOLOv8/11: AGPL-3.0.
* Qwen2.5-VL-3B: Qwen research license, no commercial use.
* PaliGemma/Gemma: custom terms with use restrictions.
* Any cloud API: forbidden by the rules.

## 5. Decisions and rejected alternatives

* **Not end-to-end video classification or training.** 21 positives across 11 types would only learn the room layout. That fails on the new angle and on unseen types.
* **Pose-based drinking rejected.** On X-46 the wrist-to-nose ratio was *higher* while drinking than while not (0.99–1.08 vs 0.88). The top-down view and seated posture make keypoints unreliable. The VLM separates drinking perfectly.
* **Fixed zones rejected.** Desk, safe, laptop and door are grounded per video. All geometry is relative to anchor size: person centre inside the anchor expanded by k × its width/height.
* **Crops over full frames** for small objects. Measured gains: screen content went from non-separable to separable, and the safe-open margin improved from 0.37 to 0.93.
* **Glass reflections:** person boxes under 45 % of the clip's 80th-percentile person height are dropped, so people outside the glass don't count.

## 6. Robustness to a new camera angle

* No pixel constants. Everything comes from per-video grounding plus relative geometry. The VLM questions are angle-independent by construction.
* Planned **pseudo-new-angle set** (Phase 5): held-out clips warped by perspective transform, rotation ±10°, crop/zoom, horizontal flip and brightness change, then the full pipeline re-run to measure the drop.
* Known angle-sensitive cues: key in lock and screen content (small objects) and door state. Their confidence is capped and they are listed as limitations.

## 7. Limitations and known error cases

| Area | Limitation | Seen on |
|---|---|---|
| T04 key in lock | The key is ~3–5 px; neither the 2B nor the 4B VLM detects it even on upscaled crops. Recall is ≈0 unless the camera is closer | X-13, X-22 |
| T09 collector without documents | Bag handover vs document handover can't be ordered reliably (one positive, one negative clip). **Disabled by default** (`FORTE_EXPERIMENTAL=1` enables it) | X-16, X-17 |
| T07 door left open | Glass door state is ambiguous to the VLM (closed doors score 0.67–0.97). Low recall, capped confidence | X-32, X-33 |
| T11 drinking | From behind, a cashier holding banknotes near the face can be read as drinking (2B), and neither face orientation nor a cup-in-hand gate fixes it | X-1, X-38 |
| T06 extra person | Depends on reliable entry detection; people walking behind the glass next to the door are filtered only by duration (≥ 5 s) | X-18 |
| Small cash stacks | Tiling helps (X-4, X-46), but a small blue stack on a busy desk is still missed | X-38 |
| Person count | People outside the glass wall are removed by the "track reaches the room" rule; in layouts where the glass line meets the desk edge, some remain | X-4, X-34 |
| Unresolved training events | X-1, X-2, X-44 (and X-47's second event) have no identified visible cue; reported as missed or found only through other types | — |
| Dark scenes | Lights switched off: laptop screens and safe states are less reliable | X-11, X-12 |
| Timecodes | Departure timecodes follow the "moment of leaving" convention, but the occupancy scan has a 2 s period, giving ±2 s jitter | — |
| Interpretation | Every finding is a *possible* violation based on what is visible. Anything requiring ledgers, documents' content or intent is out of scope by design | — |

## 8. Verdict policy

* `violation` if at least one finding passes its threshold. Thresholds are set recall-first on the probe set, then raised only where the training clips show false positives.
* The `no_violation` verdict carries no hedge flag. A partial analysis (deadline hit) still reports everything found so far and sets `partial: true`.
