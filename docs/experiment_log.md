# Experiment and decision log

| # | Date / phase | What | Result | Decision |
|---|---|---|---|---|
| 1 | 2026-10-17 · P0 | Hardware audit | RTX 3050 Laptop **4 GB VRAM** (driver 617.42, CUDA 13.4 UMD), i5-12450H 8C/12T, **15.7 GB RAM**, C: 15 GB free, E: 45 GB free | Venv and model weights go on E: (`E:\fortebank_env`); models must fit in 4 GB VRAM (small detector/pose, ≤3–4B quantized VLM) |
| 2 | P0 | Python env | System Python 3.12.4 with no packages; no ffmpeg on PATH | ffmpeg comes from the `imageio-ffmpeg` wheel; OpenCV 5.0 headless |
| 3 | P0 | Video probe (`tools/phase0_audit.py`) | 42/42 open, H.264 464×848, ~30 fps, 11.6–142.5 s | Decode with OpenCV, sample frames by timestamp |
| 4 | P0 | MD5 duplicate check | X-4 ≡ X-47 (labels @57 and @68), X-5 ≡ X-48 | Treat X-4/X-47 as one clip with 2 events; group duplicates in CV folds |
| 5 | P0 | Label normalization | Cyrillic Х vs Latin X normalized; "1минута 8 секунда" parsed as 68 s; all timestamps < duration | `data/labels.csv` |
| 6 | P1 | Visual review: contact sheets ±4–6 s around each label, full-timeline overviews of all 40 clips, high-res crops (safe lock, laptop screen, desk, door) | 18 of 22 events explained by a visible cue; 11 candidate types; 4 events unresolved | `taxonomy.md`, `data/labels_enriched.csv`, `docs/taxonomy_frames/` |
| 7 | P1 | Hypothesis "leaving with the laptop open is the violation" | **Rejected.** Normal X-23/X-29 also leave with the laptop open; the separating cue is screen *content* (spreadsheet vs lock screen/black) | T02 is defined by visible working content |
| 8 | P1 | Hypothesis "safe opened in front of a client is the violation" | **Rejected.** Normal X-35/X-39 open the safe with a client seated | T03 is about the safe left open *unattended* |
| 9 | P1 | Hand counting as a violation (X-44 vs X-43/X-45) | No visible separating cue | Left UNRESOLVED; human review requested |

| 10 | P2 | Normative search (NBRK, ARDFM, adilet). adilet.zan.kz renders client-side, so the text was taken from the zakon.uchet.kz mirror and each clause was string-matched on adilet in the in-app browser | 3 acts in force: NBRK No. 231/2019, NBRK No. 14/2020, ARDFM No. 53/2026; 20 clauses verified | `data/rules_registry.json`, `docs/normative_sources.md` |
| 11 | P2 | Workstation-lock rule | NBRK No. 48/2018 **lost force** 03.04.2026; replaced by ARDFM No. 53 (para. 71 subpara. 9 kept: auto-lock after 30 min) | T02 grounding = weak/indirect; softened wording |
| 12 | P2 | Food/drink rule | No explicit ban found in RK acts; only the allowed-items list in para. 12 part 2 of No. 231 | T11 grounding = weak (interpretation) |

| 13 | P3 | License audit (HF API) | Qwen3-VL-2B/4B, Qwen2-VL, SmolVLM2, InternVL3, RT-DETRv2, OWLv2, G-DINO: Apache-2.0; Florence-2: MIT; **Qwen2.5-VL-3B: no license tag (research license)** | Excluded Qwen2.5-VL-3B and Ultralytics (AGPL) |
| 14 | P3 | Detector benchmark, 142.5 s clip @ 2 fps | RT-DETRv2-r18: 21 ms/frame, 0.42 GB; whole clip 8 s | Detector is not the bottleneck |
| 15 | P3 | Pose for drinking (RTMPose) | 21 ms/person but wrist–nose ratio not separable (higher while drinking) | Dropped; VLM handles drinking |
| 16 | P3 | VLM probe, full frames (2B nf4) | drinking/personal/bag/safe-open separable; money, key, screen, door not | Use grounded crops |
| 17 | P3 | VLM probe, crops (2B) | screen and safe-open become separable; money and key still fail | |
| 18 | P3 | VLM probe, crops (4B nf4) | screen ≥0.996 vs 0.000; safe 1.0 vs ≤0.06; money and key fail; 0.47 s/q vs 0.28 s (2B) | Keep 2B default (speed); 4B as option, decided by end-to-end eval |
| 19 | P3 | Money with tight desk crop (margin 0.1) + 2 half-desk tiles + new prompts | X-4 found (0.94–1.0); X-46 0.45 (2B); X-38 still 0.12; negatives ≤0.007 | money TH = 0.3 |
| 20 | P3/5 | **v1 end-to-end** (8 clips) | 4/5 TP with correct type and timecode ≤1 s; 3 FP | MVP works |
| 21 | P5 | **v2 full set** (40 clips) | TP 16, FN 5, FP 6, TN 13; recall 0.76, precision 0.73; types 10/22; timecode median 4.5 s; 45 s/clip | FP analysis: person count broken by people behind the glass wall |
| 22 | P5 | **v3**: IoU tracker + "track in room" rule, desk ∪ safe workplace, T06/T10/T11 confirmations | TP 13, FN 8, FP 9, TN 10; timecode median 1.9 s; 28.5 s/clip | Several v2 "hits" were lucky wrong-type detections; new FPs from the periodic safe check; missed departures in layout B |
| 23 | P5 | v4/v5: VLM occupancy scan, safe-open duration rule, T09 window, lid/lock-screen gating, tiled money | superseded mid-run by v6 | |
| 24 | P5 | v6: door-based entries/exits, continuation rule, entry tracks ≥ 5 s | TP 14, FN 7, FP 6, TN 13 | FP analysis: safe visits counted as departures, T09 unstable, T08 late activity |
| 25 | P5 | **v7 (final)**: "beside safe" counts as workplace, safe must still be open at the end of the absence, T08 requires counting at +1 s, personal TH 0.7, T09 off by default | **TP 13, FN 8, FP 0, TN 19**; correct type 12/22; timecode median 2.0 s; 26.8 s/clip | Ship |
| 26 | P5 | T07 motion cue (not turning back to the door) | 0/2 gained, no regressions | Kept as a documented limitation |
| 27 | P5 | Drinking gates (cup in hand, face toward camera via VLM, pose face keypoints) | none separate a back-facing cashier holding cash from drinking | Documented limitation |
| 28 | P4/5 | Live API test, real models | auth/415/404/failed cases correct; 4 concurrent uploads done in 133 s | OK |

**Caveat on all numbers:** thresholds and rules were set by looking at these same 40 clips. The metrics are in-sample and optimistic. The pseudo-new-angle set (warped clips) is the closest thing to a held-out check available.

## Time per phase
* Phase 0 (environment and data audit): ~10 min
* Phase 1 (visual data review and taxonomy draft): ~25 min of wall clock in this session (frame extraction, ~60 contact sheets inspected)
