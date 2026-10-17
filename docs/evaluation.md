# Evaluation (Phase 5)

**Method.** Full pipeline run on all 40 unique training clips (duplicates X-47 = X-4 and X-48 = X-5 merged; X-4 carries both labelled events). Video-level verdict versus label; timecode error versus the labelled second; type checked against the taxonomy. Per-clip outputs: `runtime/eval_v8/results.jsonl` (v8), `runtime/eval_v8b/results.jsonl` (v8b).

**Important caveat:** all rules and thresholds were set by looking at these same clips, so every number below is **in-sample and optimistic**. The labelled set has 21 positive clips and 22 events over 12 types, several types with one or two events. It is too small for an honest held-out split per type. A pseudo-new-angle set (40 warped clips: perspective, rotation, mirror, crop, landscape) was built (`tools/make_pseudo_angles.py`) and evaluated: see [Pseudo-new-angle check](#pseudo-new-angle-check-v8-warped-clips). It shows how robust the rules are to geometry changes, but it is not a held-out set either: the thresholds were still tuned on the source clips.

## Progress across versions (40 clips)

| Version | TP | FN | FP | TN | Recall | Precision | Correct type* | Timecode median | Time per clip |
|---|---|---|---|---|---|---|---|---|---|
| v2 (first full run) | 16 | 5 | 6 | 13 | 0.76 | 0.73 | 10/22 | 4.5 s | 45 s |
| v3 (tracker, room filter) | 13 | 8 | 9 | 10 | 0.62 | 0.59 | 10/22 | 1.9 s | 28.5 s |
| v6 (occupancy, safe duration, door-based entries) | 14 | 7 | 6 | 13 | 0.67 | 0.70 | 10/22 | 2.6 s | 34 s |
| v7 (previous final, RTX 3050 4 GB) | 13 | 8 | 0 | 19 | 0.62 | 1.00 | 12/22 | 2.0 s | 26.8 s |
| **v8 (final, recall first, RTX 4060 8 GB)** | **18** | **3** | **9** | **10** | **0.86** | **0.67** | **18/22** | **1.9 s** | **25.8 s** |
| v8b (v8 + robustness fixes, RTX 4060 8 GB) | 18 | 3 | 9 | 10 | 0.86 | 0.67 | 18/22 | 1.9 s | 26.1 s |

\* Correct type = the labelled type is among the clip's predictions, anywhere in the clip (lenient). Stricter counts are in the v8 section.

v2's higher recall came largely from wrong-type detections triggered by person-count noise (people seen through the glass), so it was not a usable operating point.

**v7 pointer:** v7 had precision 1.00 at recall 0.62 (no false alarms on the 19 normal clips). It was the previous final version; v8 deliberately trades those false alarms for recall (see the trade-off below).

Timing is not comparable across machines: v7 ran on the RTX 3050 4 GB (the production target in the README), v8 and v8b ran on an RTX 4060 Laptop 8 GB. v8 has not been timed on the 3050.

## Final version (v8) in detail

Clips: 40 (violation 21, no violation 19). Run commit 4864832; v8b (commits e226394 and dfe1d33) gives identical predictions, types, timecodes and confidences on all 40 clips, with 0 partial results, 0 errors and no warnings.

| | predicted violation | predicted no violation |
|---|---|---|
| **label: violation** | 18 (TP) | 3 (FN) |
| **label: no violation** | 9 (FP) | 10 (TN) |

Recall (violation) **0.86** (18/21) · Precision **0.67** (18/27) · Accuracy **0.70** (28/40) · Specificity 0.53 (10/19)

95% Wilson intervals (in-sample, so these describe the 40 clips only): recall 0.65–0.95; specificity 0.32–0.73.

Timecode |error| (nearest prediction, same type preferred), 19/22 labelled events with a prediction: mean 3.1 s · median 1.9 s · max 20.9 s · ≤1 s 42% · ≤2 s 63% · ≤3 s 68%.
The max is X-4's second event (68 s): its only T05 prediction is at 47.1 s. `tools/evaluate.py` prints a different variant (nearest prediction of any type, same 19 events): mean 1.8 s, median 1.1 s, max 6.9 s. The table above uses the same-type variant from `tools/summarize_eval.py`.

Type agreement, strict and lenient (22 events; UNRESOLVED events can never match):

| Measure | Events |
|---|---|
| labelled type among the clip's predictions, anywhere in the clip (lenient) | 18/22 |
| same type predicted within ±5 s | 16/22 |
| same type predicted within ±3 s | 12/22 |

| Labelled type | events | correct type predicted | clip missed |
|---|---|---|---|
| T01_UNATTENDED_VALUABLES | 3 | 3 | 0 |
| T02_PC_UNLOCKED_UNATTENDED | 2 | 2 | 0 |
| T03_SAFE_LEFT_OPEN | 1 | 1 | 0 |
| T04_KEY_LEFT_IN_SAFE | 2 | 0 | 2 |
| T05_CASH_EXPOSED_NEXT_CLIENT | 3 | 3 | 0 |
| T06_UNAUTHORIZED_PERSON_CASH_ZONE | 2 | 2 | 0 |
| T07_CASH_ROOM_DOOR_NOT_CLOSED | 2 | 2 | 0 |
| T08_COUNT_WITHOUT_CLIENT | 1 | 1 | 0 |
| T09_COLLECTOR_NO_DOC_CHECK | 1 | 1 | 0 |
| T10_PERSONAL_ITEMS_AT_DESK | 1 | 1 | 0 |
| T11_FOOD_DRINK_AT_DESK | 1 | 1 | 0 |
| T12_CASH_HANDOVER_AT_DESK | 1 | 1 | 0 |
| UNRESOLVED | 2 | 0 | 1 |

Findings (all clips): 64 in total. T12 21, T06 16, T05 9, T07 4, T01 3, T09 3, T02 2, T03 2, T08 2, T10 1, T11 1. Only 16 of the 64 findings are correct (T12: 1 of 21; T06: 2 of 16; T05: 1 of 9).

**Per-type reliability (for reading the results, not a claim of accuracy):** solid T01, T02, T10, T11; moderate T03, T08; unreliable T05, T06, T07, T09, T12; never detected T04. Most types have one to three events, so even "solid" is a small sample.

### Failures

Misses (3 of 21 violation clips):
* FN X-13: label T04_KEY_LEFT_IN_SAFE @47, predicted nothing
* FN X-22: label T04_KEY_LEFT_IN_SAFE @14, predicted nothing
* FN X-44: label UNRESOLVED @16, predicted nothing

False alarms (9 of 19 normal clips): X-3 (T12 ×2), X-5 (T05, T12), X-12 (T06), X-16 (T09, T06, T12), X-19 (T05, T06, T12), X-31 (T07), X-35 (T06, T12, T05), X-37 (T12), X-39 (T12, T06).

### Why the misses happen
* **X-13, X-22 (T04 key left in the safe lock):** the key is 3–5 px; both Qwen3-VL-2B and 4B fail even on upscaled crops. Recall for this type is about zero unless the camera is closer.
* **X-44 (UNRESOLVED):** no visible violating element was identified in Phase 1, so no rule can target it.

### Caught in v8, not in v7
* X-17 (T09), X-18 (T06), X-32 and X-33 (T07). X-2 (UNRESOLVED) is also flagged, through T06 and T12 hits; its label has no type, so it is not a type match.
* X-30 was already a v7 true positive; in v8 it has the correct type (T06).
* No v7 true positive was lost: 13 + 5 = 18.

### False-alarm analysis

Counterfactuals below are in-sample, computed from the v8 predictions (no re-run):

| Setting | TP | FN | FP | TN | Type agreement | Effect |
|---|---|---|---|---|---|---|
| v8 as evaluated | 18 | 3 | 9 | 10 | 18/22 | — |
| T12 not counted toward the verdict | 18 | 3 | 7 | 12 | 17/22 | no TP lost; X-3 and X-37 become normal |
| T09 off | 17 | 4 | 9 | 10 | 17/22 | X-17 lost |
| T12 and T06 off | 16 | 5 | 5 | 14 | — | X-1 and X-2 lost |

**Team decision:** keep T12 counting toward the verdict (recall first). Turning it off removes only two false alarms (X-3, X-37) and loses the labelled type of X-1 (type agreement 18 to 17). T12 is the only detector that can produce the T12 type. T12 is validated by one labelled clip (X-1), which was relabelled from UNRESOLVED to T12 in the same commit that added the detector. That is a thin basis, and the decision is recorded as such.

### Fragility (thresholds that sit close together)
* X-17 (T09) scores 0.753, while the normal reference clip X-16 scores 0.77 and 0.777 on the same check.
* T07 true positives score 0.705 (X-32) and 0.688 (X-33); the false alarm X-31 scores 0.684.
* X-18 (labelled T06) is detected at 0.551 against a threshold of 0.5. The earlier threshold of 0.6 would have missed it.
* X-1 and X-2 are partly caught by T06 hits at 3.5 s and 5.0 s. Those hits look like a client approaching the desk, not an extra person in the cash zone, so the hit is right for the wrong reason.

## Pseudo-new-angle check (v8, warped clips)

Each of the 40 clips was re-rendered once with one geometric change (`runtime/aug_v8/index.csv`) and the full pipeline was run on the result (`runtime/eval_v8_aug/`). Baseline numbers in brackets are v8c on the same clips, unwarped.

| Transform | Clips | Violation clips | Recall | False alarms | Labelled type found | Timecode median |
|---|---|---|---|---|---|---|
| crop / zoom | 6 | 3 | 3/3 (3/3) | 3/3 (2/3) | 3/3 (3/3) | 1.5 s (4.0 s) |
| landscape | 6 | 3 | 3/3 (3/3) | 0/3 (0/3) | 3/3 (3/3) | 5.0 s (2.0 s) |
| mirror | 7 | 4 | 4/4 (3/4) | 1/3 (2/3) | 3/4 (3/4) | 3.0 s (1.9 s) |
| perspective | 14 | 8 | 5/8 (6/8) | 2/6 (2/6) | 3/8 (5/8) | 2.1 s (0.9 s) |
| rotation ±10° | 7 | 3 | 2/3 (3/3) | 2/4 (3/4) | 2/3 (3/3) | 4.0 s (0.9 s) |
| **all** | **40** | **21** | **17/21 = 0.81 (0.86)** | **8/19 (9/19)** | **14/21 (17/21)** | **2.1 s (1.7 s)** |

Precision 0.68, 0 partial results, 0 errors, 23.0 s mean per clip (not comparable: a second idle service shared the GPU).

* Recall drops by one clip overall. Perspective and rotation cost the most: X-11 (T02, perspective) and X-24 (T01, rotation) are lost; X-34 keeps its verdict but loses its T08 event. Mirroring gains X-13.
* Type agreement drops more than recall (17 → 14): under warps the clip is still flagged, but more often through T06/T12 than through its labelled type.
* Timecodes get noisier (median 1.7 s → 2.1 s): occupancy and track timing shift when the room geometry is distorted.
* No new failure mode appeared (no partial results, no anchor-missing warnings), which supports the per-video grounding design; but a real new camera position can differ more than these warps.

## API test (jury-style, real models, `tools/api_test.py`)

Run locally with real models on the code at dfe1d33. **Not tested through ngrok.**
* 401 without a key or with a wrong key (POST and GET).
* 415 for a `.txt` upload and for a file named `.mp4` whose content is not a video container (signature check).
* 413 for a 305 MB file, decided from Content-Length before the body is stored.
* 404 for an unknown job, and for a jury job requested through the reviewer UI (`/ui/jobs/{id}`).
* A corrupt `.mov` ends as `failed` with a clear message.
* **4 concurrent jury uploads** (X-26, X-14 = 142.5 s, X-46, X-3): all `done`, schema valid, all finished in **97.5 s** including upload time (the earlier v7 test took 133 s on the 3050).
* 4 rapid reviewer-UI uploads: 200, 200, 200, 429 (the UI queue holds at most 3 pending jobs).
* No `upload_*` files left in the work directory afterwards.

## Business trade-off
The final operating point favours recall. Against v7 it finds 5 more violation clips (13 to 18 of 21) and in exchange raises 9 false alarms on the 19 normal clips (0 to 9). A false alarm costs the reviewer a glance at the flagged timecode and frames. A miss is never seen. That asymmetry is why the operating point was chosen.

The cost is stated plainly: on this data, 9 of 19 normal clips (47%) would be sent to a reviewer for no reason. Most of those alarms come from T12 (11 clips) and T06 (7 clips), the two least reliable checks. Precision is 0.67 on the training clips, and the false-alarm rate on new footage is unknown.

Recall is limited by cue types, not only by thresholds. Lowering thresholds would add false alarms without recovering T04. Next steps to raise recall and cut false alarms: a closer or higher-resolution view of the safe lock (T04), a door-state sensor or a dedicated door model (T07), and labels for the unresolved clips. A held-out or new-angle evaluation is needed before any of these numbers are treated as performance on new footage.
