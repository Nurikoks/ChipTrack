# Evaluation (Phase 5)

## Update: types aligned with the team's Excel (v9 → v10)
Labels now use the team's verified Excel types (X-22 corrected to "key left in the safe lock").
**v9, full run on 40 clips:** TP 13 · FN 8 · FP 2 · TN 17. Recall 0.62, precision 0.87, correct type 12/22, timecode median 1.9 s, 32 s/clip.
**v10:** the two false alarms were fixed (T12 now requires a visitor actually present; T06 needs P ≥ 0.85) and verified on X-5, X-12 (now clean) and X-17, X-18 (still detected). Expected on the full set: **TP 13 · FP 0**.

| Excel type | Episodes | Found with correct type |
|---|---|---|
| T01 valuables/documents left | 3 | 3 |
| T02 laptop left on | 2 | 2 |
| T03 safe open | 1 | 1 |
| T05 cash left as the next client approaches | 2 | 2 |
| T06 two clients at once (new) | 1 | 1 |
| T10 personal items | 1 | 1 |
| T11 drinks | 1 | 1 |
| T12 documents not checked (new) | 3 | 1 (X-17) |
| T04 key in the lock | 1 | 0: the key is too small to see |
| T07 door not closed | 3 | 0: door state is ambiguous |
| T13 money mixed / T14 issued without recount / T15 not counted | 4 | 0: not detectable with this model |

The sections below describe the earlier v2–v7 runs, which used my Phase 1 type interpretation.

**Method.** Full pipeline run on all 40 unique training clips (duplicates X-47 = X-4 and X-48 = X-5 merged; X-4 carries both labelled events). Video-level verdict versus label; timecode error versus the single labelled second; type checked against the Phase 1 taxonomy.
**Important caveat:** all rules and thresholds were set by looking at these same clips, so the numbers are **in-sample and optimistic**. The dataset (21 positives across 11 types) is too small for an honest held-out split per type. A pseudo-new-angle set (40 warped clips: perspective, rotation, mirror, crop, landscape) was built (`tools/make_pseudo_angles.py`), but its full evaluation was not completed in the time available.

## Progress across versions (40 clips)

| Version | TP | FN | FP | TN | Recall | Precision | Correct type | Timecode median | Time per clip |
|---|---|---|---|---|---|---|---|---|---|
| v2 (first full run) | 16 | 5 | 6 | 13 | 0.76 | 0.73 | 10/22 | 4.5 s | 45 s |
| v3 (tracker, room filter) | 13 | 8 | 9 | 10 | 0.62 | 0.59 | 10/22 | 1.9 s | 28.5 s |
| v6 (occupancy, safe duration, door-based entries) | 14 | 7 | 6 | 13 | 0.67 | 0.70 | 10/22 | 2.6 s | 34 s |
| **v7 (final)** | **13** | **8** | **0** | **19** | **0.62** | **1.00** | **12/22** | **2.0 s** | **26.8 s** |

v2's higher recall came largely from wrong-type detections triggered by person-count noise (people seen through the glass), so it was not a usable operating point.

## Final version (v7) in detail

Clips: 40 (violation 21, no violation 19)

| | predicted violation | predicted no violation |
|---|---|---|
| **label: violation** | 13 (TP) | 8 (FN) |
| **label: no violation** | 0 (FP) | 19 (TN) |

Recall (violation) **0.62** · Precision **1.00** · Accuracy 0.80

Timecode |error| (nearest prediction, same type preferred), 14/22 labelled events with a prediction: mean 3.6 s · median 2.0 s · max 14.9 s · ≤1 s 29% · ≤2 s 57% · ≤3 s 57%

Type agreement: labelled type present among predictions for **12/22** events (UNRESOLVED events can never match).

| Labelled type | events | correct type predicted | clip missed |
|---|---|---|---|
| T01_UNATTENDED_VALUABLES | 3 | 3 | 0 |
| T02_PC_UNLOCKED_UNATTENDED | 2 | 2 | 0 |
| T03_SAFE_LEFT_OPEN | 1 | 1 | 0 |
| T04_KEY_LEFT_IN_SAFE | 2 | 0 | 2 |
| T05_CASH_EXPOSED_NEXT_CLIENT | 3 | 3 | 0 |
| T06_UNAUTHORIZED_PERSON_CASH_ZONE | 2 | 0 | 1 |
| T07_CASH_ROOM_DOOR_NOT_CLOSED | 2 | 0 | 2 |
| T08_COUNT_WITHOUT_CLIENT | 1 | 1 | 0 |
| T09_COLLECTOR_NO_DOC_CHECK | 1 | 0 | 1 |
| T10_PERSONAL_ITEMS_AT_DESK | 1 | 1 | 0 |
| T11_FOOD_DRINK_AT_DESK | 1 | 1 | 0 |
| UNRESOLVED | 3 | 0 | 2 |

Predicted types (all clips): T01_UNATTENDED_VALUABLES 3, T11_FOOD_DRINK_AT_DESK 2, T05_CASH_EXPOSED_NEXT_CLIENT 2, T02_PC_UNLOCKED_UNATTENDED 2, T03_SAFE_LEFT_OPEN 2, T08_COUNT_WITHOUT_CLIENT 2, T07_CASH_ROOM_DOOR_NOT_CLOSED 1, T10_PERSONAL_ITEMS_AT_DESK 1

Types firing on no-violation clips (false alarms): none

Processing time per clip: mean 26.8 s · max 68.7 s · partial results: 0 · errors: 0

Failures:

* FN X-2: label [(10.0, 'UNRESOLVED')], predicted []
* FN X-13: label [(47.0, 'T04_KEY_LEFT_IN_SAFE')], predicted []
* FN X-17: label [(7.0, 'T09_COLLECTOR_NO_DOC_CHECK')], predicted []
* FN X-18: label [(20.0, 'T06_UNAUTHORIZED_PERSON_CASH_ZONE')], predicted []
* FN X-22: label [(14.0, 'T04_KEY_LEFT_IN_SAFE')], predicted []
* FN X-32: label [(5.0, 'T07_CASH_ROOM_DOOR_NOT_CLOSED')], predicted []
* FN X-33: label [(6.0, 'T07_CASH_ROOM_DOOR_NOT_CLOSED')], predicted []
* FN X-44: label [(16.0, 'UNRESOLVED')], predicted []

### Why the 8 misses happen
* **X-2, X-44 (UNRESOLVED):** no visible violating element was identified in Phase 1, so no rule can target them.
* **X-13, X-22 (T04 key left in the safe lock):** the key is 3–5 px; both Qwen3-VL-2B and 4B fail even on upscaled crops.
* **X-32, X-33 (T07 door not closed):** the glass-door state is ambiguous to the VLM, and a motion cue (not turning back to the door) gave no gain.
* **X-17 (T09):** the collector check is disabled by default because it could not be validated (it caused a false positive on X-16).
* **X-18 (T06):** the entry of the extra person is not separated from the client's.

## API test (jury-style, real models, `tools/api_test.py`)
* 401 without a key or with a wrong key (POST and GET); 415 for a .txt upload; 404 for an unknown job; a corrupt .mov ends as `failed` with a clear message.
* **4 concurrent uploads** (X-26, X-14 = 142.5 s, X-46, X-3): all `done`, schema valid, all finished in **133 s** (well under 5 min), verdicts matched the labels.

## Business trade-off
The final operating point favours precision: **no false alarms on the 19 normal clips**. Recall is limited by the four cue types above, not by thresholds. Lowering thresholds would add false alarms without recovering those types. The next steps to raise recall are a closer or higher-resolution view of the safe lock (T04), a door-state sensor or a dedicated door model (T07), and labels for the unresolved clips.
