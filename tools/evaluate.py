"""Offline evaluation of the full pipeline on labelled clips.

Usage: python evaluate.py <out_dir> [video_id ...]
Writes <out_dir>/results.jsonl (one line per clip, without base64) and prints metrics.
Duplicate files (X-47 = X-4, X-48 = X-5) are analysed once; X-4/X-47 labels are both kept as
reference events of that clip.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.pipeline.analyze import Analyzer  # noqa: E402

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
args = sys.argv[2:]
aug_dir = None
if args and args[0] == "--aug":          # evaluate the pseudo-new-angle set instead
    aug_dir, args = Path(args[1]), args[2:]
only = set(args)
video_dir = aug_dir or (ROOT / "videos")

clips = {}
if aug_dir:
    for r in pd.read_csv(aug_dir / "index.csv").itertuples():
        c = clips.setdefault(r.video_id, {"video_id": r.video_id, "file": r.file, "label": r.label,
                                          "events": [], "transform": r.transform})
        if r.label == "violation":
            c["events"].append({"t": float(r.timestamp_sec), "type": r.type_candidate})
else:
    # one entry per unique file, with all reference events
    lab = pd.read_csv(ROOT / "data" / "labels_enriched.csv")
    for r in lab.itertuples():
        key = r.duplicate_of if isinstance(r.duplicate_of, str) and r.duplicate_of in clips else r.video_id
        c = clips.setdefault(key, {"video_id": key, "file": r.file, "label": r.label, "events": []})
        if r.label == "violation":
            c["label"] = "violation"
            c["events"].append({"t": float(r.timestamp_sec), "type": r.type_candidate})
if only:
    clips = {k: v for k, v in clips.items() if k in only}
if not clips:
    sys.exit(f"no clips to evaluate (unknown ids: {sorted(only)})")

an = Analyzer()
rows = []
with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fo:
    for k, c in clips.items():
        job = SimpleNamespace(deadline=time.time() + config.JOB_DEADLINE_SEC, stage="", progress=0.0)
        t0 = time.time()
        try:
            res = an.analyze(video_dir / c["file"], job)
            err = None
        except Exception as e:
            res, err = {"verdict": "error", "violations": []}, repr(e)
        dt = time.time() - t0
        preds = [{"t": v["timecode_sec"], "type": v["type_id"], "conf": v["confidence"],
                  "desc": v["description"][:160]} for v in res["violations"]]
        row = {**c, "pred_verdict": res["verdict"], "preds": preds, "sec": round(dt, 1),
               "partial": res.get("partial"), "warnings": res.get("warnings"), "error": err,
               "debug": res.get("debug")}
        rows.append(row)
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"{k:5s} gt={c['label']:12s} pred={res['verdict']:12s} {dt:5.1f}s  "
              f"gt_ev={[(e['t'], e['type'][:12]) for e in c['events']]}  pred={[(p['t'], p['type'][:12], p['conf']) for p in preds]}")

# ---- metrics ----
# a crashed clip is neither a hit nor a correct "no violation": count it separately
errors = [r for r in rows if r["error"]]
ok_rows = [r for r in rows if not r["error"]]
tp = sum(r["label"] == "violation" and r["pred_verdict"] == "violation" for r in ok_rows)
fn = sum(r["label"] == "violation" and r["pred_verdict"] != "violation" for r in ok_rows)
fp = sum(r["label"] == "no_violation" and r["pred_verdict"] == "violation" for r in ok_rows)
tn = sum(r["label"] == "no_violation" and r["pred_verdict"] != "violation" for r in ok_rows)
print(f"\nVideo-level: TP={tp} FN={fn} FP={fp} TN={tn} | recall={tp / max(1, tp + fn):.2f} "
      f"precision={tp / max(1, tp + fp):.2f} | errors={len(errors)} partial={sum(bool(r['partial']) for r in rows)}")
for r in errors:
    print(f"  ERROR {r['video_id']}: {r['error']}")
errs, type_hits, n_ev = [], 0, 0
for r in rows:
    for e in r["events"]:
        n_ev += 1
        if not r["preds"]:
            continue
        best = min(r["preds"], key=lambda p: abs(p["t"] - e["t"]))
        errs.append(abs(best["t"] - e["t"]))
        type_hits += any(p["type"] == e["type"] for p in r["preds"])
if errs:
    s = pd.Series(errs)
    print(f"Timecode |err| over {len(errs)}/{n_ev} events: mean={s.mean():.1f} median={s.median():.1f} max={s.max():.1f} "
          f"<=1s {100 * (s <= 1).mean():.0f}%  <=2s {100 * (s <= 2).mean():.0f}%  <=3s {100 * (s <= 3).mean():.0f}%")
print(f"Type agreement (gt type among predictions): {type_hits}/{n_ev}")
print(f"Processing time: mean={pd.Series([r['sec'] for r in rows]).mean():.1f}s max={max(r['sec'] for r in rows):.1f}s")
