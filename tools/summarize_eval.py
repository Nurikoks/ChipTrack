"""Markdown summary of one evaluation run (results.jsonl from tools/evaluate.py).

Usage: python summarize_eval.py <results.jsonl> [title]
"""
import json
import sys
from collections import Counter, defaultdict

import pandas as pd

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
title = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
V = lambda r: r["pred_verdict"] == "violation"
ok_rows = [r for r in rows if not r.get("error")]   # crashed clips are reported, not counted as TN/FN
tp = sum(r["label"] == "violation" and V(r) for r in ok_rows)
fn = sum(r["label"] == "violation" and not V(r) for r in ok_rows)
fp = sum(r["label"] == "no_violation" and V(r) for r in ok_rows)
tn = sum(r["label"] == "no_violation" and not V(r) for r in ok_rows)

out = [f"### {title}", "", f"Clips: {len(rows)} (violation {tp + fn}, no violation {fp + tn})", "",
       "| | predicted violation | predicted no violation |", "|---|---|---|",
       f"| **label: violation** | {tp} (TP) | {fn} (FN) |", f"| **label: no violation** | {fp} (FP) | {tn} (TN) |", "",
       f"Recall (violation) **{tp / max(1, tp + fn):.2f}** · Precision **{tp / max(1, tp + fp):.2f}** · "
       f"Accuracy {(tp + tn) / max(1, len(rows)):.2f}", ""]

errs, type_hit, n_ev, per_type = [], 0, 0, defaultdict(Counter)
for r in rows:
    for e in r["events"]:
        n_ev += 1
        per_type[e["type"]]["events"] += 1
        if not r["preds"]:
            per_type[e["type"]]["missed_clip"] += 1
            continue
        same = [p for p in r["preds"] if p["type"] == e["type"]]
        best = min(same or r["preds"], key=lambda p: abs(p["t"] - e["t"]))
        errs.append(abs(best["t"] - e["t"]))
        if same:
            type_hit += 1
            per_type[e["type"]]["type_ok"] += 1
if errs:
    s = pd.Series(errs)
    out += [f"Timecode |error| (nearest prediction, same type preferred), {len(errs)}/{n_ev} labelled events with a "
            f"prediction: mean {s.mean():.1f} s · median {s.median():.1f} s · max {s.max():.1f} s · "
            f"≤1 s {100 * (s <= 1).mean():.0f}% · ≤2 s {100 * (s <= 2).mean():.0f}% · ≤3 s {100 * (s <= 3).mean():.0f}%", ""]
out += [f"Type agreement: labelled type present among predictions for **{type_hit}/{n_ev}** events "
        f"(UNRESOLVED events can never match).", "", "| Labelled type | events | correct type predicted | clip missed |",
        "|---|---|---|---|"]
for t, c in sorted(per_type.items()):
    out.append(f"| {t} | {c['events']} | {c['type_ok']} | {c['missed_clip']} |")
pred_types = Counter(p["type"] for r in rows for p in r["preds"])
fp_types = Counter(p["type"] for r in rows if r["label"] == "no_violation" for p in r["preds"])
out += ["", "Predicted types (all clips): " + ", ".join(f"{k} {v}" for k, v in pred_types.most_common()),
        "", "Types firing on no-violation clips (false alarms): " +
        (", ".join(f"{k} {v}" for k, v in fp_types.most_common()) or "none"), ""]
secs = pd.Series([r["sec"] for r in rows])
out += [f"Processing time per clip: mean {secs.mean():.1f} s · max {secs.max():.1f} s · "
        f"partial results: {sum(bool(r.get('partial')) for r in rows)} · errors: {sum(bool(r.get('error')) for r in rows)}", ""]
warn = Counter(w for r in rows for w in (r.get("warnings") or []))
if warn:
    out += ["Warnings: " + ", ".join(f"{k} {v}" for k, v in warn.most_common()), ""]
out += ["Failures:", ""]
for r in rows:
    if r.get("error"):
        out.append(f"* ERROR {r['video_id']}: {r['error']}")
    elif (r["label"] == "violation") != V(r):
        kind = "FN" if r["label"] == "violation" else "FP"
        out.append(f"* {kind} {r['video_id']}{' (' + r['transform'] + ')' if r.get('transform') else ''}: "
                   f"label {[(e['t'], e['type']) for e in r['events']]}, predicted {[(p['t'], p['type']) for p in r['preds']]}")
print("\n".join(out))
