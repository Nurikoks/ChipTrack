"""Jury-style API test: concurrent uploads via curl, polling, exact schema validation, error cases.

Usage: python api_test.py <base_url> <api_key> <video> [<video> ...]
Set API_TEST_MAX_MB if the service runs with a FORTE_MAX_UPLOAD_MB other than 300.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

base, key, videos = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]


def curl(*args):
    r = subprocess.run(["curl.exe", "-s", "-w", "\n%{http_code}", *args], capture_output=True, text=True,
                       encoding="utf-8")
    body, _, code = r.stdout.rpartition("\n")
    return int(code or 0), body


def check_schema(j):
    errs = []
    if set(["job_id", "status", "verdict", "violations"]) - set(j):
        errs.append(f"missing top-level keys: {set(['job_id', 'status', 'verdict', 'violations']) - set(j)}")
    if j.get("verdict") not in ("violation", "no_violation"):
        errs.append(f"bad verdict {j.get('verdict')}")
    if j.get("verdict") == "no_violation" and j.get("violations"):
        errs.append("no_violation with non-empty violations")
    if j.get("verdict") == "violation" and not j.get("violations"):
        errs.append("violation with empty violations")
    for v in j.get("violations", []):
        for k, t in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
            if not isinstance(v.get(k), t):
                errs.append(f"violation.{k} missing/wrong type")
        if not 1 <= len(v.get("evidence_frames", [])) <= 3:
            errs.append("evidence_frames must have 1-3 items")
        for f in v.get("evidence_frames", []):
            if not isinstance(f.get("timestamp_sec"), (int, float)) or not str(f.get("image_base64", "")).startswith("/9j/"):
                errs.append("bad evidence frame (timestamp or JPEG base64)")
    return errs


def run(video):
    t0 = time.time()
    code, body = curl("-X", "POST", f"{base}/api/v1/analyze", "-H", f"X-API-Key: {key}", "-F", f"file=@{video}")
    if code != 200:
        return video, f"upload HTTP {code}: {body[:200]}", None, 0
    jid = json.loads(body)["job_id"]
    while True:
        time.sleep(3)
        code, body = curl(f"{base}/api/v1/jobs/{jid}", "-H", f"X-API-Key: {key}")
        j = json.loads(body)
        if j["status"] != "processing":
            break
        if time.time() - t0 > 330:
            return video, "TIMEOUT > 330 s", None, time.time() - t0
    dt = time.time() - t0
    if j["status"] == "failed":
        return video, f"failed: {j.get('error')}", j, dt
    errs = check_schema(j)
    summary = f"{j['verdict']} " + ", ".join(f"{v['timecode_sec']}s {v['type'][:40]}" for v in j["violations"])
    return video, ("SCHEMA ERRORS " + "; ".join(errs)) if errs else "ok  " + summary, j, dt


print("--- error cases")
tmp = Path(tempfile.gettempdir())
(tmp / "broken.mov").write_bytes(b"\x00\x00\x00 ftypqt  definitely not a full movie")
(tmp / "notes.txt").write_text("hello")
(tmp / "fake.mp4").write_text("not a video, just text with a video extension")
max_mb = int(os.environ.get("API_TEST_MAX_MB", "300"))     # the service's FORTE_MAX_UPLOAD_MB
with open(tmp / "huge.mp4", "wb") as f:
    f.write(b"\x00\x00\x00\x18ftypmp42")
    f.truncate((max_mb + 5) * 2**20)
print("health        ->", curl(f"{base}/health")[0])
print("fake .mp4     ->", curl("-X", "POST", f"{base}/api/v1/analyze", "-H", f"X-API-Key: {key}", "-F", f"file=@{tmp / 'fake.mp4'}"))
print(f"{max_mb + 5} MB file   ->", curl("-X", "POST", f"{base}/api/v1/analyze", "-H", f"X-API-Key: {key}", "-F", f"file=@{tmp / 'huge.mp4'}"))
(tmp / "huge.mp4").unlink()
print("no key        ->", curl("-X", "POST", f"{base}/api/v1/analyze", "-F", f"file=@{videos[0]}")[0])
print("wrong key     ->", curl("-X", "POST", f"{base}/api/v1/analyze", "-H", "X-API-Key: wrong", "-F", f"file=@{videos[0]}")[0])
print("wrong key GET ->", curl(f"{base}/api/v1/jobs/abc", "-H", "X-API-Key: wrong")[0])
print("txt file      ->", curl("-X", "POST", f"{base}/api/v1/analyze", "-H", f"X-API-Key: {key}", "-F", f"file=@{tmp / 'notes.txt'}"))
print("unknown job   ->", curl(f"{base}/api/v1/jobs/doesnotexist", "-H", f"X-API-Key: {key}")[0])
print("broken .mov   ->", run(str(tmp / "broken.mov"))[1])

print(f"--- {len(videos)} concurrent uploads")
t0 = time.time()
api_ids = []
with ThreadPoolExecutor(len(videos)) as ex:
    for video, status, j, dt in ex.map(run, videos):
        print(f"{Path(video).name:12s} {dt:6.1f}s  {status}{'  PARTIAL ' + str(j.get('warnings')) if j and j.get('partial') else ''}")
        if j:
            api_ids.append(j["job_id"])
print(f"all done in {time.time() - t0:.1f}s")

print("--- reviewer UI (no key)")
if api_ids:
    print("jury job via /ui/jobs ->", curl(f"{base}/ui/jobs/{api_ids[0]}")[0], "(expected 404)")
# parallel, so the limit must also count uploads still in transfer
with ThreadPoolExecutor(5) as ex:
    codes = sorted(c for c, _ in ex.map(lambda _: curl("-X", "POST", f"{base}/ui/analyze", "-F", f"file=@{videos[0]}"),
                                        range(5)))
print("5 parallel UI uploads ->", codes, "(expected 200, 200, 200, 429, 429)")
