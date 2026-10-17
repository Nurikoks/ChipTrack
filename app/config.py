"""Service configuration (environment variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# stripped: `set FORTE_API_KEY=abc && ...` on Windows keeps the trailing space
API_KEY = os.environ.get("FORTE_API_KEY", "").strip()
WORK_DIR = Path(os.environ.get("FORTE_WORK_DIR", ROOT / "runtime"))
MAX_UPLOAD_MB = int(os.environ.get("FORTE_MAX_UPLOAD_MB", "300"))
# Hard wall-clock limit per job, counted from the start of the upload request (jury limit is 5 min).
JOB_DEADLINE_SEC = float(os.environ.get("FORTE_JOB_DEADLINE_SEC", "270"))
# Finished job results are kept in memory this long, then dropped (no video is ever kept).
RESULT_TTL_SEC = float(os.environ.get("FORTE_RESULT_TTL_HOURS", "6")) * 3600
RULES_PATH = Path(os.environ.get("FORTE_RULES", ROOT / "data" / "rules_registry.json"))

DETECTOR_ID = os.environ.get("FORTE_DETECTOR", "PekingU/rtdetr_v2_r18vd")
VLM_ID = os.environ.get("FORTE_VLM", "Qwen/Qwen3-VL-2B-Instruct")
VLM_4BIT = os.environ.get("FORTE_VLM_4BIT", "1") == "1"
SAMPLE_FPS = float(os.environ.get("FORTE_SAMPLE_FPS", "2"))
# Set FORTE_DUMMY=1 to run the API/UI without loading models (UI development, CI).
DUMMY = os.environ.get("FORTE_DUMMY", "0") == "1"
# Adds a "debug" block (anchors, timeline, every VLM answer) to results. Off for the jury.
DEBUG = os.environ.get("FORTE_DEBUG", "0") == "1"
# Checks that could not be fully validated on the training data (currently: T09 collector document check).
# On by default: a missed violation is worse than a false alarm the reviewer dismisses.
EXPERIMENTAL = os.environ.get("FORTE_EXPERIMENTAL", "1") == "1"

ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
