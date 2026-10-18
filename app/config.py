"""Service configuration (environment variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORK_DIR = Path(os.environ.get("FORTE_WORK_DIR", ROOT / "runtime"))
_api_key_file = WORK_DIR / 'api-key.txt'
API_KEY = os.environ.get("FORTE_API_KEY", _api_key_file.read_text(encoding='utf-8').strip() if _api_key_file.exists() else '')
MAX_UPLOAD_MB = int(os.environ.get("FORTE_MAX_UPLOAD_MB", "300"))
# Analysis budget from worker start. Queue waiting is reported separately.
JOB_DEADLINE_SEC = float(os.environ.get("FORTE_JOB_DEADLINE_SEC", "270"))
# Legacy setting; persistent jobs and owned media are retained in SQLite/storage.
RESULT_TTL_SEC = float(os.environ.get("FORTE_RESULT_TTL_HOURS", "6")) * 3600
RULES_PATH = Path(os.environ.get("FORTE_RULES", ROOT / "data" / "rules_registry.json"))

DETECTOR_ID = os.environ.get("FORTE_DETECTOR", str(ROOT / 'models' / 'detector') if (ROOT / 'models' / 'detector' / 'chiptrick-source.json').exists() else "PekingU/rtdetr_v2_r18vd")
VLM_ID = os.environ.get("FORTE_VLM", str(ROOT / 'models' / 'vlm') if (ROOT / 'models' / 'vlm' / 'chiptrick-source.json').exists() else "Qwen/Qwen3-VL-2B-Instruct")
VLM_4BIT = os.environ.get("FORTE_VLM_4BIT", "1") == "1"
POSE_PATH=Path(os.environ.get('FORTE_POSE',ROOT/'models/pose/pose_landmarker_full.task'))
SAMPLE_FPS = float(os.environ.get("FORTE_SAMPLE_FPS", "2"))
# Set FORTE_DUMMY=1 to run the API/UI without loading models (UI development, CI).
DUMMY = os.environ.get("FORTE_DUMMY", "0") == "1"
# Adds a "debug" block (anchors, timeline, every VLM answer) to results. Off for the jury.
DEBUG = os.environ.get("FORTE_DEBUG", "0") == "1"
# Enables checks that could not be validated on the training data (currently: T09 collector document check).
EXPERIMENTAL = os.environ.get("FORTE_EXPERIMENTAL", "0") == "1"

ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
