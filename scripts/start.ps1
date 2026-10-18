$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
try {
    $taskHealth=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
    if ($taskHealth.service -eq 'ChipTRick') { Write-Host 'ChipTRick is already running: http://127.0.0.1:8000'; return }
} catch { }
$env:HF_HOME = Join-Path $taskRoot 'hf'
$env:FORTE_DETECTOR = Join-Path $taskRoot 'models\detector'
$env:FORTE_VLM = Join-Path $taskRoot 'models\vlm'
$env:PYTHONUTF8 = '1'
$env:FORTE_WORK_DIR=Join-Path $taskRoot 'runtime'
$env:CHIPTRICK_TEST_MODE='0'
$env:FORTE_DUMMY='0'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& "$taskRoot\.venv\Scripts\python.exe" -m app.cli bootstrap
& "$taskRoot\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
