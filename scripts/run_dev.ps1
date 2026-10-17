# Dev launcher. Usage: powershell -File scripts/run_dev.ps1 [-Dummy] [-Port 8000]
param([switch]$Dummy, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "E:\fortebank_env\hf" }
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
if (-not $env:FORTE_API_KEY) {
    $keyFile = Join-Path $root "runtime\api_key.txt"
    if (Test-Path $keyFile) { $env:FORTE_API_KEY = (Get-Content $keyFile -Raw).Trim() } else { $env:FORTE_API_KEY = "dev-key" }
}
$env:FORTE_DUMMY = if ($Dummy) { "1" } else { "0" }
$py = if ($env:FORTE_PYTHON) { $env:FORTE_PYTHON } else { "E:\fortebank_env\Scripts\python.exe" }
& $py -m uvicorn app.main:app --host 0.0.0.0 --port $Port --workers 1
