# Dev launcher. Usage: powershell -File scripts/run_dev.ps1 [-Dummy] [-Port 8000] [-BindHost 127.0.0.1]
param([switch]$Dummy, [int]$Port = 8000, [string]$BindHost = "127.0.0.1")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "E:\fortebank_env\hf" }
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
if (-not $env:FORTE_API_KEY) {
    $keyFile = Join-Path $root "runtime\api_key.txt"
    if (-not (Test-Path $keyFile)) { throw "no API key: set FORTE_API_KEY or run scripts\start_service.ps1 once to generate runtime\api_key.txt" }
    $env:FORTE_API_KEY = (Get-Content $keyFile -Raw).Trim()
}
$env:FORTE_DUMMY = if ($Dummy) { "1" } else { "0" }
$py = if ($env:FORTE_PYTHON) { $env:FORTE_PYTHON } else { "E:\fortebank_env\Scripts\python.exe" }
& $py -m uvicorn app.main:app --host $BindHost --port $Port --workers 1
