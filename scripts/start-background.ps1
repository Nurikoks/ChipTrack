$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
try {
    $taskHealth=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
    if ($taskHealth.service -eq 'ChipTRick') { Write-Host 'ChipTRick is already running: http://127.0.0.1:8000'; return }
} catch { }
$env:PYTHONUTF8='1'
$env:FORTE_WORK_DIR=Join-Path $taskRoot 'runtime'
$env:CHIPTRICK_TEST_MODE='0'
$env:FORTE_DUMMY='0'
$env:HF_HOME=Join-Path $taskRoot 'hf'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:FORTE_DETECTOR=Join-Path $taskRoot 'models/detector'
$env:FORTE_VLM=Join-Path $taskRoot 'models/vlm'
$taskPython=Join-Path $taskRoot '.venv/Scripts/python.exe'
Set-Location -LiteralPath $taskRoot
& $taskPython -m app.cli bootstrap
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap failed' }
$taskRuntime=Join-Path $taskRoot 'runtime'
$taskServer=Start-Process -FilePath $taskPython -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000','--no-access-log') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRuntime 'server.stdout.log') -RedirectStandardError (Join-Path $taskRuntime 'server.stderr.log') -PassThru
@{pid=$taskServer.Id;root=$taskRoot;started=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRuntime 'server-process.json') -Encoding UTF8
Write-Host ('ChipTRick starting, PID '+$taskServer.Id+': http://127.0.0.1:8000')
