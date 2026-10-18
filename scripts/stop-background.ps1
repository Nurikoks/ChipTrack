$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskState=Join-Path $taskRoot 'runtime/server-process.json'
if (-not (Test-Path -LiteralPath $taskState)) { Write-Host 'No background server record. Stop a foreground server with Ctrl+C.'; return }
$taskRecord=Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
$taskProcess=Get-CimInstance Win32_Process -Filter ('ProcessId='+[int]$taskRecord.pid)
if (-not $taskProcess) { Write-Host 'The recorded server has already stopped.'; return }
$taskExpected=Join-Path $taskRoot '.venv\Scripts\python.exe'
if ($taskProcess.CommandLine -notlike ('*'+$taskExpected+'*uvicorn app.main:app*')) { throw 'PID belongs to another process; no process was stopped.' }
& taskkill.exe /PID ([string]$taskRecord.pid) /T /F
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the recorded server' }
Write-Host 'ChipTRick stopped. Videos, results and logs remain saved.'
