$ErrorActionPreference='Stop'
$chipRoot=Split-Path $PSScriptRoot -Parent
$chipStatePath=Join-Path $chipRoot 'runtime\public-tunnel.json'
if (-not (Test-Path -LiteralPath $chipStatePath)) { Write-Host 'No tunnel record.'; return }
$chipRecord=Get-Content -LiteralPath $chipStatePath -Raw | ConvertFrom-Json
$chipProcess=Get-Process -Id ([int]$chipRecord.pid) -ErrorAction SilentlyContinue
if (-not $chipProcess) { Write-Host 'Tunnel already stopped.'; return }
$chipExpected=Join-Path $chipRoot 'runtime\tools\cloudflared.exe'
if ($chipProcess.Path -ne $chipExpected -or [Math]::Abs(($chipProcess.StartTime.ToUniversalTime()-([DateTime]$chipRecord.process_started).ToUniversalTime()).TotalSeconds) -ge 2) { throw 'PID belongs to a different process. Nothing stopped.' }
& taskkill.exe /PID ([string]$chipRecord.pid) /F
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the recorded tunnel.' }
$chipRecord | Add-Member -NotePropertyName stopped -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
$chipRecord | ConvertTo-Json | Set-Content -LiteralPath $chipStatePath -Encoding UTF8
Set-Content -LiteralPath (Join-Path $chipRoot 'runtime\public-url.txt') -Value 'Stopped. Run Start-Public.cmd to obtain a new URL.' -Encoding UTF8
Write-Host 'Public access stopped. The local server is still running.'
