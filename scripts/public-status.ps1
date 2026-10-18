$ErrorActionPreference='Stop'
$chipRoot=Split-Path $PSScriptRoot -Parent
$chipStatePath=Join-Path $chipRoot 'runtime\public-tunnel.json'
if (-not (Test-Path -LiteralPath $chipStatePath)) { Write-Host 'No public tunnel has been started.'; return }
$chipRecord=Get-Content -LiteralPath $chipStatePath -Raw | ConvertFrom-Json
Write-Host ('URL: '+$chipRecord.url)
Write-Host ('Log: '+$chipRecord.log)
try { Invoke-RestMethod -Uri ($chipRecord.url+'/health') -TimeoutSec 15 | ConvertTo-Json } catch { Write-Host ('Public endpoint unavailable: '+$_.Exception.Message) }
