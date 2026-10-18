$ErrorActionPreference='Stop'
$chipRoot=Split-Path $PSScriptRoot -Parent
$chipRuntime=Join-Path $chipRoot 'runtime'
$chipStatePath=Join-Path $chipRuntime 'public-tunnel.json'
$chipExe=Join-Path $chipRuntime 'tools\cloudflared.exe'
if (-not (Test-Path -LiteralPath $chipExe)) { throw 'Missing runtime/tools/cloudflared.exe. Install the official Windows amd64 binary first.' }
if (Test-Path -LiteralPath $chipStatePath) {
    $chipOld=Get-Content -LiteralPath $chipStatePath -Raw | ConvertFrom-Json
    $chipRunning=Get-Process -Id ([int]$chipOld.pid) -ErrorAction SilentlyContinue
    if ($chipRunning -and $chipRunning.Path -eq $chipExe -and [Math]::Abs(($chipRunning.StartTime.ToUniversalTime()-([DateTime]$chipOld.process_started).ToUniversalTime()).TotalSeconds) -lt 2) {
        Write-Host ('Public tunnel already running: '+$chipOld.url)
        Write-Host ('Log: '+$chipOld.log)
        return
    }
}
& (Join-Path $PSScriptRoot 'start-background.ps1')
$chipLog=Join-Path $chipRuntime ('logs\tunnel-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')+'.log')
$chipOut=$chipLog+'.stdout'
$chipProcess=Start-Process -FilePath $chipExe -ArgumentList @('tunnel','--url','http://127.0.0.1:8000','--no-autoupdate','--protocol','http2','--metrics','127.0.0.1:0','--loglevel','info') -WorkingDirectory $chipRoot -WindowStyle Hidden -RedirectStandardOutput $chipOut -RedirectStandardError $chipLog -PassThru
$chipRecord=@{pid=$chipProcess.Id;process_started=$chipProcess.StartTime.ToUniversalTime().ToString('o');url=$null;log=$chipLog;root=$chipRoot}
$chipRecord | ConvertTo-Json | Set-Content -LiteralPath $chipStatePath -Encoding UTF8
Set-Content -LiteralPath (Join-Path $chipRuntime 'public-url.txt') -Value 'Starting; see public-tunnel.json and tunnel log.' -Encoding UTF8
for ($chipAttempt=0; $chipAttempt -lt 40; $chipAttempt++) {
    Start-Sleep -Seconds 1
    $chipProcess.Refresh()
    if ($chipProcess.HasExited) { throw ('Tunnel stopped; see '+$chipLog) }
    $chipText=Get-Content -LiteralPath $chipLog -Raw -ErrorAction SilentlyContinue
    if ($chipText -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
        $chipRecord.url=$Matches[0]
        $chipRecord | ConvertTo-Json | Set-Content -LiteralPath $chipStatePath -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $chipRuntime 'public-url.txt') -Value $chipRecord.url -Encoding UTF8
        Write-Host ('Public URL: '+$chipRecord.url)
        Write-Host 'Keep this computer, the server and the tunnel running. The URL changes after a new tunnel starts.'
        Write-Host ('Log: '+$chipLog)
        return
    }
}
throw ('No URL yet. Tunnel may still be connecting; inspect '+$chipLog)
