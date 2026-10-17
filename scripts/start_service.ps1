# Production launcher (Windows, native venv).
#   powershell -ExecutionPolicy Bypass -File scripts/start_service.ps1 [-Port 8000] [-Tunnel] [-BindHost 127.0.0.1]
# - generates a random API key once (runtime/api_key.txt) and prints it
# - runs fully offline (HF_HUB_OFFLINE=1): weights must be downloaded first (scripts/download_weights.py)
# - -Tunnel additionally starts a Cloudflare quick tunnel (public https URL) to the local port;
#   it only forwards HTTP traffic, all video processing stays on this machine.
param([int]$Port = 8000, [switch]$Tunnel, [string]$BindHost = "127.0.0.1")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force "$root\runtime" | Out-Null

$keyFile = "$root\runtime\api_key.txt"
if (-not (Test-Path $keyFile)) {
    $bytes = New-Object byte[] 24
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = ([Convert]::ToBase64String($bytes)).Replace("+", "A").Replace("/", "B").TrimEnd("=")
    Set-Content -Path $keyFile -Value $key -Encoding ascii -NoNewline
    Write-Host "Generated new API key -> $keyFile"
}
$env:FORTE_API_KEY = (Get-Content $keyFile -Raw).Trim()
Write-Host "X-API-Key: $($env:FORTE_API_KEY)"

if (-not $env:HF_HOME) { $env:HF_HOME = "E:\fortebank_env\hf" }
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:FORTE_DUMMY = "0"
$py = if ($env:FORTE_PYTHON) { $env:FORTE_PYTHON } else { "E:\fortebank_env\Scripts\python.exe" }

if ($Tunnel) {
    $cf = if ($env:CLOUDFLARED) { $env:CLOUDFLARED } else { "E:\fortebank_env\bin\cloudflared.exe" }
    if (-not (Test-Path $cf)) { throw "cloudflared not found at $cf (see README: Public URL)" }
    Start-Process -FilePath $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://localhost:$Port" `
        -RedirectStandardError "$root\runtime\tunnel.log" -NoNewWindow
    Write-Host "Tunnel starting; public URL will appear in runtime\tunnel.log (https://*.trycloudflare.com)"
}
# localhost only: the tunnel connects locally; -BindHost 0.0.0.0 exposes the service to the whole LAN
& $py -m uvicorn app.main:app --host $BindHost --port $Port --workers 1
