$taskRoot = Split-Path $PSScriptRoot -Parent
$taskLogs = Join-Path $PSScriptRoot 'logs.ps1'
if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
    & wt.exe -w new new-tab --title 'ChipTRick | Analysis' powershell -NoExit -NoProfile -ExecutionPolicy Bypass -File $taskLogs -Stream inference ';' new-tab --title 'ChipTRick | Training' powershell -NoExit -NoProfile -ExecutionPolicy Bypass -File $taskLogs -Stream training ';' new-tab --title 'ChipTRick | Versions' powershell -NoExit -NoProfile -ExecutionPolicy Bypass -File $taskLogs -Stream versions
} else {
    Write-Host 'Open Logs-Inference.cmd, Logs-Training.cmd and Logs-Versions.cmd in separate terminal tabs.'
}
