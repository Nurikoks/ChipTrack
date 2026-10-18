param([ValidateSet('all','inference','training','versions')][string]$Stream = 'all')
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
& "$taskRoot\.venv\Scripts\python.exe" -m app.cli logs $Stream --follow
