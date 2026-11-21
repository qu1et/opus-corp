# Обёртка над scripts/run_all.py: все аргументы передаются как есть.
# Примеры:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -StageArgs '--stages','1','2'
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -StageArgs '--raw-src','my.en','--raw-tgt','my.ru','--name','MyCorpus'

param(
    [string[]]$StageArgs = @()
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).ProviderPath }

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py (Join-Path $root "scripts\run_all.py") @StageArgs
