# Setup: venv + pip deps + CUDA torch + (опционально) сборка fast_align.
# Запуск:
#   powershell -ExecutionPolicy Bypass -File setup.ps1            # только Python-часть
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Tools     # плюс MSYS2/MinGW и fast_align
# Дальнейшие запуски скриптов используют .venv (run_all.py сам его находит).

param(
    [switch]$Tools,
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).ProviderPath }

$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Создаю виртуальное окружение $venv"
    python -m venv $venv
}

Write-Host "Обновляю pip"
& $py -m pip install --upgrade pip 2>&1 | Select-Object -Last 3

if (-not $SkipTorch) {
    Write-Host "Ставлю torch с CUDA (cu128; для RTX 50xx подходит)"
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | Select-Object -Last 3
}

Write-Host "Ставлю остальные зависимости из requirements.txt"
& $py -m pip install -r (Join-Path $root "requirements.txt") 2>&1 | Select-Object -Last 3

& $py -c "import torch, sentencepiece, ftlangdetect, sentence_transformers; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); print('sentencepiece', sentencepiece.__version__)"

Write-Host "Pre-fetch: модель определения языка lid.176 (для задачи 1)"
New-Item -ItemType Directory -Force -Path (Join-Path $root ".cache") | Out-Null
$bin = Join-Path $root ".cache\lid.176.bin"
if (-not (Test-Path $bin)) {
    foreach ($u in @(
        "https://huggingface.co/arc-r/fasttext-language-identification/resolve/main/lid.176.bin",
        "https://huggingface.co/crash-sv/scribe-fasttext-lid176/resolve/main/lid.176.bin",
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    )) {
        curl.exe -s -L --retry 2 --max-time 150 -o $bin $u
        if ((Test-Path $bin) -and (Get-Item $bin).Length -gt 1000000) {
            Write-Host "  lid.176.bin: $([math]::Round((Get-Item $bin).Length / 1MB, 1)) МБ ($u)"
            break
        } else {
            Remove-Item $bin -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-Path $bin)) { Write-Host "  модель не скачалась: этап языка можно пропустить (--no-lang)" }
}

if ($Tools) {
    $bash = "C:\msys64\usr\bin\bash.exe"
    if (-not (Test-Path $bash)) {
        Write-Host "Ставлю MSYS2 через winget"
        winget install --id MSYS2.MSYS2 -e --accept-package-agreements --accept-source-agreements 2>&1 | Select-Object -Last 4
    }
    Write-Host "Ставлю MinGW-компилятор, cmake и make"
    & $bash -lc "pacman -S --noconfirm --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make make 2>&1 | tail -5"

    Write-Host "Собираю fast_align и atools"
    & $bash -lc "bash /d/practice/scripts/build_fast_align.sh 2>&1 | tail -20"

    $exe = Join-Path $root "tools\fast_align\build\fast_align.exe"
    if (Test-Path $exe) { Write-Host "готов: $exe" } else { Write-Host "бинарь не появился: $exe" }
}

Write-Host ""
Write-Host "Дальше: powershell -ExecutionPolicy Bypass -File run_all.ps1  (или python scripts/run_all.py)"
Write-Host "Для гейтованной модели эмбеддингов нужен токен HF: & $py -m huggingface_hub.cli.login (или HF_TOKEN)"
Write-Host "Лицензии: https://huggingface.co/google/embeddinggemma-300m и https://huggingface.co/Unbabel/wmt22-cometkiwi-da"
