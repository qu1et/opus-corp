# Setup: venv + pip deps + CUDA torch + (опционально) сборка fast_align.
# Запуск:
#   powershell -ExecutionPolicy Bypass -File setup.ps1            # только Python-часть
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Tools     # плюс MSYS2/MinGW и fast_align
# Дальнейшие запуски скриптов используют .venv (run_all.py сам его находит).
# Файл сохранён в UTF-8 С BOM: PowerShell 5.1 без BOM читает его как ANSI и кириллица
# в строках превращается в кашу (плюс ломается разбор строк со скобками).

param(
    [switch]$Tools,
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

# --- Консоль в UTF-8 -------------------------------------------------------
try {
    $enc = New-Object System.Text.UTF8Encoding($false)
    $OutputEncoding = $enc
    [Console]::InputEncoding = $enc
    [Console]::OutputEncoding = $enc
    & chcp.com 65001 > $null
} catch {
    Write-Warning 'не удалось переключить консоль в UTF-8 (только косметика)'
}

# --- Вызов внешних команд --------------------------------------------------
# Windows PowerShell 5.1 при $ErrorActionPreference=Stop превращает любую строку stderr
# внешней программы в terminating-ошибку (падало на предупреждении torch). Функция гасит
# это и превращает вывод (stdout+stderr) в обычные строки.
function Invoke-Native {
    param([scriptblock]$Block)
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Block 2>&1 | ForEach-Object { "$_" } }
    finally { $ErrorActionPreference = $eap }
}

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).ProviderPath }

$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Создаю виртуальное окружение $venv"
    Invoke-Native { python -m venv $venv } | Out-Null
}

Write-Host "Обновляю pip"
Invoke-Native { & $py -m pip install --upgrade pip } | Select-Object -Last 3

if (-not $SkipTorch) {
    Write-Host "Ставлю torch с CUDA (cu128; для RTX 50xx подходит)"
    Invoke-Native { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128 } | Select-Object -Last 3
}

Write-Host "Ставлю остальные зависимости из requirements.txt"
Invoke-Native { & $py -m pip install -r (Join-Path $root "requirements.txt") } | Select-Object -Last 3

Write-Host "Проверяю окружение"
# torch пишет в stderr служебное предупреждение elastic/redirects (безвредное) — отсекаем его.
$env:TORCH_CPP_LOG_LEVEL = "ERROR"
Invoke-Native { & $py -c "import torch, sentencepiece, ftlangdetect, sentence_transformers; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); print('sentencepiece', sentencepiece.__version__)" } |
    Where-Object { $_ -notmatch 'Redirects are currently not supported' }

Write-Host "Pre-fetch: модель определения языка lid.176 (для задачи 1)"
New-Item -ItemType Directory -Force -Path (Join-Path $root ".cache") | Out-Null
$bin = Join-Path $root ".cache\lid.176.bin"
if (Test-Path $bin) {
    Write-Host ("  уже есть: {0} МБ" -f [math]::Round((Get-Item $bin).Length / 1MB, 1))
} else {
    foreach ($u in @(
        "https://huggingface.co/arc-r/fasttext-language-identification/resolve/main/lid.176.bin",
        "https://huggingface.co/crash-sv/scribe-fasttext-lid176/resolve/main/lid.176.bin",
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    )) {
        Invoke-Native { & curl.exe -s -L --retry 2 --max-time 150 -o $bin $u } | Out-Null
        if ((Test-Path $bin) -and (Get-Item $bin).Length -gt 1000000) {
            Write-Host ("  lid.176.bin: {0} МБ ({1})" -f [math]::Round((Get-Item $bin).Length / 1MB, 1), $u)
            break
        } else {
            Remove-Item $bin -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-Path $bin)) { Write-Host '  модель не скачалась: этап языка можно пропустить (--no-lang)' }
}

if ($Tools) {
    $bash = "C:\msys64\usr\bin\bash.exe"
    if (-not (Test-Path $bash)) {
        Write-Host 'MSYS2 не найден: ставлю его через winget'
        Invoke-Native { winget install --id MSYS2.MSYS2 -e --accept-package-agreements --accept-source-agreements } | Select-Object -Last 4
    }
    if (Test-Path $bash) {
        Write-Host "Ставлю MinGW-компилятор, cmake и make"
        Invoke-Native { & $bash -lc 'pacman -S --noconfirm --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make make 2>&1 | tail -5' }

        Write-Host "Собираю fast_align и atools"
        # Пути берём из расположения setup.ps1, в нотации MSYS2 (D:\x -> /d/x).
        $sh = ($root -replace '\\', '/') + '/scripts/build_fast_align.sh'
        $dest = ($root -replace '\\', '/') + '/tools/fast_align'
        Invoke-Native { & $bash -lc "bash $sh $dest 2>&1 | tail -20" }

        $exe = Join-Path $root "tools\fast_align\build\fast_align.exe"
        if (Test-Path $exe) { Write-Host "готов: $exe" } else { Write-Host "бинарные файлы не появились: $exe" }
    } else {
        Write-Host 'MSYS2 недоступен: собери fast_align вручную (scripts/build_fast_align.sh) на Linux/WSL'
    }
}

Write-Host ""
Write-Host 'Дальше: powershell -ExecutionPolicy Bypass -File run_all.ps1   (или python scripts/run_all.py)'
Write-Host ('Для гейтованной модели эмбеддингов нужен токен HF: & ' + $py + ' -m huggingface_hub.cli.login   (или HF_TOKEN)')
Write-Host 'Лицензии: https://huggingface.co/google/embeddinggemma-300m и https://huggingface.co/Unbabel/wmt22-cometkiwi-da'
