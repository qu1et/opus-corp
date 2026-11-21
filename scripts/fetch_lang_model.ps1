# Качает модель lid.176 (.ftz или .bin) кусками: некоторые хосты отдают файл очень медленно,
# параллельные куски сокращают общее время. Результат складывается в .cache\<имя>.
param(
    [string]$Name = "lid.176.ftz",
    [string]$Url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
    [int]$Parts = 6,
    [int]$TimeoutSec = 600
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).ProviderPath }
$cache = Join-Path $root ".cache"
New-Item -ItemType Directory -Force -Path $cache | Out-Null

$total = 0
$h = curl.exe -s -I --max-time 60 $Url
foreach ($line in $h) { if ($line -match '^content-length:\s*(\d+)') { $total = [int]$Matches[1] } }
if (-not $total) { Write-Error "Не удалось получить размер файла: $Url"; exit 1 }

$chunk = [math]::Ceiling($total / $Parts)
$jobs = @()
for ($i = 0; $i -lt $Parts; $i++) {
    $from = $i * $chunk
    $to = [math]::Min($total - 1, $from + $chunk - 1)
    if ($from -gt $to) { continue }
    $part = Join-Path $cache "$Name.part$i"
    $jobs += Start-Process -PassThru -WindowStyle Hidden curl.exe -ArgumentList @(
        "-s", "-L", "--max-time", "$TimeoutSec", "-r", "$from-$to", "-o", $part
    )
}
foreach ($p in $jobs) { $p.WaitForExit() }

$final = Join-Path $cache $Name
$stream = [System.IO.File]::Open($final, [System.IO.FileMode]::Create)
try {
    for ($i = 0; $i -lt $Parts; $i++) {
        $part = Join-Path $cache "$Name.part$i"
        if (Test-Path $part) {
            $bytes = [System.IO.File]::ReadAllBytes($part)
            $stream.Write($bytes, 0, $bytes.Length)
            Remove-Item $part -Force
        }
    }
} finally { $stream.Close() }

$size = (Get-Item $final).Length
Write-Host "$Name: $size из $total байт"
if ($size -ne $total) { Write-Error "Файл неполный ($size из $total); повторите скрипт или скачайте другим зеркалом"; exit 1 }
