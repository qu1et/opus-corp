"""Загрузка параллельного корпуса: из OPUS или из локальных файлов.

Использование:

  Скачать корпус с OPUS (файлы лежат на object.pouta.csc.fi):
    python scripts/00_fetch_corpus.py --corpus Tatoeba --version v2026-07-08 --pair en-ru

  Положить свои два файла (один на входной язык, второй на выходной):
    python scripts/00_fetch_corpus.py --files my.en my.ru --name MyCorpus --src en --tgt ru

  Посмотреть, что вообще доступно для пары en-ru:
    python scripts/00_fetch_corpus.py --probe en-ru

Файлы складываются в data/raw под канонические имена
<имя>.<src>-<tgt>.<src> и <имя>.<src>-<tgt>.<tgt>; остальные скрипты
работают именно с этой схемой имён (см. common.corpus_files).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

BASE = "https://object.pouta.csc.fi/OPUS-"

KNOWN_CANDIDATES = [
    ("Tatoeba", "v2026-07-08", "moses", "zip"),
    ("EUbookshop", "v2", "moses", "zip"),
    ("Tanzil", "v1", "moses", "zip"),
    ("WikiMatrix", "v1", "moses", "zip"),
    ("OpenSubtitles", "v2024", "moses", "zip"),
    ("CCAligned", "v1", "moses", "zip"),
    ("Books", "v1", "moses", "zip"),
    ("ada83", "v1", "moses", "zip"),
]


def url_for(corpus: str, version: str, preprocessor: str, pair: str, kind: str) -> str:
    if kind == "zip":
        return f"{BASE}{corpus}/{version}/{preprocessor}/{pair}.txt.zip"
    return f"{BASE}{corpus}/{version}/{preprocessor}/{pair}.txt.gz"


def head(url: str) -> tuple[int, int]:
    """HEAD через curl: статус и размер файла (urllib на этой машине нестабилен)."""
    res = subprocess.run(
        ["curl.exe", "-s", "-I", "-L", "--max-redirs", "5", "--connect-timeout", "10",
         "--max-time", "20", url],
        capture_output=True, text=True, encoding="utf-8",
    )
    status = 0
    size = 0
    for line in (res.stdout or "").splitlines():
        low = line.lower()
        if low.startswith("http/"):
            try:
                status = int(line.split()[1])
            except (IndexError, ValueError):
                status = 0
        elif low.startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                size = 0
    return status, size


def probe(pair: str) -> None:
    """Показывает доступные корпуса и размер файла для указанной языковой пары."""
    print(f"Пара: {pair} (хост object.pouta.csc.fi)")
    any_ok = False
    for corpus, version, preproc, kind in KNOWN_CANDIDATES:
        url = url_for(corpus, version, preproc, pair, kind)
        status, size = head(url)
        if status == 200:
            any_ok = True
            print(f"  {corpus:<14} {version:<14} {preproc:<9} {size / 1e6:6.1f} МБ  {url}")
        elif status == 404:
            print(f"  {corpus:<14} {version:<14} {preproc:<9} нет такой пары (404)")
        else:
            print(f"  {corpus:<14} {version:<14} {preproc:<9} хост не ответил ({status or 'таймаут'})")
    if not any_ok:
        print("Хост OPUS сейчас не отвечает: используйте --files со своими корпусами "
              "(или повторите позже — скачивание идёт тем же curl).")


def split_columns(text: str) -> tuple[list[str], list[str]]:
    """Корпус в одну колонку с табуляцией: 'src \t tgt' на строку."""
    src, tgt = [], []
    for line in text.splitlines():
        if not line.strip():
            src.append("")
            tgt.append("")
            continue
        a, b = line.split("\t", 1)
        src.append(a.rstrip())
        tgt.append(b.rstrip())
    return src, tgt


def write_lines(lines: list[str], dest: Path) -> None:
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def extract_zip(path: Path, tmp: Path) -> list[Path]:
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    return [p for p in tmp.rglob("*") if p.is_file()]


def place_pair(files: list[Path], src: str, tgt: str, out_src: Path, out_tgt: Path) -> None:
    """Разкладывает распакованный архив/гз в канонические имена пары."""
    by_suffix = {p.suffix.lstrip("."): p for p in files if p.suffix and len(p.suffix) <= 4}
    if src in by_suffix and tgt in by_suffix:
        shutil.copyfile(by_suffix[src], out_src)
        shutil.copyfile(by_suffix[tgt], out_tgt)
        return

    # Вариант: один файл с двумя колонками (или пара .txt внутри архива)
    txt = [p for p in files if p.suffix == ".txt"]
    if len(txt) >= 2:
        big = sorted(txt, key=lambda p: p.stat().st_size, reverse=True)[:2]
        write_lines(big[0].read_text(encoding="utf-8").splitlines(), out_src)
        write_lines(big[1].read_text(encoding="utf-8").splitlines(), out_tgt)
        return
    if len(txt) == 1:
        a, b = split_columns(txt[0].read_text(encoding="utf-8"))
        write_lines(a, out_src)
        write_lines(b, out_tgt)
        return

    raise SystemExit(
        f"Не понял, как разложить архив на пару файлов: {[p.name for p in files]}. "
        "Используйте --files, чтобы указать свои файлы вручную."
    )


def download(url: str, dest: Path) -> None:
    """Скачивание через curl (на этой машине urllib к некоторым хостам не проходит)."""
    print(f"Скачиваю {url}")
    res = subprocess.run(
        ["curl.exe", "-s", "-L", "--max-redirs", "5", "--retry", "2", "--retry-connrefused",
         "--max-time", "600", "-o", dest.as_posix(), url],
        capture_output=True, text=True, encoding="utf-8",
    )
    if res.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit(f"Скачивание не удалось (curl код {res.returncode}): {url}")
    print(f"Готово: {dest} ({dest.stat().st_size/1e6:.1f} МБ)")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", help="имя корпуса на OPUS, например Tatoeba")
    ap.add_argument("--version", help="версия корпуса, например v2026-07-08")
    ap.add_argument("--preprocessor", default="moses", help="moses | multiple | xml")
    ap.add_argument("--pair", default=f"{cfg['corpus']['src']}-{cfg['corpus']['tgt']}")
    ap.add_argument("--name", help="имя корпуса для имён файлов (по умолчанию --corpus)")
    ap.add_argument("--src", default=cfg["corpus"]["src"], help="код входного языка")
    ap.add_argument("--tgt", default=cfg["corpus"]["tgt"], help="код выходного языка")
    ap.add_argument("--files", nargs=2, metavar=("SRC_FILE", "TGT_FILE"),
                    help="локальная пара файлов вместо скачивания с OPUS")
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--probe", metavar="PAIR", help="показать доступные корпуса для пары")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.probe:
        probe(args.probe)
        return

    name = args.name or args.corpus or cfg["corpus"]["name"]
    out_src = dest / f"{name}.{args.src}-{args.tgt}.{args.src}"
    out_tgt = dest / f"{name}.{args.src}-{args.tgt}.{args.tgt}"

    if args.files:
        a, b = Path(args.files[0]), Path(args.files[1])
        shutil.copyfile(a, out_src)
        shutil.copyfile(b, out_tgt)
        n_src = len(out_src.read_text(encoding="utf-8").splitlines())
        n_tgt = len(out_tgt.read_text(encoding="utf-8").splitlines())
        print(f"Загружено: {out_src.name} ({n_src} строк), {out_tgt.name} ({n_tgt} строк)")
        if n_src != n_tgt:
            print("ВНИМАНИЕ: число строк в файлах не совпадает — параллельность нарушена")
        return

    if not args.corpus or not args.version:
        raise SystemExit("Нужно --corpus и --version (или --files для своих файлов)")

    tried = []
    blob = None
    for kind in ("zip", "gz"):
        url = url_for(args.corpus, args.version, args.preprocessor, args.pair, kind)
        status, size = head(url)
        tried.append(f"{url} -> {status}")
        if status == 200:
            blob = url
            break
    if blob is None:
        raise SystemExit("Не найдено:\n  " + "\n  ".join(tried))

    raw = dest / f"{args.pair}.{blob.rsplit('.', 1)[1]}"
    download(blob, raw)
    tmp = dest / "_unpacked"
    tmp.mkdir(exist_ok=True)

    if raw.suffix == ".zip":
        files = extract_zip(raw, tmp)
    else:
        txt = tmp / f"{args.pair}.txt"
        with gzip.open(raw, "rt", encoding="utf-8") as f, txt.open("w", encoding="utf-8") as o:
            shutil.copyfileobj(f, o)
        files = [txt]

    place_pair(files, args.src, args.tgt, out_src, out_tgt)
    shutil.rmtree(tmp, ignore_errors=True)
    raw.unlink(missing_ok=True)

    n_src = len(out_src.read_text(encoding="utf-8").splitlines())
    n_tgt = len(out_tgt.read_text(encoding="utf-8").splitlines())
    print(f"Корпус: {out_src.name} ({n_src} строк), {out_tgt.name} ({n_tgt} строк)")
    if n_src != n_tgt:
        print("ВНИМАНИЕ: число строк не совпадает")


if __name__ == "__main__":
    main()
