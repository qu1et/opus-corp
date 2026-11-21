"""Задача 4: файл выравнивания слов через fast_align.

Шаги ТЗ:
  ./fast_align -i text.en-ru -d -o -v > forward.align
  ./fast_align -i text.en-ru -d -o -v -r > reverse.align
  ./atools -i forward.align -j reverse.align -c grow-diag-final-and > align

Формат входа обязателен: на строке «вход ||| выход» (см. README fast_align),
причём строки должны быть токенизованы пробелами. Скрипт сам делает токинизацию
из прочищенного корпуса (по умолчанию word-level; --pieces берёт уже разбитые
SentencePiece-файлы, где куски разделены пробелом).

Выход:
  align/<corpus>.<src>-<tgt>.txt          — «вход ||| выход»
  align/forward.align, align/reverse.align
  align/<corpus>.<src>-<tgt>.align        — итоговое grow-diag-final-and
  logs/04_align_summary.txt               — команды, число пар, распределение типов выравниваний

Бинари fast_align собираются setup.ps1 (-Tools) через MSYS2/MinGW; если их нет,
скрипт подскажет, как собрать.

Запуск:
  python scripts/04_align.py                       # вход — data/scored, word-level
  python scripts/04_align.py --input-dir spm_data --pieces   # вход — куски SPM
  python scripts/04_align.py --fast-align-dir tools/fast_align/build
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT,
    corpus_files,
    load_config,
    path,
    read_pair,
)


def find_binary(name: str, hint_dir: Path) -> Path | None:
    """Ищет executable в указанном каталоге или рядом с ним."""
    for cand in (
        hint_dir / name,
        hint_dir / f"{name}.exe",
        hint_dir / "build" / name,
        hint_dir / "build" / f"{name}.exe",
        ROOT / "tools" / "bin" / name,
        ROOT / "tools" / "bin" / f"{name}.exe",
    ):
        if cand.exists():
            return cand
    return None


def run(cmd: list[str], out_file: Path) -> None:
    """Запускает команду и пишет её stdout в файл."""
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"Команда завершилась с кодом {res.returncode}: {res.stderr.strip()}")
    out_file.write_text(res.stdout, encoding="utf-8", newline="\n")


def main() -> None:
    cfg = load_config()
    a_cfg = cfg["align"]
    ap = argparse.ArgumentParser(description="выравнивание слов (задача 4)")
    ap.add_argument("--input-dir", default="scored", help="каталог с парой прочищенных файлов")
    ap.add_argument("--pieces", action="store_true",
                    help="вход — разбитые SPM файлы (.piece), куски разделены пробелом")
    ap.add_argument("--fast-align-dir", default=a_cfg["fast_align_dir"])
    ap.add_argument("--iterations", type=int, default=a_cfg["passes"],
                    help="-I у fast_align: число EM-итераций")
    ap.add_argument("--no-null", action="store_true", help="-N: не выделять пустое выравнивание")
    ap.add_argument("--multiplier", type=float, default=0.0,
                    help="-m: ожидаемый множитель длины таргета (0 = дефолт fast_align)")
    ap.add_argument("--atools-command", default=a_cfg["diagonal"],
                    help="симметризация: grow-diag-final-and, grow, diag, union, intersect")
    args = ap.parse_args()

    align_dir = path(cfg, "align")
    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    fa_dir = Path(args.fast_align_dir)
    if not fa_dir.is_absolute():
        fa_dir = ROOT / fa_dir

    if args.pieces:
        spm_dir = path(cfg, "spm_data")
        src_file = spm_dir / f"{tag}.{cfg['corpus']['src']}.piece"
        tgt_file = spm_dir / f"{tag}.{cfg['corpus']['tgt']}.piece"
        if not src_file.exists() or not tgt_file.exists():
            raise SystemExit(f"Нет разбитых файлов: {src_file}, {tgt_file}. Сначала выполните 03_spm.py")
        src_lines = src_file.read_text(encoding="utf-8").splitlines()
        tgt_lines = tgt_file.read_text(encoding="utf-8").splitlines()
    else:
        src_file, tgt_file = corpus_files(cfg, path(cfg, args.input_dir))
        pairs = read_pair(src_file, tgt_file)
        src_lines = [" ".join(p[0].split()) for p in pairs]
        tgt_lines = [" ".join(p[1].split()) for p in pairs]

    if len(src_lines) != len(tgt_lines):
        raise SystemExit("Число строк в частях корпуса не совпадает")

    paired = align_dir / f"{tag}.txt"
    with paired.open("w", encoding="utf-8", newline="\n") as f:
        for s, t in zip(src_lines, tgt_lines):
            f.write(f"{s} ||| {t}\n")
    print(f"Файл «вход ||| выход»: {paired.name} ({len(src_lines)} строк)")

    fast_align = find_binary("fast_align", fa_dir)
    atools = find_binary("atools", fa_dir)
    if fast_align is None or atools is None:
        raise SystemExit(
            "Не найдены бинари fast_align/atools в каталоге "
            f"{fa_dir}. Соберите их: powershell -File setup.ps1 -Tools\n"
            "(нужен MSYS2/MinGW: g++, cmake, make)."
        )

    # Каноническая команда из README fast_align: -d -o -v (прямое направление),
    # -r добавляется для обратного направления.
    base = [fast_align.as_posix(), "-i", paired.as_posix(), "-d", "-o", "-v"]
    if args.iterations:
        base += ["-I", str(args.iterations)]
    if args.no_null:
        base += ["-N"]
    if args.multiplier:
        base += ["-m", str(args.multiplier)]

    fwd = align_dir / f"{tag}.forward.align"
    rev = align_dir / f"{tag}.reverse.align"
    run(base, fwd)
    run(base + ["-r"], rev)

    final = align_dir / f"{tag}.align"
    run(
        [
            atools.as_posix(),
            "-i",
            fwd.as_posix(),
            "-j",
            rev.as_posix(),
            "-c",
            args.atools_command,
        ],
        final,
    )

    final_lines = final.read_text(encoding="utf-8").splitlines()
    fwd_lines = fwd.read_text(encoding="utf-8").splitlines()
    rev_lines = rev.read_text(encoding="utf-8").splitlines()
    types: Counter[str] = Counter()
    empty = 0
    for line in final_lines:
        if not line.strip():
            empty += 1
        for tok in line.split():
            if ":" in tok:
                types[tok.split(":")[-1]] += 1

    summary = [
        "Команды:",
        f"  {fast_align.name} -i {paired.name} -d -o -v -I {args.iterations} > {fwd.name}",
        f"  {fast_align.name} -i {paired.name} -d -o -v -r -I {args.iterations} > {rev.name}",
        f"  {atools.name} -i {fwd.name} -j {rev.name} -c {args.atools_command} > {final.name}",
        "",
        f"Пар в корпусе: {len(src_lines)}",
        f"Строк: forward={len(fwd_lines)}, reverse={len(rev_lines)}, итог={len(final_lines)}",
    ]
    if types:
        summary.append("Типы выравниваний в Pharaoh-формате (1 = 1-1, 2 = frag, 3 = partial, 4 = null):")
        for typ in ("1", "2", "3", "4"):
            if types.get(typ):
                summary.append(f"  {typ}: {types[typ]}")
    else:
        summary.append(
            "Суффиксы типа (вид 0-1:1) в этом выходе atools отсутствуют: grow-diag-final-and "
            "печатает пары i-j без типа. Это корректный формат; типы при желании добавляет "
            "force_align.py из репозитория fast_align."
        )
    if empty:
        summary.append(f"  пустые строки выравнивания: {empty}")
    scored_src = path(cfg, "scored") / f"{tag}.{cfg['corpus']['src']}"
    scored_tgt = path(cfg, "scored") / f"{tag}.{cfg['corpus']['tgt']}"
    summary += [
        "",
        f"Вход ||| выход: {paired.name}",
        f"Итоговый файл: {final.name}",
        f"Следующий шаг: python scripts/05_get_rand_sample.py {len(final_lines)} 3000 "
        f"{scored_src} {scored_tgt} {final.name}",
    ]
    logs = path(cfg, "logs")
    (logs / f"{tag}_04_align_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
