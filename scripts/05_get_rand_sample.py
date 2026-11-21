"""Задача 5: выделение случайной части из группы файлов, не нарушая параллельности.

Аргументы (позиционные, как в примере из ТЗ):
  python get_rand_sample.py 1067840 3000 Tanzil.en-ru.en Tanzil.en-ru.ru Tanzil.en-ru.align

  argv[1] — число строк во входном корпусе
  argv[2] — сколько строк нужно в случайной части
  argv[3:] — файлы корпусов, которые требуется обработать (от 1 и больше)

На выход создаёт для каждого входного файла файл с суффиксом .rand
(Tanzil.en-ru.en -> Tanzil.en-ru.en.rand). Одна и та же выборка индексов
применяется ко всем файлам, поэтому параллельность сохраняется: строка N в
входном корпусе, её перевод и её выравнивание попадают в одну и ту же строку
выборки.

Дополнительные ключи (не обязательны по ТЗ):
  --seed N            воспроизводимость выборки
  --out-dir каталог   куда писать .rand (по умолчанию рядом с исходными файлами)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> tuple[int, int, list[Path], dict[str, object]]:
    """Разбирает sys.argv: сначала позиционные аргументы, потом ключи."""
    positional: list[str] = []
    opts: dict[str, object] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--seed":
            opts["seed"] = int(argv[i + 1])
            i += 2
        elif token == "--out-dir":
            opts["out_dir"] = argv[i + 1]
            i += 2
        elif token == "--no-indexes":
            opts["no_indexes"] = True
            i += 1
        else:
            positional.append(token)
            i += 1

    if len(positional) < 3:
        raise SystemExit(
            "Аргументы: <число строк во входном корпусе> <нужное число строк> <файл> [файл…]\n"
            "Пример: python 05_get_rand_sample.py 554788 3000 "
            "data/scored/Tatoeba.en-ru.en data/scored/Tatoeba.en-ru.ru align/Tatoeba.en-ru.align"
        )
    total = int(positional[0])
    take = int(positional[1])
    files = [Path(p) for p in positional[2:]]
    return total, take, files, opts


def main(argv: list[str] | None = None) -> None:
    # sys.argv[0] — имя скрипта, позиционные аргументы идут с argv[1] (как в ТЗ).
    argv = sys.argv[1:] if argv is None else argv
    total, take, files, opts = parse_args(argv)

    if take > total:
        raise SystemExit(f"Запрошено {take} строк, а в корпусе заявлено {total}")

    contents: list[list[str]] = []
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        if len(lines) != total:
            raise SystemExit(
                f"{f.name}: {len(lines)} строк, а ожидается {total} — параллельность нарушена"
            )
        contents.append(lines)

    rng = random.Random(opts.get("seed")) if opts.get("seed") is not None else random.Random()
    indexes = sorted(rng.sample(range(total), take))

    out_dir = Path(str(opts.get("out_dir"))) if opts.get("out_dir") else None
    written: list[Path] = []
    for f, lines in zip(files, contents):
        target = (out_dir / (f.name + ".rand")) if out_dir else f.with_name(f.name + ".rand")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as out:
            for i in indexes:
                out.write(lines[i] + "\n")
        written.append(target)
        print(f"{f.name} -> {target.name} ({take} строк)")

    # Индексы выборки сохраняем: по ним можно точно проверить параллельность
    # (в корпусе есть повторяющиеся строки, так что поиск по содержимому ненадёжен).
    if not opts.get("no_indexes"):
        idx_file = (out_dir if out_dir else files[0].parent) / "rand_indexes.txt"
        idx_file.write_text("\n".join(str(i) for i in indexes) + "\n", encoding="utf-8", newline="\n")
        print(f"Индексы выборки: {idx_file.name} ({take} чисел)")

    check = {len(target.read_text(encoding='utf-8').splitlines()) for target in written}
    if check != {take}:
        raise SystemExit(f"Проверка не пройдена: числа строк в .rand файлах {sorted(check)}")
    print(f"Готово: {len(written)} файлов по {take} строк, выборка индексов одна для всех файлов")


if __name__ == "__main__":
    main()
