"""Проверка, что случайная часть не нарушает параллельность.

Точная сверка по сохранённому файлу индексов rand/rand_indexes.txt: для каждой
строки выборки берём тот же индекс из исходного корпуса и проверяем, что вход,
выход и выравнивание соответствуют одной и той же паре.
"""
import pathlib
import sys

indexes_file = pathlib.Path("rand/rand_indexes.txt")
if not indexes_file.exists():
    raise SystemExit(
        "Нет rand/rand_indexes.txt — запустите 05_get_rand_sample.py без --no-indexes"
    )

src_file = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/scored/Tatoeba.en-ru.en")
tgt_file = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/scored/Tatoeba.en-ru.ru")
align_file = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "align/Tatoeba.en-ru.align")
rand_dir = pathlib.Path(sys.argv[4] if len(sys.argv) > 4 else "rand")

src = src_file.read_text(encoding="utf-8").splitlines()
tgt = tgt_file.read_text(encoding="utf-8").splitlines()
align = align_file.read_text(encoding="utf-8").splitlines()
if not (len(src) == len(tgt) == len(align)):
    raise SystemExit(f"Размеры исходных файлов не совпадают: {len(src)}, {len(tgt)}, {len(align)}")

indexes = [int(x) for x in indexes_file.read_text(encoding="utf-8").split()]
rand_src = (rand_dir / (src_file.name + ".rand")).read_text(encoding="utf-8").splitlines()
rand_tgt = (rand_dir / (tgt_file.name + ".rand")).read_text(encoding="utf-8").splitlines()
rand_align = (rand_dir / (align_file.name + ".rand")).read_text(encoding="utf-8").splitlines()

print(f"исходный корпус: {len(src)} пар | выборка: {len(rand_src)} строк")
if not (len(indexes) == len(rand_src) == len(rand_tgt) == len(rand_align)):
    raise SystemExit(
        f"Размеры выборки не совпадают: indexes={len(indexes)}, src={len(rand_src)}, "
        f"tgt={len(rand_tgt)}, align={len(rand_align)}"
    )

bad = 0
for i, pos in enumerate(indexes):
    if src[pos] != rand_src[i] or tgt[pos] != rand_tgt[i] or align[pos] != rand_align[i]:
        bad += 1
        if bad <= 5:
            print(f"рассинхрон на позиции {pos}: {rand_src[i][:50]} | {rand_tgt[i][:50]}")
print(f"проверено строк: {len(indexes)}, рассинхронных: {bad}")
print("OK: параллельность сохранена" if bad == 0 else "PARALLELISM BROKEN")
