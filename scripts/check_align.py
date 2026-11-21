"""Быстрая сверка файла выравнивания: формат, несколько пар глазами."""
import collections
import pathlib

align = pathlib.Path("align/Tatoeba.en-ru.align").read_text(encoding="utf-8").splitlines()
paired = pathlib.Path("align/Tatoeba.en-ru.txt").read_text(encoding="utf-8").splitlines()
print("строк в align:", len(align), "| строк в |||-файле:", len(paired))

types = collections.Counter()
for line in align:
    for tok in line.split():
        parts = tok.split(":")
        if len(parts) == 3:
            types[parts[2]] += 1
        elif len(parts) == 2:
            types["без типа"] += 1
print("типы:", dict(types))
print("пустые строки выравнивания:", sum(1 for line in align if not line.strip()))

for i in range(4):
    src, tgt = paired[i].split(" ||| ")
    s_tokens = src.split()
    t_tokens = tgt.split()
    shown = []
    for tok in align[i].split():
        a, b = tok.split("-")[0], tok.split("-")[1]
        shown.append(f"{s_tokens[int(a)]}->{t_tokens[int(b)]}")
    print(f"--- пара {i + 1}: {src} | {tgt}")
    print("    ", " ".join(shown) if shown else "(пусто)")
