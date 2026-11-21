"""Задача 2 (альтернативный инструмент): скоринг параллельности через fast_align -s.

Как указано в ТЗ: к стандартной команде получения выравнивания добавляется флаг -s,
и вместе с выравниванием возвращается alignment score. Оценка делается в обе стороны
(forward и reverse), финальный скор — среднее двух значений (как в варианте с
marian-scorer: два направления, среднее из двух).

Выход:
  data/scored/<corpus>.<src>-<tgt>.fastalign.scores.tsv — скор \t вход \t выход,
      нормированный (raw / число токенов пары); тот же формат, что и у эмбеддингового
      скора, поэтому дальше с файлом работает scripts/02_filter.py (--scores).
  data/scored/<corpus>.<src>-<tgt>.fastalign.rawscores.tsv — сырой alignment score
      (суммарное лог-правдоподобие пары, отрицательное, растёт по длине).
  align/<corpus>.<src>-<tgt>.scores-input.txt — «вход ||| выход» для fast_align.
  logs/<corpus>.<src>-<tgt>_02_fastalign_summary.txt — команды и статистика скора

Требования: собранные fast_align.exe (setup.ps1 -Tools). По ТЗ этот метод плохо
работает на малых корпусах и требователен к памяти — на корпусах от ~100k пар он
нормальный.

Запуск:
  python scripts/02_score_fastalign.py --input-dir clean
  python scripts/02_filter.py --scores data/scored/..fastalign.scores.tsv --percentile 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT,
    corpus_files,
    load_config,
    path,
    read_pair,
)


def find_binary(name: str, hint_dir: Path) -> Path | None:
    for cand in (hint_dir / name, hint_dir / f"{name}.exe", ROOT / "tools" / "bin" / name):
        if cand.exists():
            return cand
    return None


def parse_score(line: str) -> float | None:
    """Скор из строки с флагом -s: fast_align печатает «пары ||| скор».

    Скор — логарифм правдоподобия выравнивания, он отрицательный; чем ближе к нулю,
    тем лучше выравнивание (значение «в норму 0..1» здесь не приводится, пороги
    подбираются квантилями: python scripts/02_filter.py --scores ... --percentile 5).
    """
    if " ||| " not in line:
        return None
    tail = line.rsplit(" ||| ", 1)[1].strip()
    try:
        return float(tail)
    except ValueError:
        return None


def run_forward(paired: Path, exe: Path, extra: list[str]) -> list[float]:
    cmd = [exe.as_posix(), "-i", paired.as_posix(), "-d", "-o", "-v", "-s"] + extra
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"fast_align не выполнился ({res.returncode}): {res.stderr.strip()[:400]}")
    return [parse_score(line) or 0.0 for line in res.stdout.splitlines()]


def main() -> None:
    cfg = load_config()
    a_cfg = cfg["align"]
    ap = argparse.ArgumentParser(description="скоринг через fast_align -s")
    ap.add_argument("--input-dir", default="clean", help="каталог с парой прочищенных файлов")
    ap.add_argument("--fast-align-dir", default=a_cfg["fast_align_dir"])
    ap.add_argument("--iterations", type=int, default=a_cfg["passes"])
    ap.add_argument("--from-raw-scores", default=None,
                    help="не запускать fast_align, а нормализовать уже полученный файл raw-скоров")
    args = ap.parse_args()

    src_file, tgt_file = corpus_files(cfg, path(cfg, args.input_dir))
    pairs = read_pair(src_file, tgt_file)
    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    align_dir = path(cfg, "align")

    if args.from_raw_scores:
        raw_path = Path(args.from_raw_scores)
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        raw = np.asarray([float(line.split("\t", 1)[0])
                          for line in raw_path.read_text(encoding="utf-8").splitlines()],
                         dtype=np.float32)
        if len(raw) != len(pairs):
            raise SystemExit(f"raw-скоров {len(raw)}, а пар в корпусе {len(pairs)}")
        fa = None
        print(f"Нормализация существующих raw-скоров: {raw_path.name}")
    else:
        fa = find_binary("fast_align", Path(args.fast_align_dir))
        if fa is None:
            raise SystemExit(
                "Нет fast_align.exe. Соберите: powershell -ExecutionPolicy Bypass -File setup.ps1 -Tools"
            )
        # Отдельный «вход ||| выход»: для скоринга корпус берётся до порога (clean),
        # а файл выравнивания задачи 4 строится на корпусе после порога (scored),
        # поэтому общий файл с именем {tag}.txt использовать нельзя.
        paired = align_dir / f"{tag}.scores-input.txt"
        with paired.open("w", encoding="utf-8", newline="\n") as f:
            for s, t in pairs:
                f.write(f"{' '.join(s.split())} ||| {' '.join(t.split())}\n")
        print(f"Файл «вход ||| выход»: {paired.name} ({len(pairs)} строк)")

        extra = ["-I", str(args.iterations)]
        fwd = run_forward(paired, fa, extra)
        rev = run_forward(paired, fa, extra + ["-r"])
        if len(fwd) != len(pairs) or len(rev) != len(pairs):
            raise SystemExit(f"Разные число строк: forward={len(fwd)}, reverse={len(rev)}, пар={len(pairs)}")
        raw = (np.asarray(fwd) + np.asarray(rev)) / 2.0

    # Скор — лог-правдоподобие всей пары, он растёт по длине предложения. Для порога
    # нормализуем на число токенов пары (raw / (len_src + len_tgt)): тогда порог
    # одинаково работает для коротких и длинных предложений.
    tok = np.asarray([len(s.split()) + len(t.split()) for s, t in pairs], dtype=np.float32)
    scores = np.asarray(raw / np.maximum(tok, 1.0), dtype=np.float32)

    scores_file = path(cfg, "scored") / f"{tag}.fastalign.scores.tsv"
    with scores_file.open("w", encoding="utf-8", newline="\n") as f:
        for (src, tgt), sc in zip(pairs, scores):
            f.write(f"{sc:.4f}\t{src}\t{tgt}\n")

    raw_file = path(cfg, "scored") / f"{tag}.fastalign.rawscores.tsv"
    with raw_file.open("w", encoding="utf-8", newline="\n") as f:
        for (src, tgt), sc in zip(pairs, raw):
            f.write(f"{sc:.4f}\t{src}\t{tgt}\n")

    logs = path(cfg, "logs")
    if fa is not None:
        commands = [
            f"Команды: {fa.name} -i {paired.name} -d -o -v -s -I {args.iterations}",
            f"         {fa.name} -i {paired.name} -d -o -v -s -r -I {args.iterations}",
        ]
    else:
        commands = [f"Скоры взяты из готового файла: {raw_path.name} (fast_align не запускался)"]
    summary = [
        "Инструмент: fast_align с флагом -s (alignment score)",
        "Оценка в обе стороны, финальный скор = среднее forward и reverse",
        *commands,
        f"Пар оценено: {len(scores)}",
        f"Скоры raw (лог-правдоподобие пары, отрицательные): min={raw.min():.3f} "
        f"max={raw.max():.3f} mean={raw.mean():.3f} std={raw.std():.3f}  файл {raw_file.name}",
        f"Скоры normalized (raw / число токенов пары): min={scores.min():.4f} "
        f"max={scores.max():.4f} mean={scores.mean():.4f} std={scores.std():.4f}  файл {scores_file.name}",
        "Квантили normalized (1,10,25,50,75,90,99%): " + ", ".join(
            f"{v:.4f}" for v in np.quantile(scores, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        ),
        "Порог подбирается по normalized скору: --percentile или --threshold (значение отрицательное).",
        f"Дальше: python scripts/02_filter.py --scores {scores_file.as_posix()} --percentile 5",
        "Примечание по ТЗ: метод не подходит для малых корпусов и требователен к памяти.",
    ]
    (logs / f"{tag}_02_fastalign_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
