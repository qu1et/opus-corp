"""Задача 2 (часть 2): анализ скоров и прочистка корпуса по порогу.

Читает scores.tsv (см. 02_score.py), строит распределение, показывает таблицу
порогов «сколько пар уйдёт при таком пороге», пишет файл на ручную проверку пар
около границы и затем применяет порог.

Выход:
  data/scored/<corpus>.<src>-<tgt>.<src> / .<tgt>  — прочищенный по скору корпус
  logs/02_histogram.txt          — гистограмма распределения скоров
  logs/02_threshold_candidates.txt — таблица порогов и доли удалённых
  logs/02_review.txt             — пары около порога для ручной оценки
  logs/02_filtered.log           — удалённые пары, формат ТЗ + скор:
      SCORE_LOW <скор>
      <строка на входном языке>
      <строка на выходном языке>
  logs/02_summary.txt            — итог: параметр порога и числа

Запуск:
  python scripts/02_filter.py --threshold 0.65      # явный порог
  python scripts/02_filter.py --auto --target-drop 0.12   # порог по целевой доле удалений
  python scripts/02_filter.py --percentile 10       # порог = 10-й процентиль скоров
  python scripts/02_filter.py --report-only         # только анализ, без прочистки
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    load_config,
    out_pair,
    path,
    write_pair,
)


def read_scores(scores_file: Path) -> tuple[np.ndarray, list[tuple[str, str]]]:
    scores: list[float] = []
    pairs: list[tuple[str, str]] = []
    with scores_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sc, src, tgt = line.split("\t", 2)
            scores.append(float(sc))
            pairs.append((src.rstrip("\n"), tgt.rstrip("\n")))
    return np.asarray(scores, dtype=np.float32), pairs


def histogram(scores: np.ndarray, bins: int = 40) -> str:
    """Гистограмма по фактическому диапазону скоров (эмбеддинговые скоры 0..1,
    скоры fast_align отрицательные — диапазон берём по данным)."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    hist, edges = np.histogram(scores, bins=bins, range=(lo, hi))
    width = 70
    step = (hi - lo) / bins
    lines = [
        f"Гистограмма скоров ({bins} корзин по {step:.4f}, шаг = {max(hist) / width:.0f} пар):",
        "",
    ]
    for count, edge in zip(hist, edges[:-1]):
        bar = "#" * int(count / max(hist) * width) if max(hist) else ""
        lines.append(f"{edge:.4f}..{edge + step:.4f} {count:>7} {bar}")
    return "\n".join(lines) + "\n"


def candidates(scores: np.ndarray) -> list[tuple[float, int, float]]:
    """Таблица порогов: для нормированных скоров (0..1) фиксированные пороги,
    иначе — пороги по квантилям."""
    total = len(scores)
    lo, hi = float(scores.min()), float(scores.max())
    if 0.0 <= lo and hi <= 1.0:
        ths = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    else:
        ths = tuple(float(np.quantile(scores, q)) for q in
                    (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75))
    rows = []
    for th in ths:
        removed = int((scores < th).sum())
        rows.append((th, removed, removed / total))
    return rows


def auto_threshold(scores: np.ndarray, target_drop: float) -> float:
    """Порог, при котором уходит примерно target_drop доля пар."""
    ordered = np.sort(scores)
    idx = int(len(ordered) * target_drop)
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx]) + 1e-6


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="анализ скоров и прочистка по порогу")
    ap.add_argument("--scores", default=None, help="файл scores.tsv (по умолчанию из data/scored)")
    ap.add_argument("--threshold", type=float, default=cfg["score"]["threshold"])
    ap.add_argument("--auto", action="store_true", help="выбрать порог автоматически")
    ap.add_argument("--target-drop", type=float, default=0.12,
                    help="целевая доля удаляемых пар при --auto")
    ap.add_argument("--percentile", type=float, default=None,
                    help="порог как процентиль скоров (например 10)")
    ap.add_argument("--review-lines", type=int, default=30, help="сколько пар около границы писать на проверку")
    ap.add_argument("--report-only", action="store_true", help="только анализ, без записи корпуса")
    ap.add_argument("--source-label", default="",
                    help="метка источника скора в именах логов (например fastalign), чтобы анализ "
                         "разных инструментов не перезаписывал выводы другого")
    args = ap.parse_args()

    scored_dir = path(cfg, "scored")
    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    logtag = tag + ("_" + args.source_label if args.source_label else "")
    scores_file = Path(args.scores) if args.scores else scored_dir / f"{tag}.scores.tsv"
    if not scores_file.exists():
        raise SystemExit(f"Нет файла скоров: {scores_file}. Сначала выполните 02_score.py")

    scores, pairs = read_scores(scores_file)
    logs = path(cfg, "logs")

    (logs / f"{logtag}_02_histogram.txt").write_text(histogram(scores), encoding="utf-8")

    rows = candidates(scores)
    header = ["Порог -> сколько пар уходит (доля):", ""]
    if float(scores.min()) < 0.0:
        header.append(
            "(скоры отрицательные: это лог-правдоподобие выравнивания fast_align, "
            "значения приведены по квантилям; ближе к 0 = лучше)"
        )
        header.append("")
    table = header
    for th, removed, frac in rows:
        table.append(f"  >= {th:.4f}  удалено {removed:>7} ({frac * 100:5.1f}%)")
    (logs / f"{logtag}_02_threshold_candidates.txt").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))

    if args.percentile is not None:
        threshold = float(np.quantile(scores, args.percentile / 100.0))
        how = f"процентиль {args.percentile}"
    elif args.auto:
        threshold = auto_threshold(scores, args.target_drop)
        how = f"auto, target_drop={args.target_drop}"
    elif args.threshold is not None:
        threshold = float(args.threshold)
        how = "явный порог"
    else:
        raise SystemExit(
            "Не задан порог. Посмотрите "
            f"{logs / f'{logtag}_02_threshold_candidates.txt'} и запустите\n"
            "  --threshold X, --auto --target-drop Y или --percentile P"
        )

    below = np.where(scores < threshold)[0]
    above = np.where(scores >= threshold)[0]

    # Пары около границы: верхние из удалённых и нижние из оставленных — для ручной оценки.
    near_low = sorted(below, key=lambda i: -scores[i])[: args.review_lines]
    near_high = sorted(above, key=lambda i: scores[i])[: args.review_lines]
    review = [f"Порог {threshold:.4f} ({how}). Ниже порога (кандидаты на возврат):", ""]
    for i in near_low:
        review.append(f"{scores[i]:.3f} | {pairs[i][0]} | {pairs[i][1]}")
    review += ["", "Выше порога (кандидаты на удаление):", ""]
    for i in near_high:
        review.append(f"{scores[i]:.3f} | {pairs[i][0]} | {pairs[i][1]}")
    (logs / f"{logtag}_02_review.txt").write_text("\n".join(review) + "\n", encoding="utf-8")

    dropped = [("SCORE_LOW %.4f" % scores[i], pairs[i][0], pairs[i][1]) for i in below]
    filtered_log = logs / f"{logtag}_02_filtered.log"
    with filtered_log.open("w", encoding="utf-8", newline="\n") as f:
        for typ, src, tgt in dropped:
            f.write(f"{typ}\n{src}\n{tgt}\n")

    kept = [(pairs[i][0], pairs[i][1]) for i in above]
    if not args.report_only:
        out_src, out_tgt = out_pair(cfg, "scored")
        write_pair(kept, out_src, out_tgt)
        kept_files = (out_src.name, out_tgt.name)
    else:
        kept_files = ("не записан (--report-only)", "")

    score_desc = (f"источник скора: {args.source_label}" if args.source_label
                  else f"{cfg['score']['model']}, косинусное сходство эмбеддингов")
    summary = [
        f"Корпус: {cfg['corpus']['name']} {cfg['corpus']['src']} -> {cfg['corpus']['tgt']}",
        f"Скор: {scores_file.name} ({score_desc})",
        f"Порог: {threshold:.4f} ({how})",
        f"Пар всего: {len(scores)}",
        f"Осталось: {len(kept)}",
        f"Удалено: {len(below)} ({len(below) / len(scores) * 100:.1f}%)",
        f"Скор оставленных: min={scores[above].min():.3f} mean={scores[above].mean():.3f}",
        f"Файлы: {kept_files[0]} {kept_files[1]}",
        f"Лог удалённых: {filtered_log.name}",
    ]
    (logs / f"{logtag}_02_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
