"""Собирает итоговый отчёт по всем этапам в report.md из логов и файлов артефактов.

Отчёт — то, что обычно отправляется преподавателю по каждому заданию: выбранный корпус,
результат каждого этапа, параметры прочистки, лог отброшенных строк и код. Здесь только
сводка по уже выполненным этапам; сами артефакты лежат в data/, spm/, align/, rand/, logs/.

Запуск:
  python scripts/06_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, count_lines, load_config, path, read_pair  # noqa: E402


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else "(нет файла)"


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def file_table(files: list[Path]) -> str:
    rows = ["| файл | строк | размер |", "|---|---:|---:|"]
    for f in files:
        if not f.exists():
            continue
        try:
            n = count_lines(f)
        except Exception:
            n = "-"
        rows.append(f"| `{f.as_posix()}` | {n} | {f.stat().st_size / 1e6:.1f} МБ |")
    return "\n".join(rows)


def main() -> None:
    cfg = load_config()
    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    logs = path(cfg, "logs")

    raw_src, raw_tgt = path(cfg, "raw") / f"{tag}.{cfg['corpus']['src']}", path(cfg, "raw") / f"{tag}.{cfg['corpus']['tgt']}"
    clean_src, clean_tgt = path(cfg, "clean") / f"{tag}.{cfg['corpus']['src']}", path(cfg, "clean") / f"{tag}.{cfg['corpus']['tgt']}"
    scored_src, scored_tgt = path(cfg, "scored") / f"{tag}.{cfg['corpus']['src']}", path(cfg, "scored") / f"{tag}.{cfg['corpus']['tgt']}"
    piece_src = path(cfg, "spm_data") / f"{tag}.{cfg['corpus']['src']}.piece"
    piece_tgt = path(cfg, "spm_data") / f"{tag}.{cfg['corpus']['tgt']}.piece"
    align_file = path(cfg, "align") / f"{tag}.align"
    paired = path(cfg, "align") / f"{tag}.txt"

    parts = [
        "# Отчёт по практическим заданиям 1-5\n",
        f"Корпус: **{cfg['corpus']['name']}**, направление **{cfg['corpus']['src']} → {cfg['corpus']['tgt']}**.\n",
    ]

    parts.append(section("Задание 1: прочистка по правилам ТЗ", read_text(logs / f"{tag}_01_summary.txt")))
    parts.append(section("Файлы после задания 1", file_table([raw_src, raw_tgt, clean_src, clean_tgt])))
    parts.append(section("Логи отброшенных строк (формат: ТИП / вход / выход)", file_table(
        [logs / f"{tag}_{code}.log" for code in ("SHORT", "LEN_RATIO", "ALPHA", "LANG", "DUP")]
    )))

    parts.append(section("Задание 2: скоринг и прочистка по скору", read_text(logs / f"{tag}_02_score_summary.txt")))
    parts.append(section("Порог и результат прочистки по скору", read_text(logs / f"{tag}_02_summary.txt")))
    parts.append(section("Распределение скоров", read_text(logs / f"{tag}_02_histogram.txt")))
    parts.append(section("Таблица порогов", read_text(logs / f"{tag}_02_threshold_candidates.txt")))
    parts.append(section("Ручная проверка пар около границы", read_text(logs / f"{tag}_02_review.txt")))
    parts.append(section("Альтернативный скоринг fast_align -s", read_text(logs / f"{tag}_02_fastalign_summary.txt")))
    parts.append(section("Альтернативный скоринг: сводка анализа и порог", read_text(logs / f"{tag}_fastalign_02_summary.txt")))
    parts.append(section("Альтернативный скоринг: распределение скоров", read_text(logs / f"{tag}_fastalign_02_histogram.txt")))
    parts.append(section("Альтернативный скоринг: таблица порогов", read_text(logs / f"{tag}_fastalign_02_threshold_candidates.txt")))
    parts.append(section("Файлы после задания 2", file_table([scored_src, scored_tgt])))
    parts.append(section("Лог удалённых по скору (формат: SCORE_LOW скор / вход / выход)",
                         file_table([logs / f"{tag}_02_filtered.log"])))

    parts.append(section("Задание 3: SentencePiece", read_text(logs / f"{tag}_03_spm_stats.txt")))
    parts.append(section("Файлы после задания 3", file_table(
        [path(cfg, "spm") / f"{cfg['corpus']['name']}.{cfg['corpus']['src']}.model",
         path(cfg, "spm") / f"{cfg['corpus']['name']}.{cfg['corpus']['tgt']}.model",
         piece_src, piece_tgt]
    )))

    parts.append(section("Задание 4: файл выравнивания", read_text(logs / f"{tag}_04_align_summary.txt")))
    parts.append(section("Файлы после задания 4", file_table([paired,
                                                            path(cfg, "align") / f"{tag}.forward.align",
                                                            path(cfg, "align") / f"{tag}.reverse.align",
                                                            align_file])))

    rand_dir = path(cfg, "rand")
    rand_files = sorted(p for p in rand_dir.iterdir() if p.is_file())
    parts.append(section("Задание 5: случайная часть", file_table(rand_files)))

    code = sorted((ROOT / "scripts").glob("*.py"))
    parts.append(section("Код", "\n".join(f"- `scripts/{p.name}`" for p in code)))

    if clean_src.exists() and scored_src.exists():
        n_clean = len(read_pair(clean_src, clean_tgt))
        n_scored = len(read_pair(scored_src, scored_tgt))
        parts.append(section("Итог по числам",
                             f"Сырой корпус: {count_lines(raw_src)} пар → после правил ТЗ: {n_clean} пар → "
                             f"после скоринга: {n_scored} пар"))

    (ROOT / "report.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Отчёт: {ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
