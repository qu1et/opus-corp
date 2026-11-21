"""Оркестратор: прочистка корпуса и все пять задач одной командой.

Свои корпуса:
  python scripts/run_all.py --raw-src my.en --raw-tgt my.ru --name MyCorpus --src en --tgt ru
Корпус с OPUS:
  python scripts/run_all.py --corpus Tatoeba --version v2026-07-08 --pair en-ru
Только прочистка по правилам ТЗ (задача 1):
  python scripts/run_all.py --stages 1
Скоринг и прочистка по скору (задача 2) с явным порогом:
  python scripts/run_all.py --stages 2 --threshold 0.65
Без гейтованной Gemma-модели (не нужен токен HF):
  python scripts/run_all.py --stages 2 --score-model intfloat/multilingual-e5-large
Скоринг через fast_align -s (вместо эмбеддингов, требует собранного fast_align):
  python scripts/run_all.py --stages 2 --score-method fastalign --percentile 10
Задача 4 на разбитом SentencePiece корпусе:
  python scripts/run_all.py --stages 4 --align-input pieces

Параметры по умолчанию читаются из config.json; ключи командной строки переписывают
config.json, чтобы остальные скрипты работали с теми же значениями.

Выход каждого этапа лежит в data/steps, data/clean, data/scored, data/spm, spm/,
align/, rand/, логи и сводки — в logs/. Итоговая сводка всех этапов: logs/ALL_summary.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, count_lines, load_config, path  # noqa: E402

SCRIPTS = {
    "fetch": "00_fetch_corpus.py",
    "clean": "01_clean.py",
    "score": "02_score.py",
    "score_fastalign": "02_score_fastalign.py",
    "filter": "02_filter.py",
    "spm": "03_spm.py",
    "align": "04_align.py",
    "sample": "05_get_rand_sample.py",
    "report": "06_report.py",
}


def python_exe() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return venv.as_posix()
    alt = ROOT / ".venv" / "bin" / "python"
    if alt.exists():
        return alt.as_posix()
    return sys.executable


def run(script: str, extra: list[str], label: str) -> None:
    cmd = [python_exe(), (ROOT / "scripts" / script).as_posix()] + extra
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if res.returncode != 0:
        raise SystemExit(f"Этап '{label}' завершился с кодом {res.returncode}")


def save_config(cfg: dict) -> None:
    (ROOT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_input(cfg: dict) -> str:
    """Для этапов 3 и 4: прочищенный по скору корпус, а если скоринга не было — прочистка по правилам ТЗ."""
    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    scored = path(cfg, "scored") / f"{tag}.{cfg['corpus']['src']}"
    return "scored" if scored.exists() else "clean"


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="пайплайн задач 1-5")
    ap.add_argument("--stages", default="1,2,3,4,5,6",
                    help="номера этапов через запятую: 1-5 плюс 6 (отчёт report.md); 0 = скачать корпус")
    ap.add_argument("--corpus", help="имя корпуса на OPUS")
    ap.add_argument("--version", help="версия корпуса на OPUS")
    ap.add_argument("--pair", default=None, help="языковая пара, например en-ru")
    ap.add_argument("--preprocessor", default="moses")
    ap.add_argument("--raw-src", help="локальный файл входной части своего корпуса")
    ap.add_argument("--raw-tgt", help="локальный файл выходной части своего корпуса")
    ap.add_argument("--name", help="имя корпуса для имён файлов")
    ap.add_argument("--src", help="код входного языка")
    ap.add_argument("--tgt", help="код выходного языка")
    ap.add_argument("--min-words", type=int, default=cfg["clean"]["min_words"])
    ap.add_argument("--alpha-ratio", type=float, default=cfg["clean"]["alpha_ratio"])
    ap.add_argument("--len-ratio", type=float, default=cfg["clean"]["len_ratio"])
    ap.add_argument("--max-chars", type=int, default=cfg["clean"]["max_chars"])
    ap.add_argument("--no-lang", action="store_true")
    ap.add_argument("--dedup", action="store_true", default=cfg["clean"]["dedup"])
    ap.add_argument("--cyrillic-alpha", action="store_true")
    ap.add_argument("--alpha-ignore-digits", action="store_true",
                    help="убрать цифры из знаменателя доли алфавитных символов")
    ap.add_argument("--score-model", default=cfg["score"]["model"])
    ap.add_argument("--score-method", default="embedding", choices=["embedding", "fastalign"],
                    help="чем скорить: эмбеддинги (по умолчанию) или fast_align -s (нужен собранный fast_align)")
    ap.add_argument("--batch-size", type=int, default=cfg["score"]["batch_size"])
    ap.add_argument("--max-seq-len", type=int, default=cfg["score"].get("max_seq_len", 0),
                    help="обрезка длины для эмбеддера (0 = максимум модели)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threshold", type=float, default=cfg["score"]["threshold"])
    ap.add_argument("--auto", action="store_true", help="порог по целевой доле удалений")
    ap.add_argument("--target-drop", type=float, default=cfg["score"].get("target_drop", 0.12))
    ap.add_argument("--percentile", type=float)
    ap.add_argument("--vocab-size", type=int, default=cfg["spm"]["vocab_size"])
    ap.add_argument("--model-type", default=cfg["spm"]["model_type"], choices=["bpe", "unigram"])
    ap.add_argument("--align-input", default="word", choices=["word", "pieces"],
                    help="что подавать на выравнивание: слова корпуса или куски SPM")
    ap.add_argument("--sample-size", type=int, default=cfg["sample"]["size"])
    ap.add_argument("--seed", type=int, default=cfg["sample"]["seed"])
    ap.add_argument("--include-pieces", action="store_true",
                    help="в задачу 5 дополнительно захватить .piece файлы")
    args = ap.parse_args()

    stages = [int(x) for x in args.stages.split(",") if x.strip()]

    # Имя и направление корпуса пишем в config.json — все скрипты читают его.
    if args.name or args.src or args.tgt:
        cfg["corpus"]["name"] = args.name or cfg["corpus"]["name"]
        cfg["corpus"]["src"] = args.src or cfg["corpus"]["src"]
        cfg["corpus"]["tgt"] = args.tgt or cfg["corpus"]["tgt"]
    cfg["clean"].update(
        {"min_words": args.min_words, "alpha_ratio": args.alpha_ratio,
         "len_ratio": args.len_ratio, "max_chars": args.max_chars, "dedup": args.dedup}
    )
    cfg["score"].update({"model": args.score_model, "batch_size": args.batch_size,
                         "threshold": args.threshold, "target_drop": args.target_drop})
    cfg["spm"].update({"vocab_size": args.vocab_size, "model_type": args.model_type})
    cfg["sample"].update({"size": args.sample_size, "seed": args.seed})
    save_config(cfg)

    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    print(f"Корпус: {tag}")
    print(f"Этапы: {stages}")

    if 0 in stages or args.raw_src or args.corpus:
        fetch = [
            "--name", cfg["corpus"]["name"], "--src", cfg["corpus"]["src"],
            "--tgt", cfg["corpus"]["tgt"],
        ]
        if args.raw_src and args.raw_tgt:
            fetch += ["--files", args.raw_src, args.raw_tgt]
        elif args.corpus:
            fetch += ["--corpus", args.corpus, "--version", args.version,
                      "--pair", args.pair or f"{cfg['corpus']['src']}-{cfg['corpus']['tgt']}",
                      "--preprocessor", args.preprocessor]
        run(SCRIPTS["fetch"], fetch, "0. корпус на месте")

    if 1 in stages:
        extra = []
        if args.no_lang:
            extra.append("--no-lang")
        if args.cyrillic_alpha:
            extra.append("--cyrillic-alpha")
        if args.alpha_ignore_digits:
            extra.append("--alpha-ignore-digits")
        run(SCRIPTS["clean"], extra, "1. прочистка по правилам ТЗ")

    if 2 in stages:
        if args.score_method == "fastalign":
            run(SCRIPTS["score_fastalign"], ["--input-dir", "clean"], "2a. скоринг через fast_align -s")
            scores_file = path(cfg, "scored") / f"{tag}.fastalign.scores.tsv"
            filt = ["--scores", scores_file.as_posix(), "--source-label", "fastalign"]
        else:
            emb = ["--input-dir", "clean", "--batch-size", str(args.batch_size)]
            if args.max_seq_len:
                emb += ["--max-seq-len", str(args.max_seq_len)]
            if args.cpu:
                emb.append("--cpu")
            run(SCRIPTS["score"], emb, "2a. скоринг эмбеддингами")
            filt = []
        if args.threshold:
            filt += ["--threshold", str(args.threshold)]
        elif args.percentile is not None:
            filt += ["--percentile", str(args.percentile)]
        else:
            filt += ["--auto", "--target-drop", str(args.target_drop)]
        run(SCRIPTS["filter"], filt, "2b. анализ скоров и прочистка по порогу")

    if 3 in stages:
        run(SCRIPTS["spm"], ["--input-dir", default_input(cfg)], "3. SentencePiece")

    if 4 in stages:
        extra = ["--input-dir", default_input(cfg)]
        if args.align_input == "pieces":
            extra += ["--pieces"]
        run(SCRIPTS["align"], extra, "4. файл выравнивания")

    if 5 in stages:
        align_dir = path(cfg, "align")
        input_dir = default_input(cfg)
        files = [
            path(cfg, input_dir) / f"{tag}.{cfg['corpus']['src']}",
            path(cfg, input_dir) / f"{tag}.{cfg['corpus']['tgt']}",
            align_dir / f"{tag}.align",
        ]
        if args.include_pieces:
            spm_data = path(cfg, "spm_data")
            files += [
                spm_data / f"{tag}.{cfg['corpus']['src']}.piece",
                spm_data / f"{tag}.{cfg['corpus']['tgt']}.piece",
            ]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"Нет входных файлов для выборки: {[m.name for m in missing]}")
        n = count_lines(files[0])
        run(SCRIPTS["sample"], [str(n), str(args.sample_size),
                                *[f.as_posix() for f in files],
                                "--seed", str(args.seed),
                                "--out-dir", path(cfg, "rand").as_posix()],
            "5. случайная часть")

    if 6 in stages:
        run(SCRIPTS["report"], [], "6. отчёт по всем этапам")

    lines = ["Итог пайплайна:", f"Корпус: {tag}", ""]
    for label, d in (("сырой корпус", "raw"), ("после правил ТЗ", "clean"),
                     ("после скоринга", "scored"), ("куски SPM", "spm_data"),
                     ("выравнивание", "align"), ("случайная часть", "rand")):
        p = path(cfg, d)
        items = sorted(f.name for f in p.iterdir() if f.is_file())
        lines.append(f"{label}: {p}")
        for name in items[:12]:
            lines.append(f"    {name} ({count_lines(p / name)} строк)" if name.endswith(
                (".en", ".ru", ".align", ".piece", ".rand", ".txt", ".tsv")) else f"    {name}")
    logs = path(cfg, "logs")
    lines += ["", "Логи и сводки:", *[f"    {f.name}" for f in sorted(logs.iterdir())]]
    (logs / "ALL_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
