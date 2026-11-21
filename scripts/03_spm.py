"""Задача 3: SentencePiece — две модели и разбивка обеих частей корпуса.

Обучает одну модель на входной части прочищенного корпуса и отдельную модель на
выходной части, затем разбивает оба корпуса этими моделями.

Выход:
  spm/<corpus>.<src>.model / .vocab       — модель и словарь для входной части
  spm/<corpus>.<tgt>.model / .vocab       — модель и словарь для выходной части
  data/spm/<corpus>.<src>-<tgt>.<src>.piece / .<tgt>.piece — разбитые корпуса
  logs/03_spm_stats.txt                   — средний размер предложения в кусках,
                                            размер словаря, примеры разбивки

Запуск (вход по умолчанию — прочищенный по скору корпус из data/scored):
  python scripts/03_spm.py
  python scripts/03_spm.py --input-dir clean --vocab-size 32000 --model-type unigram
  python scripts/03_spm.py --input-dir scored --vocab-size 16000 --model-type bpe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    corpus_files,
    load_config,
    path,
    read_pair,
)


def train_side(input_file: Path, model_stem: Path, args) -> spm.SentencePieceProcessor:
    """Обучает модель и возвращает процессор для неё.

    SPM пишет <model_prefix>.model и <model_prefix>.vocab, поэтому передаём
    model_prefix (поля model в TrainerSpec нет).
    """
    kwargs = {
        "input": str(input_file),
        "model_prefix": model_stem.as_posix(),
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "input_sentence_size": args.input_sentence_size,
        "shuffle_input_sentence": not args.no_shuffle,
        "hard_vocab_limit": args.hard_vocab_limit,
        "character_coverage": args.character_coverage,
    }
    if args.normalization_rule_file:
        kwargs["normalization_rule_file"] = str(args.normalization_rule_file)
    else:
        kwargs["normalization_rule_name"] = args.normalization
    spm.SentencePieceTrainer.train(**kwargs)
    # SPM дописывает .model/.vocab к model_prefix; with_suffix здесь не годится,
    # потому что в префиксе уже есть точка (имя корпуса и код языка).
    model_file = Path(model_stem.as_posix() + ".model")
    return spm.SentencePieceProcessor(model_file=model_file.as_posix())


def tokenize(sp: spm.SentencePieceProcessor, lines: list[str], out_file: Path) -> None:
    """Записывает разбитые строки по одному предложению на строку."""
    with out_file.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(" ".join(sp.encode(line, out_type=str)) + "\n")


def side_stats(sp: spm.SentencePieceProcessor, lines: list[str], sample: int = 5) -> list[str]:
    """Средний размер предложения в кусках, размер словаря и несколько примеров."""
    probe = lines[:5000]
    sizes = [len(sp.encode(line, out_type=str)) for line in probe]
    rows = [
        f"  vocab_size={sp.get_piece_size()}",
        f"  piece_count: avg={sum(sizes) / max(1, len(sizes)):.2f}, max={max(sizes) if sizes else 0}",
        "  примеры разбивки:",
    ]
    for line in lines[:sample]:
        pieces = sp.encode(line, out_type=str)
        rows.append(f"    {line!r} -> {' '.join(pieces)}")
    return rows


def main() -> None:
    cfg = load_config()
    sp_cfg = cfg["spm"]
    ap = argparse.ArgumentParser(description="обучение и применение SentencePiece (задача 3)")
    ap.add_argument("--input-dir", default="scored", help="каталог с парой прочищенных файлов")
    ap.add_argument("--vocab-size", type=int, default=sp_cfg["vocab_size"])
    ap.add_argument("--model-type", default=sp_cfg["model_type"], choices=["bpe", "unigram"])
    ap.add_argument("--input-sentence-size", type=int, default=sp_cfg["input_sentence_size"],
                    help="сколько предложений максимум используется при обучении")
    ap.add_argument("--no-shuffle", action="store_true", help="не перемешивать вход при обучении")
    ap.add_argument("--hard-vocab-limit", action="store_true", default=sp_cfg["hard_vocab_limit"],
                    help="жёстко ограничить размер словаря (иначе SPM может дать чуть меньше)")
    ap.add_argument("--normalization", default=sp_cfg["normalization_rule_name"],
                    help="правило нормализации (identity, nfc, nfd, nfk, nfkd); "
                         "Windows-колесо содержит только identity")
    ap.add_argument("--normalization-rule-file", default=None,
                    help="файл с собственными правилами нормализации вместо встроенного правила")
    ap.add_argument("--character-coverage", type=float, default=0.999)
    ap.add_argument("--review-lines", type=int, default=5)
    args = ap.parse_args()

    src_file, tgt_file = corpus_files(cfg, path(cfg, args.input_dir))
    pairs = read_pair(src_file, tgt_file)
    src_lines = [p[0] for p in pairs]
    tgt_lines = [p[1] for p in pairs]
    print(f"Корпус для SPM: {len(pairs)} пар из {src_file.name} / {tgt_file.name}")

    spm_dir = path(cfg, "spm")
    data_spm = path(cfg, "spm_data")

    name = cfg["corpus"]["name"]
    tag = f"{name}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"

    report = [
        f"Корпус: {tag}",
        f"Пар: {len(pairs)}",
        f"Параметры обучения: model_type={args.model_type} vocab_size={args.vocab_size} "
        f"input_sentence_size={args.input_sentence_size} shuffle={not args.no_shuffle} "
        f"hard_vocab_limit={args.hard_vocab_limit} normalization={args.normalization} "
        f"character_coverage={args.character_coverage}",
        "",
    ]

    src_model = spm_dir / f"{name}.{cfg['corpus']['src']}"
    tgt_model = spm_dir / f"{name}.{cfg['corpus']['tgt']}"

    print(f"Обучение модели для входной части ({cfg['corpus']['src']})…")
    sp_src = train_side(src_file, src_model, args)
    print(f"Обучение модели для выходной части ({cfg['corpus']['tgt']})…")
    sp_tgt = train_side(tgt_file, tgt_model, args)

    src_piece = data_spm / f"{tag}.{cfg['corpus']['src']}.piece"
    tgt_piece = data_spm / f"{tag}.{cfg['corpus']['tgt']}.piece"
    tokenize(sp_src, src_lines, src_piece)
    tokenize(sp_tgt, tgt_lines, tgt_piece)

    src_model_file = Path(src_model.as_posix() + ".model")
    src_vocab_file = Path(src_model.as_posix() + ".vocab")
    tgt_model_file = Path(tgt_model.as_posix() + ".model")
    tgt_vocab_file = Path(tgt_model.as_posix() + ".vocab")

    report.append(f"Модель входной части: {src_model_file.name}, "
                  f"словарь: {src_vocab_file.name}")
    report.extend(side_stats(sp_src, src_lines, args.review_lines))
    report.append("")
    report.append(f"Модель выходной части: {tgt_model_file.name}, "
                  f"словарь: {tgt_vocab_file.name}")
    report.extend(side_stats(sp_tgt, tgt_lines, args.review_lines))
    report.append("")
    report.append(f"Разбитые корпуса: {src_piece.name}, {tgt_piece.name}")
    report.append(f"Число строк совпадает с корпусом: {len(src_lines)} пар")

    logs = path(cfg, "logs")
    (logs / f"{tag}_03_spm_stats.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"Сводка: {logs / f'{tag}_03_spm_stats.txt'}")


if __name__ == "__main__":
    main()
