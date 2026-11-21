"""Общая конфигурация пайплайна прочистки параллельного корпуса.

Все скрипты читают config.json (в корне проекта) и могут переопределять
значения ключами командной строки. Конфигурация описывает:

  corpus   — имя корпуса, направления (src/tgt) и путь к сырым файлам
  clean    — пороги прочистки по 4 правилам ТЗ (задача 1)
  score    — модель эмбеддингов и порог скоринга (задача 2)
  spm      — параметры обучения SentencePiece (задача 3)
  align    — путь к бинарям fast_align и параметры выравнивания (задача 4)

Каталоги: data/raw (вход), data/steps (промежуточные корпуса по этапам),
data/clean (прочистка по задаче 1), data/scored (прочистка по скору),
spm/ (модели и куски), align/ (выравнивание), logs/ (логи отброшенных строк).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS = {
    "corpus": {
        "name": "Tatoeba",
        "src": "en",
        "tgt": "ru",
        "raw_src": "data/raw/Tatoeba.en-ru.en",
        "raw_tgt": "data/raw/Tatoeba.en-ru.ru",
    },
    "clean": {
        "min_words": 3,
        "max_chars": 800,
        "len_ratio": 2.0,
        "alpha_ratio": 0.8,
        "alpha_ignore_space": True,
        "alpha_ignore_digits": False,
        "dedup": False,
        "src_lang": "en",
        "tgt_lang": "ru",
        "low_memory_langdetect": True,
    },
    "score": {
        "model": "intfloat/multilingual-e5-large",
        "batch_size": 192,
        "max_seq_len": 0,
        "use_cuda": True,
        "threshold": None,
        "target_drop": 0.12,
    },
    "spm": {
        "model_type": "bpe",
        "vocab_size": 32000,
        "input_sentence_size": 100000,
        "shuffle": True,
        "hard_vocab_limit": False,
        "normalization_rule_name": "identity",
    },
    "align": {
        "fast_align_dir": "tools/fast_align/build",
        "diagonal": "grow-diag-final-and",
        "passes": 3,
    },
    "sample": {
        "size": 3000,
        "seed": 777,
    },
    "paths": {
        "raw": "data/raw",
        "steps": "data/steps",
        "clean": "data/clean",
        "scored": "data/scored",
        "spm": "spm",
        "spm_data": "data/spm",
        "align": "align",
        "rand": "rand",
        "logs": "logs",
    },
}


def load_config() -> dict:
    """Читает config.json поверх значений по умолчанию."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if CONFIG_PATH.exists():
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for key, val in user.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def path(cfg: dict, key: str) -> Path:
    """Каталог из конфига, относительно корня проекта."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_files(cfg: dict, directory: str) -> tuple[Path, Path]:
    """Пути парных файлов (src/tgt) в указанном каталоге."""
    name = cfg["corpus"]["name"]
    src = cfg["corpus"]["src"]
    tgt = cfg["corpus"]["tgt"]
    base = Path(directory)
    if not base.is_absolute():
        base = ROOT / base
    candidates = [
        base / f"{name}.{src}-{tgt}.{src}",
        base / f"{name}.{src}.{src}",
        base / f"{src}.{src}",
        base / "src.txt",
    ]
    src_path = next((c for c in candidates if c.exists()), None)
    tgt_candidates = [
        base / f"{name}.{src}-{tgt}.{tgt}",
        base / f"{name}.{src}.{tgt}",
        base / f"{tgt}.{tgt}",
        base / "tgt.txt",
    ]
    tgt_path = next((c for c in tgt_candidates if c.exists()), None)
    if src_path is None or tgt_path is None:
        raise FileNotFoundError(
            f"Нет пары файлов в {base}. Нужны {name}.{src}-{tgt}.{src} и {name}.{src}-{tgt}.{tgt}"
        )
    return src_path, tgt_path


def out_pair(cfg: dict, directory: str, suffix: str = "") -> tuple[Path, Path]:
    """Пути для записи парных файлов (src/tgt) в указанный каталог."""
    name = cfg["corpus"]["name"]
    src = cfg["corpus"]["src"]
    tgt = cfg["corpus"]["tgt"]
    base = path(cfg, directory)
    tag = f"{name}.{src}-{tgt}"
    return base / f"{tag}.{src}{suffix}", base / f"{tag}.{tgt}{suffix}"


def read_pair(src: Path, tgt: Path) -> list[tuple[str, str]]:
    """Построчная читка параллельной пары с проверкой синхронности."""
    with src.open(encoding="utf-8") as a, tgt.open(encoding="utf-8") as b:
        lines_a = a.read().splitlines()
        lines_b = b.read().splitlines()
    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"Параллельность нарушена: {src.name}={len(lines_a)} строк, "
            f"{tgt.name}={len(lines_b)} строк"
        )
    return list(zip(lines_a, lines_b))


def write_pair(pairs: list[tuple[str, str]], src_out: Path, tgt_out: Path) -> None:
    """Построчная запись пары; переводы строк в обоих файлах одинаковые."""
    with src_out.open("w", encoding="utf-8", newline="\n") as a, tgt_out.open(
        "w", encoding="utf-8", newline="\n"
    ) as b:
        for s, t in pairs:
            a.write(s + "\n")
            b.write(t + "\n")


def log_error(errors: list[tuple[str, str, str]], log_file: Path) -> None:
    """Лог отброшенных строк в формате ТЗ: ТИП / src / tgt."""
    with log_file.open("w", encoding="utf-8", newline="\n") as f:
        for typ, s, t in errors:
            f.write(f"{typ}\n{s}\n{t}\n")


def count_lines(p: Path) -> int:
    with p.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def hr(msg: str) -> None:
    print(f"[{msg}]", flush=True)


def require_hf_token() -> None:
    """Проверяет наличие токена Hugging Face (нужен для гейтованных моделей)."""
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        sys.stderr.write(
            "Нет токена HF. Для гейтованных моделей (Gemma/COMET) выполните один раз:\n"
            "  .venv/Scripts/python.exe -m pip install huggingface_hub\n"
            "  .venv/Scripts/huggingface-cli.exe login\n"
            "или задайте переменную окружения HF_TOKEN.\n"
        )


def detect_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
