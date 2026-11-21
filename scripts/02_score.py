"""Задача 2 (часть 1): скоринг параллельности мультиязычными эмбеддингами.

Для каждой пары берём векторы входного и выходного предложения одной
мультиязычной моделью (по умолчанию intfloat/multilingual-e5-large; в ТЗ
предложена google/embeddinggemma-300m) и считаем косинусное сходство.
Это и есть скор параллельности.

Выход:
  data/scored/<corpus>.<src>-<tgt>.scores.tsv — скор \t вход \t выход (скор первой колонкой)
  data/scored/<corpus>.<src>-<tgt>.<src>_all / .<tgt>_all — тот же корпус без вырезки
  logs/<corpus>.<src>-<tgt>_02_score_summary.txt — параметры прогона и сводка по скору

Запуск:
  python scripts/02_score.py                    # корпус из data/clean
  python scripts/02_score.py --model google/embeddinggemma-300m
  python scripts/02_score.py --cpu              # если без GPU
  python scripts/02_score.py --max-seq-len 128  # короче и меньше VRAM на GPU
  python scripts/02_score.py --from-scores data/scored/Tatoeba.en-ru.scores.tsv
      # взять готовые скоры из tsv и только пересчитать сводку (без загрузки модели)

embeddinggemma гейтованная (лицензия Gemma): нужен аккаунт HF, согласие с лицензией
и токен: huggingface-cli login или переменная окружения HF_TOKEN.
Модель intfloat/multilingual-e5-large (MIT) токена не требует.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    corpus_files,
    load_config,
    out_pair,
    path,
    read_pair,
    require_hf_token,
    write_pair,
)


def encode(model, texts: list[str], batch_size: int, device: str, prompt: str = "") -> np.ndarray:
    """Эмбеддинги батчами; prompt дописывается к каждой строке (нужно для e5-моделей)."""
    if prompt:
        try:
            vecs = model.encode(texts, batch_size=batch_size, device=device,
                                show_progress_bar=True, prompt=prompt)
        except TypeError:
            vecs = model.encode([prompt + t for t in texts], batch_size=batch_size,
                                device=device, show_progress_bar=True)
    else:
        vecs = model.encode(texts, batch_size=batch_size, device=device, show_progress_bar=True)
    return np.asarray(vecs, dtype=np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / (na * nb)


def report_summary(cfg, tag: str, scores: np.ndarray, model: str, device: str,
                   batch_size: int, scores_file: Path) -> None:
    """Сводка этапа: параметры прогона и статистика скора (min/max/mean/квантили)."""
    logs = path(cfg, "logs")
    q = np.quantile(scores, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    summary = [
        f"Модель эмбеддингов: {model}",
        f"Устройство: {device}, batch_size={batch_size}",
        f"Пар оценено: {len(scores)}",
        f"Скор: min={scores.min():.3f} max={scores.max():.3f} "
        f"mean={float(scores.mean()):.3f} std={scores.std():.3f}",
        "Квантили (1,10,25,50,75,90,99%): " + ", ".join(f"{v:.3f}" for v in q),
        f"Файл скоров: {scores_file.name}",
        "Следующий шаг: python scripts/02_filter.py (анализ распределения и прочистка по порогу)",
    ]
    out = logs / f"{tag}_02_score_summary.txt"
    out.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Сводка: {out}")


def save_all_pair(cfg, pairs: list[tuple[str, str]]) -> None:
    """Сохраняет пару «как есть», чтобы 02_filter работал на ней же."""
    clean_src, clean_tgt = out_pair(cfg, "scored", "_all")
    write_pair(pairs, clean_src, clean_tgt)


def read_scores_tsv(scores_file: Path) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Читает scores.tsv (скор \t вход \t выход)."""
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


def main() -> None:
    cfg = load_config()
    s = cfg["score"]
    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    ap = argparse.ArgumentParser(description="скоринг параллельности эмбеддингами")
    ap.add_argument("--input-dir", default="clean", help="каталог с прочищенной парой")
    ap.add_argument("--model", default=s["model"])
    ap.add_argument("--batch-size", type=int, default=s["batch_size"])
    ap.add_argument("--cpu", action="store_true", help="не использовать GPU")
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="обрезать длину последовательности эмбеддера (0 = максимум модели); "
                         "для коротких предложений это заметно ускоряет и экономит VRAM")
    ap.add_argument("--from-scores", default="",
                    help="не считать эмбеддинги, а взять скоры из готового .tsv и только "
                         "пересписать сводку (числа те же, GPU не нужен)")
    ap.add_argument("--device-label", default="",
                    help="строка про устройство для сводки при --from-scores "
                         "(например 'cuda (RTX 5070), сводка из готового прогона')")
    args = ap.parse_args()

    scores_file = path(cfg, "scored") / f"{tag}.scores.tsv"

    if args.from_scores:
        src_tsv = Path(args.from_scores)
        if not src_tsv.exists():
            raise SystemExit(f"Нет файла скоров: {src_tsv}")
        scores, pairs = read_scores_tsv(src_tsv)
        device = args.device_label or "cpu (сводка по готовому файлу скоров)"
        print(f"Скоры взяты из готового файла: {src_tsv.name} ({len(pairs)} пар)")
        save_all_pair(cfg, pairs)
        report_summary(cfg, tag, scores, args.model, device, args.batch_size, scores_file)
        return

    src_file, tgt_file = corpus_files(cfg, path(cfg, args.input_dir))
    pairs = read_pair(src_file, tgt_file)
    print(f"Пар для оценки: {len(pairs)} из {src_file.name} / {tgt_file.name}")

    require_hf_token()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Модель: {args.model}, устройство: {device}")
    model = SentenceTransformer(args.model, device=device)
    if args.max_seq_len:
        model.max_seq_length = args.max_seq_len
        print(f"max_seq_length: {args.max_seq_len}")

    # Для e5-моделей симметричное сходство считается с одинаковым префиксом у обеих сторон.
    prompt = "query: " if "e5" in args.model.lower() else ""
    if prompt:
        print(f"Префикс к строкам: {prompt!r}")

    started = time.time()
    src_emb = encode(model, [p[0] for p in pairs], args.batch_size, device, prompt)
    tgt_emb = encode(model, [p[1] for p in pairs], args.batch_size, device, prompt)
    scores = cosine_rows(src_emb, tgt_emb)
    print(f"Эмбеддинги и скоры получены за {time.time() - started:.0f} c")

    with scores_file.open("w", encoding="utf-8", newline="\n") as f:
        for (src, tgt), sc in zip(pairs, scores):
            f.write(f"{sc:.4f}\t{src}\t{tgt}\n")

    print("Скоры:", scores_file)

    # сохраняем и прочищенную пару «как есть», чтобы 02_filter работал на ней же
    save_all_pair(cfg, pairs)

    report_summary(cfg, tag, scores, args.model, device, args.batch_size, scores_file)


if __name__ == "__main__":
    main()
