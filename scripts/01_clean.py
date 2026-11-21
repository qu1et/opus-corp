"""Задача 1: прочистка параллельного корпуса по четырем правилам ТЗ.

Правила применяются последовательно, каждое удаляет пару целиком (параллельность
не нарушается), и каждый этап пишет:

  data/steps/step<N>_<corpus>.<src> / .<tgt>  — корпус после этапа
  logs/<код>.log                              — отброшенные пары в формате ТЗ:
      <ТИП_ОШИБКИ>
      <строка на входном языке>
      <строка на выходном языке>
  logs/01_summary.txt                         — сводка по этапам и параметры

Этапы:
  SHORT      — в одной из строк меньше min_words слов (по умолчанию 3, т.е. режем 1-2 слова)
  LEN_RATIO  — одна строка длиннее другой в len_ratio раз (по умолчанию 2) по символам
  ALPHA      — доля алфавитных символов ниже alpha_ratio (по умолчанию 0.8)
  LANG       — fasttext-langdetect: вход распознан не как src, или выход не как tgt

Дополнительные фильтры (не обязательны по ТЗ, включаются ключами):
  MAX_CHARS  — слишком длинные строки (0 = не применять)
  DUP        — полностью повторяющиеся пары

Знаменатель доли алфавитных символов по умолчанию считает только nonspace-символы.
Если считать по всем символам вместе с пробелами и пунктуацией, в лог массово
выписываются нормальные строки (на Tatoeba en-ru: 390k английских строк вместо 3.7k) —
это тот случай, который ТЗ разрешает лечить изменением параметров.

Запуск:
  python scripts/01_clean.py                     # параметры из config.json
  python scripts/01_clean.py --min-words 3 --alpha-ratio 0.75 --cyrillic-alpha
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT,
    corpus_files,
    load_config,
    log_error,
    out_pair,
    path,
    read_pair,
    write_pair,
)

CYRILLIC = {chr(c) for c in range(1040, 1104)} | {"ё"}


def alpha_ratio(text: str, ignore_space: bool, alphabet: set[str] | None = None,
                ignore_digits: bool = False) -> float:
    """Доля алфавитных символов; alphabet задаёт узкий алфавит (например кириллицу).

    ignore_digits убирает цифры из знаменателя: иначе строки вида «This will cost €30.»
    отбраковываются только потому, что в них есть числа.
    """
    chars = [c for c in text if not (ignore_space and c.isspace())]
    if ignore_digits:
        chars = [c for c in chars if not c.isdigit()]
    if not chars:
        return 0.0
    if alphabet is not None:
        return sum(1 for c in chars if c in alphabet) / len(chars)
    return sum(1 for c in chars if c.isalpha()) / len(chars)


# Зеркала модели: fbaipublicfiles отдаёт файл очень медленно (сотни байт в секунду),
# поэтому сначала пробуем HF-репозитории, которые кладут тот же lid.176.
MODEL_MIRRORS = {
    "lid.176.bin": [
        "https://huggingface.co/arc-r/fasttext-language-identification/resolve/main/lid.176.bin",
        "https://huggingface.co/crash-sv/scribe-fasttext-lid176/resolve/main/lid.176.bin",
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    ],
    "lid.176.ftz": [
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
    ],
}


def ensure_lang_model(low_memory: bool, cache_dir: Path) -> None:
    """Скачивает модель lid.176 (.ftz или .bin) в кэш, если её там нет."""
    name = "lid.176.ftz" if low_memory else "lid.176.bin"
    target = cache_dir / name
    if target.exists() and target.stat().st_size > 100_000:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    ok = False
    for url in MODEL_MIRRORS[name]:
        print(f"Скачиваю модель определения языка: {url}")
        res = subprocess.run(
            ["curl.exe", "-s", "-L", "--max-redirs", "5", "--retry", "2", "--retry-connrefused",
             "--max-time", "300", "-o", tmp.as_posix(), url],
            capture_output=True, text=True, encoding="utf-8",
        )
        if res.returncode == 0 and tmp.exists() and tmp.stat().st_size > 100_000:
            ok = True
            break
        print("  зеркало не сработало")
        tmp.unlink(missing_ok=True)
    if not ok:
        raise SystemExit(
            f"Не удалось скачать {name}. Скачайте вручную в {cache_dir}:\n"
            "  curl.exe -L -o "
            f"{target} https://huggingface.co/arc-r/fasttext-language-identification/resolve/main/lid.176.bin\n"
            "или запустите setup.ps1 (он prefetch-ит модель), затем повторите этап 1.\n"
            "Можно пропустить этап языка: python scripts/01_clean.py --no-lang"
        )
    shutil.move(tmp.as_posix(), target.as_posix())
    print(f"Модель в кэше: {target.name} ({target.stat().st_size / 1e6:.1f} МБ)")


def init_langdetect(low_memory: bool):
    """Возвращает функцию detect(text) -> код языка; модель кэшируется в .cache."""
    cache = ROOT / ".cache"
    os.environ["FTLANG_CACHE"] = str(cache)
    ensure_lang_model(low_memory, cache)
    from ftlangdetect import detect

    def one(text: str) -> str:
        if not text.strip():
            return ""
        return detect(text=text, low_memory=low_memory)["lang"]

    return one


def main() -> None:
    cfg = load_config()
    c = cfg["clean"]
    ap = argparse.ArgumentParser(description="прочистка параллельного корпуса (задача 1)")
    ap.add_argument("--min-words", type=int, default=c["min_words"])
    ap.add_argument("--max-chars", type=int, default=c["max_chars"],
                    help="максимум символов в строке; 0 — не применять")
    ap.add_argument("--len-ratio", type=float, default=c["len_ratio"])
    ap.add_argument("--alpha-ratio", type=float, default=c["alpha_ratio"])
    ap.add_argument("--all-chars", action="store_true",
                    help="считать долю алфавитных символов по всем символам, включая пробелы")
    ap.add_argument("--alpha-ignore-digits", action="store_true", default=c.get("alpha_ignore_digits", False),
                    help="убрать цифры из знаменателя доли алфавитных символов")
    ap.add_argument("--cyrillic-alpha", action="store_true",
                    help="для выходной строки считать только кириллические буквы")
    ap.add_argument("--no-lang", action="store_true", help="пропустить этап определения языка")
    ap.add_argument("--dedup", action="store_true", default=c["dedup"])
    ap.add_argument("--low-memory", action="store_true", default=c["low_memory_langdetect"],
                    help="сжатая модель lid.176.ftz (быстрее и меньше памяти)")
    args = ap.parse_args()

    src_file, tgt_file = corpus_files(cfg, path(cfg, "raw"))
    pairs = read_pair(src_file, tgt_file)
    total_in = len(pairs)
    print(f"Вход: {src_file.name} + {tgt_file.name} = {total_in} пар")

    tag = f"{cfg['corpus']['name']}.{cfg['corpus']['src']}-{cfg['corpus']['tgt']}"
    ignore_space = not args.all_chars
    tgt_alphabet = CYRILLIC if args.cyrillic_alpha else None
    logs_dir = path(cfg, "logs")
    summary: list[str] = []

    def log_name(code: str) -> str:
        """Имя лога с меткой корпуса: в одном каталоге логи разных корпусов не перезаписываются."""
        return f"{tag}_{code}"

    if args.dedup:
        seen: set[tuple[str, str]] = set()
        kept: list[tuple[str, str]] = []
        dropped: list[tuple[str, str, str]] = []
        for s, t in pairs:
            if (s, t) in seen:
                dropped.append(("DUP", s, t))
            else:
                seen.add((s, t))
                kept.append((s, t))
        log_error(dropped, logs_dir / log_name("DUP.log"))
        print(f"DUP: {total_in} -> {len(kept)} (удалено {len(dropped)})")
        summary.append(f"DUP дубли           вход={total_in:>7} осталось={len(kept):>7} удалено={len(dropped):>6}")
        pairs = kept

    def run_stage(code: str, label: str, predicate) -> list[tuple[str, str]]:
        kept: list[tuple[str, str]] = []
        dropped: list[tuple[str, str, str]] = []
        for s, t in pairs:
            typ = predicate(s, t)
            if typ:
                dropped.append((typ, s, t))
            else:
                kept.append((s, t))
        so, to = out_pair(cfg, "steps", f"_{code}")
        write_pair(kept, so, to)
        log_error(dropped, logs_dir / log_name(f"{code}.log"))
        print(f"{code}: {len(pairs)} -> {len(kept)} (удалено {len(dropped)})")
        summary.append(f"{code} {label:<14} вход={len(pairs):>7} осталось={len(kept):>7} удалено={len(dropped):>6}")
        return kept

    def check_short(s: str, t: str) -> str | None:
        if len(s.split()) < args.min_words or len(t.split()) < args.min_words:
            return "SHORT"
        if args.max_chars and (len(s) > args.max_chars or len(t) > args.max_chars):
            return "MAX_CHARS"
        return None

    def check_len_ratio(s: str, t: str) -> str | None:
        ls, lt = len(s), len(t)
        if ls == 0 or lt == 0:
            return "LEN_RATIO"
        if ls >= args.len_ratio * lt or lt >= args.len_ratio * ls:
            return "LEN_RATIO"
        return None

    def check_alpha(s: str, t: str) -> str | None:
        rs = alpha_ratio(s, ignore_space, None, args.alpha_ignore_digits)
        rt = alpha_ratio(t, ignore_space, tgt_alphabet, args.alpha_ignore_digits)
        if rs < args.alpha_ratio and rt < args.alpha_ratio:
            return "ALPHA_BOTH"
        if rs < args.alpha_ratio:
            return "ALPHA_SRC"
        if rt < args.alpha_ratio:
            return "ALPHA_TGT"
        return None

    pairs = run_stage("SHORT", "число слов", check_short)
    pairs = run_stage("LEN_RATIO", "разница длин", check_len_ratio)
    pairs = run_stage("ALPHA", "алфавитные символы", check_alpha)

    if not args.no_lang:
        detect_lang = init_langdetect(args.low_memory)
        want_src, want_tgt = c["src_lang"], c["tgt_lang"]

        def check_lang(s: str, t: str) -> str | None:
            ls = detect_lang(s)
            lt = detect_lang(t)
            if ls != want_src and lt != want_tgt:
                return "LANG_BOTH"
            if ls != want_src:
                return f"LANG_SRC({ls})"
            if lt != want_tgt:
                return f"LANG_TGT({lt})"
            return None

        pairs = run_stage("LANG", f"язык {want_src}/{want_tgt}", check_lang)

    clean_src, clean_tgt = out_pair(cfg, "clean")
    write_pair(pairs, clean_src, clean_tgt)

    report = [
        f"Корпус: {cfg['corpus']['name']} {cfg['corpus']['src']} -> {cfg['corpus']['tgt']}",
        f"Пар на входе: {total_in}",
        f"Пар на выходе: {len(pairs)}",
        f"Параметры: min_words={args.min_words} max_chars={args.max_chars} "
        f"len_ratio={args.len_ratio} alpha_ratio={args.alpha_ratio} "
        f"alpha_ignore_space={ignore_space} alpha_ignore_digits={args.alpha_ignore_digits} "
        f"cyrillic_alpha={args.cyrillic_alpha} "
        f"lang={c['src_lang']}/{c['tgt_lang']} lang_model_low_memory={args.low_memory} dedup={args.dedup}",
        "",
        *summary,
        "",
        f"Финальный корпус: {clean_src.name}, {clean_tgt.name}",
        f"Промежуточные корпуса: каталог {path(cfg, 'steps')}",
        f"Логи: каталог {logs_dir}",
    ]
    (logs_dir / log_name("01_summary.txt")).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Финальный прочищенный корпус: {clean_src.name} / {clean_tgt.name}")
    print(f"Сводка: {logs_dir / log_name('01_summary.txt')}")


if __name__ == "__main__":
    main()
