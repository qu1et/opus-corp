# Пайплайн прочистки параллельного корпуса (практические задания 1-5)

Направление по умолчанию: **en → ru**, корпус **OPUS Tatoeba v2026-07-08**
(589 883 пар, файл 12.7 МБ). Всё настраивается в `config.json`, каждый скрипт
переопределяет его ключами командной строки.

## Один раз: установка

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1          # venv + pip + torch (CUDA)
powershell -ExecutionPolicy Bypass -File setup.ps1 -Tools   # плюс MSYS2/MinGW и сборка fast_align
```

`setup.ps1 -Tools` ставит MSYS2 (winget), `gcc`, `cmake`, `mingw32-make` и собирает
`fast_align.exe` и `atools.exe` в `tools/fast_align/build` (задача 4). Это уже сделано
на этой машине.

## Загрузить свой корпус

Вариант 1 — свои два файла (входной язык и выходной, построчно параллельные, UTF-8):

```powershell
python scripts\00_fetch_corpus.py --files path\my.en path\my.ru --name MyCorpus --src en --tgt ru
```

Вариант 2 — корпус с OPUS:

```powershell
python scripts\00_fetch_corpus.py --corpus Tatoeba --version v2026-07-08 --pair en-ru
python scripts\00_fetch_corpus.py --probe en-ru    # что доступно для пары и какого размера
```

Файлы складываются в `data/raw` под имена `Имя.src-tgt.src` и `Имя.src-tgt.tgt`.
Скрипт сверяет число строк в обеих частях и ругается, если параллельность не сходится.
Другое направление/имя задаётся через `--name/--src/--tgt` (или правкой `config.json`).

Хост OPUS (`object.pouta.csc.fi`) в момент подготовки не отвечал (таймаут соединения),
поэтому скачивание с OPUS может не пройти — тогда используйте `--files` со своими
корпусами. Корпус Tatoeba уже скачан ранее и лежит в `data/raw`, так что весь пайплайн
работает и без сети. `python scripts\00_fetch_corpus.py --probe en-ru` показывает,
доступен ли хост и какого размера файлы.

## Запустить всё

```powershell
python scripts\run_all.py                      # этапы 1-6 на корпусе из config.json (6 = отчёт report.md)
powershell -ExecutionPolicy Bypass -File run_all.ps1
python scripts\run_all.py --stages 1           # только прочистка по правилам ТЗ
python scripts\run_all.py --stages 1,2         # прочистка + скоринг
python scripts\run_all.py --stages 6           # перегенерировать отчёт из логов
python scripts\run_all.py --raw-src my.en --raw-tgt my.ru --name MyCorpus   # свой корпус целиком
python scripts\run_all.py --stages 0 --corpus Tatoeba --version v2026-07-08 --pair en-ru  # скачать корпус
python scripts\run_all.py --stages 4 --align-input pieces   # выравнивание на кусках SPM
python scripts\run_all.py --stages 2 --score-model google/embeddinggemma-300m   # вариант из ТЗ
python scripts\run_all.py --stages 2 --threshold 0.65       # явный порог скоринга
python scripts\run_all.py --stages 2 --percentile 10        # порог = 10-й процентиль скоров
python scripts\run_all.py --stages 1 --alpha-ignore-digits  # цифры не считаются в знаменателе доли
python scripts\run_all.py --stages 5 --sample-size 5000 --seed 123   # размер случайной части
```

Модель эмбеддингов по умолчанию — `intfloat/multilingual-e5-large` (лицензия MIT, без
гейта HF), чтобы пайплайн запускался без токена; для e5 к обеим сторонам пары
добавляется префикс `query: ` (требование модели для симметричного сходства).
Вариант из ТЗ `google/embeddinggemma-300m` требует аккаунт HF, согласие с лицензией
Gemma и токен: `huggingface-cli login` или переменная `HF_TOKEN`:

```powershell
python scripts\run_all.py --stages 2 --score-model google/embeddinggemma-300m
```

Альтернативный скоринг по ТЗ — `fast_align` с флагом `-s` (оценка в обе стороны,
финальный скор = среднее двух значений; метод требователен к памяти и не годится для
малых корпусов):

```powershell
python scripts\run_all.py --stages 2 --score-method fastalign --percentile 10
```

Скор у fast_align — логарифм правдоподобия выравнивания, он отрицательный: чем ближе к
нулю, тем лучше выравнивание. Поэтому `02_filter.py` для таких скоров строит гистограмму
по фактическому диапазону и таблицу порогов по квантилям, а порог задаётся через
`--percentile` или явный `--threshold` (например `-2.45`). Эмбеддинговые скоры лежат в
0..1, там пороги обычные (0.75 и т.п.).

`02_score_fastalign.py` пишет два файла: `*.fastalign.scores.tsv` (скор, делённый на число
токенов пары — так порог одинаково работает для коротких и длинных предложений) и
`*.fastalign.rawscores.tsv` (исходный лог-правдоподобие). Анализ скоров от разных
инструментов не перезаписывает выводы другого: для fast_align используйте
`--source-label fastalign`, тогда логи `logs/<tag>_fastalign_02_*.txt/log`.

Модель `lid.176.bin` для определения языка качается в `.cache/` при первом этапе
(зеркала: HF `arc-r/fasttext-language-identification`, `crash-sv/scribe-fasttext-lid176`,
`dl.fbaipublicfiles.com` — последний отдаёт файл очень медленно, ~300 Б/с на этой машине).
Если скачать не удалось: `python scripts\01_clean.py --no-lang`.

## Что делает каждый этап

| этап | скрипт | вход | выход |
|---|---|---|---|
| 0 | `00_fetch_corpus.py` | OPUS или ваши файлы | `data/raw/` |
| 1 | `01_clean.py` | `data/raw` | `data/steps/*` (после каждого фильтра), `data/clean`, `logs/<tag>_SHORT.log`, `_LEN_RATIO.log`, `_ALPHA.log`, `_LANG.log`, `_01_summary.txt` |
| 2a | `02_score.py` | `data/clean` | `data/scored/*.scores.tsv` (скор + две строки), `logs/<tag>_02_score_summary.txt` |
| 2b | `02_filter.py` | скорами выше | `data/scored/<corpus>.src / .tgt`, `logs/<tag>_02_histogram.txt`, `_02_threshold_candidates.txt`, `_02_review.txt`, `_02_filtered.log` (скор в пометке), `_02_summary.txt` |
| 3 | `03_spm.py` | `data/scored` | `spm/<corpus>.src.model/.vocab`, `spm/<corpus>.tgt.model/.vocab`, `data/spm/*.piece`, `logs/<tag>_03_spm_stats.txt` |
| 4 | `04_align.py` | `data/scored` или `data/spm/*.piece` | `align/<corpus>.src-tgt.txt` (`вход \|\|\| выход`), `<tag>.forward.align`, `<tag>.reverse.align`, `<tag>.align`, `logs/<tag>_04_align_summary.txt` |
| 5 | `05_get_rand_sample.py` | пара файлов + `.align` (и опционально `.piece`) | `rand/*.rand`, `rand/rand_indexes.txt`, `logs/ALL_summary.txt` |

где `<tag>` = `<имя корпуса>.<src>-<tgt>` (для текущего корпуса `Tatoeba.en-ru`). Логи и
сводки называются с этой меткой, поэтому в одном каталоге артефакты разных корпусов
не перезаписывают друг друга; `config.json` описывает один активный корпус.

Формат лога отброшенных строк — как в ТЗ:

```
<ТИП_ОШИБКИ>
<строка на входном языке>
<строка на выходном языке>
```

Для скоринга пометка содержит скор: `SCORE_LOW 0.5312`.

## Параметры прочистки и почему они такие

- `min_words=3`: удаляет пары, где в одной из строк 1-2 слова (пункт 1 ТЗ).
- `len_ratio=2.0`: разница длин в символах более чем в два раза (пункт 2 ТЗ).
- `alpha_ratio=0.8` при знаменателе **без пробелов** (пункт 3 ТЗ). Если считать долю по всем
  символам вместе с пробелами и пунктуацией (`--all-chars`), на Tatoeba отбраковывается
  389 918 английских и 201 352 русских строк — в лог массово выписываются нормальные
  предложения с апострофами и кавычками. С nonspace-знаменателем отбраковка 3 663 en и
  2 106 ru, то есть только реально подозрительные строки. ТЗ прямо разрешает менять
  параметры, когда лог выписывает хорошие строки — это тот случай.
- `--cyrillic-alpha`: для выходной строки считать только кириллические буквы
  (`chr(1040)..chr(1103)` плюс `ё`), если нужна отдельная обработка кириллицы.
- язык: `fasttext-langdetect` (модель `lid.176`, кэш в `.cache/`), вход должен быть
  распознан как `src`, выход как `tgt`.
- необязательные: `--max-chars 800`, `--dedup`.

Порог скоринга выбирается по распределению: `02_filter.py` пишет таблицу порогов
(сколько пар уходит при каждом) и файл `02_review.txt` с парами около границы —
по нему решаем, где проходит граница качества, и фиксируем порог в `02_summary.txt`.

На этом корпусе (Tatoeba en-ru, прочищенный по правилам ТЗ: 554 043 пар) скоры
лежат узко: min 0.692, mean 0.930, квантили 0.855/0.903/0.922/0.935/0.944/0.951/0.959.
Ниже 0.75 находятся только реально не-переводы (панграммы вроде «The quick brown fox…»
против «В чащах юга жил бы цитрус?», «When does Tony study?» против «Давайте сядем на
скамейку»), в диапазоне 0.75-0.85 — идиоматические эквиваленты («Too many cooks spoil
the broth» ↔ «У семи нянек дитя без глазу»), которые надо оставить. Поэтому порог
выбран 0.75 и записан в `config.json` (`score.threshold`): уходит 11 пар (0.00%).

`--auto --target-drop 0.12` на таком корпусе режет 66 738 пар и выписывает в лог
массово хорошие строки — это ровно тот случай, который ТЗ разрешает лечить изменением
параметров; для чистых корпусов задавайте порог явно или берите маленький процентиль
(`--percentile 0.02`).

## Задача 5 отдельно

Скрипт работает ровно как в примере из ТЗ, аргументы позиционные:

```powershell
python scripts\05_get_rand_sample.py 554788 3000 data\scored\Tatoeba.en-ru.en data\scored\Tatoeba.en-ru.ru align\Tatoeba.en-ru.align
```

`argv[1]` — число строк во входном корпусе, `argv[2]` — размер случайной части,
`argv[3:]` — файлы (от одного и больше). Один и тот же набор индексов применяется ко всем
файлам, поэтому параллельность сохраняется. На выход — те же имена плюс `.rand`.
Дополнительно: `--seed` (воспроизводимость), `--out-dir`, `--no-indexes`.

Скрипт сохраняет `rand/rand_indexes.txt` — список индексов выборки. По нему сверяется
параллельность (в корпусе есть повторяющиеся строки, поэтому поиск по содержимому
ненадёжен):

```powershell
python scripts\check_rand_parallelism.py            # indexes + .rand + исходные файлы
python scripts\check_align.py                       # формат выравнивания и несколько пар глазами
```

## Проверка готового прогона на Tatoeba en-ru

Все этапы выполнены на этой машине, numbers из логов:

| этап | результат |
|---|---|
| сырой корпус | 589 883 пар |
| после SHORT (min_words=3) | 560 439, удалено 29 444 |
| после LEN_RATIO (2.0) | 558 127, удалено 2 312 |
| после ALPHA (0.8, без пробелов) | 554 776, удалено 3 351 |
| после LANG (fasttext en/ru) | 554 043, удалено 733 |
| скоринг (e5-large, косинус) | min 0.692, mean 0.930, std 0.021 |
| прочистка по скору (порог 0.75) | 554 032, удалено 11 пар (0.00%) |
| альтернативный скоринг (fast_align -s, нормализация на токены) | min -4.60, mean -1.74, квантили -2.86/-2.26/-1.97/-1.69/-1.46/-1.28/-1.01 |
| альтернативная прочистка по скору (процентиль 5, report-only) | порог -2.4575, удалено бы 27 694 пар (5.0%) |
| SentencePiece (bpe, vocab 32000, две модели) | avg 8.87 кусков на en-строку, 7.96 на ru-строку |
| fast_align (forward + reverse, grow-diag-final-and) | 554 032 строк выравнивания |
| случайная часть (seed 777, 3000 строк) | 3 файла по 3000 строк, параллельность проверена по индексам — 0 рассинхронов |

Итоговый отчёт по всем этапам: `report.md`, сводка артефактов: `logs/ALL_summary.txt`.


## Корпус и лицензии

Корпус `OPUS Tatoeba v2026-07-08`, предпроцессор `moses`, пара `en-ru`: 589 883 пар,
лицензия CC BY 2.0 FR (файл `data/raw/LICENSE.Tatoeba`). Если используете OPUS,
полагается ссылка на Tiedemann, 2012, «Parallel Data, Tools and Interfaces in OPUS» (LREC).

Модели: `intfloat/multilingual-e5-large` — MIT; `google/embeddinggemma-300m` — лицензия
Gemma (нужно согласие); `Unbabel/wmt22-cometkiwi-da` — CC BY-NC 4.0 (нужно согласие);
`fast_align` — Apache 2.0; `sentencepiece` — Apache 2.0; `fasttext-langdetect` — MIT.

Свой корпус, который вы кладёте в `data/raw`, проверяется на одинаковое число строк;
лицензию/источник корпуса описывайте отдельно — пайплайн ничего не скачивает сам,
кроме модели `lid.176.bin` (в `.cache/`) и эмбеддерной модели с HF.

## Примечания по этой машине

- Python 3.13, GPU RTX 5070, 62 ГБ RAM; WSL2/Docker не работают (Hyper-V не установлен).
- `fast_align`/`atools` собраны через MSYS2/MinGW без OpenMP, поэтому однопоточно;
  на ~500k пар это минуты-десятки минут. `fast_align -s` (скор параллельности из задачи 2)
  тоже доступен: `tools\fast_align\build\fast_align.exe -i align\Tatoeba.en-ru.txt -d -o -v -s`.
  По ТЗ этот метод не годится для малых корпусов и требует памяти — на 500k пар он рабочий.
- `bicleaner`, `marian-scorer`, `pycld2`, `kenlm` требуют C-компиляции или гейтованных
  моделей и на Python 3.13 ставятся плохо — их путь: Colab/Linux (см. План.md).
- Готовый пайплайн использует эмбеддинговый скор как основной, это соответствует пункту
  «мультиязычные эмбеддинги» из ТЗ.
