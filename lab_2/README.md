# Лабораторная работа №2

Дообучение двух диалоговых моделей для предсказания ответа на сообщение пользователя:

- decoder-only: `distilbert/distilgpt2`;
- encoder-decoder: `google/flan-t5-small`;
- датасет: `OpenRL/daily_dialog` (DailyDialog, Parquet);
- метрики: corpus BLEU (SacreBLEU) и средний METEOR (NLTK).

Эксперимент полностью выполнен на NVIDIA GeForce RTX 4060 Laptop GPU. Итоговые метрики, ответы моделей, графики и сохранённые модели находятся в `artifacts/`. Подробное объяснение теории и результатов — в `REPORT.md`.

Основной файл для просмотра и сдачи — `lab_2.ipynb`: в нём собран весь код и оставлены базовые пояснения на русском языке.

## Быстрый запуск

Откройте `lab_2.ipynb`, выберите интерпретатор `lab_1/.venv/Scripts/python.exe` и выполните ячейки сверху вниз. Для повторного обучения установите `RUN_TRAINING = True`.

Альтернативный запуск из PowerShell:

```powershell
cd G:\ITMO\LLM_2026\lab_2
.\run.ps1
```

`run.ps1` сначала ищет собственное окружение `lab_2\.venv`, а если его нет — использует уже установленное `lab_1\.venv`. Это сделано намеренно: CUDA-версия PyTorch занимает несколько гигабайт, поэтому её копия не создаётся повторно.

Все кэши направлены на диск G:

- Hugging Face: `lab_2\.cache\huggingface`;
- NLTK: `lab_2\.cache\nltk_data`;
- pip: `lab_2\.pip-cache`;
- временные файлы: `lab_2\.tmp`.

После первого скачивания моделей и датасета повторные запуски автоматически выполняются в offline-режиме.

## Параметры запуска

По умолчанию используются 6000 обучающих пар, 200 validation-примеров для генерации и одна эпоха. Для быстрого теста:

```powershell
.\run.ps1 --train-limit 32 --validation-limit 8 --validation-loss-limit 8 --batch-size 2
```

Основные аргументы можно посмотреть командой:

```powershell
.\run.ps1 --help
```

## Структура

```text
lab_2/
├── src/
│   ├── data.py          # DailyDialog -> пары сообщение/ответ и Dataset-классы
│   ├── metrics.py       # BLEU и METEOR
│   ├── modeling.py      # цикл обучения и пакетная генерация
│   └── train.py         # полный эксперимент для двух архитектур
├── artifacts/
│   ├── checkpoints/     # дообученные модели
│   ├── results.json     # параметры и итоговые метрики
│   ├── examples_*.csv   # запрос, эталон, ответы до/после обучения
│   └── *.png            # графики
├── REPORT.md
├── requirements.txt
└── run.ps1
```
