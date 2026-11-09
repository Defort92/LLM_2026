# Лабораторная работа №3

Дообучение bi-encoder модели `intfloat/multilingual-e5-small` для поиска релевантного контекста по вопросу на данных SberQuAD.

Выполнен базовый вариант из предоставленного ноутбука, обновлённый под `sentence-transformers==6.2.0`:

- современный `SentenceTransformerTrainer` вместо старого `model.fit()`;
- `MultipleNegativesRankingLoss` с in-batch negatives;
- обязательные E5-префиксы `query:` и `passage:`;
- точное сравнение каждого вопроса со всем тестовым корпусом;
- Recall@5, MRR@10, MAP@10, nDCG@10 и Accuracy@1;
- pairwise-диагностика cosine similarity.

Подробная теория, методика и анализ находятся в `REPORT.md`.

Основной файл для просмотра и сдачи — `lab_3.ipynb`: в нём собран весь код и оставлены базовые пояснения на русском языке.

## Запуск

Откройте `lab_3.ipynb`, выберите интерпретатор `lab_1/.venv/Scripts/python.exe` и выполните ячейки сверху вниз. Для повторного обучения установите `RUN_TRAINING = True`.

Альтернативный запуск из PowerShell:

```powershell
cd G:\ITMO\LLM_2026\lab_3
.\run.ps1
```

Скрипт использует `lab_3\.venv`, если оно существует, либо общее CUDA-окружение `lab_1\.venv`. Данные читаются из:

```text
G:\ITMO\LLM_2026\labs_data\lab3(base)\lab3
```

Кэши и временные файлы направлены на диск G:

- Hugging Face: `lab_3\.cache\huggingface`;
- pip: `lab_3\.pip-cache`;
- временные файлы: `lab_3\.tmp`.

После первой загрузки модели повторные запуски выполняются offline.

## Быстрый smoke-test

```powershell
.\run.ps1 `
  --train-limit 32 `
  --test-positive-limit 50 `
  --pairwise-per-class 20 `
  --batch-size 8 `
  --output-dir .\.tmp\smoke-artifacts
```

## Результаты

| Метрика | Baseline | Fine-tuned | Изменение |
|---|---:|---:|---:|
| Accuracy@1 | 0.9084 | 0.9238 | +0.0155 |
| Recall@5 | 0.9713 | 0.9801 | +0.0088 |
| MRR@10 | 0.9351 | 0.9488 | +0.0137 |
| MAP@10 | 0.9351 | 0.9488 | +0.0137 |
| nDCG@10 | 0.9457 | 0.9590 | +0.0133 |

## Структура

```text
lab_3/
├── src/
│   ├── data.py          # загрузка pickle и построение наборов
│   ├── evaluation.py    # точный поиск и метрики
│   └── train.py         # baseline, обучение, повторная оценка
├── artifacts/
│   ├── checkpoint/      # дообученная E5
│   ├── results.json
│   ├── ranking_changes.csv
│   ├── retrieval_metrics.png
│   └── training_loss.png
├── REPORT.md
├── requirements.txt
└── run.ps1
```
