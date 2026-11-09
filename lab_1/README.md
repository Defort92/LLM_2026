# Лабораторная работа №1

Сравнение `mean pooling` и `CLS pooling` при многоклассовой классификации текстов.

## Что находится в проекте

- `lab_1.ipynb` — основной вариант лабораторной: код и базовые пояснения в одном ноутбуке;
- `REPORT.md` — теория, описание эксперимента, результаты и выводы;
- `src/model.py` — encoder, pooling и линейный классификатор;
- `src/data.py` — загрузка AG News и стратифицированное разбиение;
- `src/metrics.py` — Precision, Recall, F1 и confusion matrix;
- `src/train.py` — обучение, оценка и построение графиков;
- `artifacts/results.json` — полные результаты и примеры ошибок;
- `artifacts/comparison.csv` — итоговая таблица метрик;
- `artifacts/training_history.csv` — история обучения;
- `artifacts/plots/` — графики и матрицы ошибок;
- `run.ps1` — запуск с локальными кэшами на диске проекта.

## Окружение

Эксперимент выполнен на Python 3.12, PyTorch 2.12.1 и CUDA 13.0. Точные версии находятся в `requirements.txt` и `artifacts/results.json`.

Текущее виртуальное окружение уже создано в `lab_1/.venv`. Для повторной установки:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Запуск

Рекомендуемый вариант: открыть `lab_1.ipynb`, выбрать интерпретатор `lab_1/.venv/Scripts/python.exe` и выполнить ячейки сверху вниз. По умолчанию ноутбук показывает уже сохранённые результаты. Для полного повторного обучения установите `RUN_TRAINING = True`.

Альтернативный запуск из корня репозитория:

```powershell
.\lab_1\run.ps1
```

Параметры итогового эксперимента задаются по умолчанию. Их можно изменить:

```powershell
.\lab_1\run.ps1 `
    --epochs 2 `
    --batch-size 32 `
    --train-per-class 3000 `
    --validation-per-class 1000 `
    --max-length 128 `
    --pooling both
```

Для быстрого smoke-test:

```powershell
.\lab_1\run.ps1 `
    --epochs 1 `
    --batch-size 8 `
    --train-per-class 8 `
    --validation-per-class 4 `
    --max-length 32 `
    --pooling mean
```

`run.ps1` направляет кэши Hugging Face, pip и временные файлы в каталог `lab_1` на том же диске, где расположен проект. Если модель и датасет уже загружены, автоматически включается offline-режим.

## Итоговые метрики

| Pooling | Precision macro | Recall macro | F1 macro | Accuracy |
|---|---:|---:|---:|---:|
| Mean | 0.9056 | 0.9059 | 0.9057 | 0.9059 |
| CLS | 0.9016 | 0.9012 | 0.9008 | 0.9012 |

На выбранных данных `mean pooling` превзошёл `CLS pooling` по macro F1 примерно на 0.0049.
