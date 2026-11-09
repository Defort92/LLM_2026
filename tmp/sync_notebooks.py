from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def clean_source(path: Path, local_modules: tuple[str, ...], replacements: dict[str, str]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    cleaned: list[str] = []
    skipping_import = False

    for line in lines:
        stripped = line.strip()
        if stripped == "from __future__ import annotations":
            continue
        if skipping_import:
            if stripped == ")":
                skipping_import = False
            continue

        remove_line = False
        for module in local_modules:
            if stripped.startswith(f"from {module} import"):
                skipping_import = stripped.endswith("(")
                remove_line = True
                break
        if remove_line:
            continue
        if stripped == 'if __name__ == "__main__":':
            break

        for old, new in replacements.items():
            line = line.replace(old, new)
        cleaned.append(line)

    return "".join(cleaned).rstrip() + "\n"


def set_code(cell: dict, source: str, index: int) -> None:
    cell["source"] = source.rstrip().splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []
    cell["id"] = hashlib.sha1(f"{index}:{source}".encode("utf-8")).hexdigest()[:8]


def synchronize(
    notebook_path: Path,
    source_by_heading: dict[str, str],
    run_source: str,
    run_heading: str,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    for index, cell in enumerate(cells[:-1]):
        if cell["cell_type"] != "markdown":
            continue
        heading = "".join(cell["source"]).strip()
        if heading in source_by_heading:
            set_code(cells[index + 1], source_by_heading[heading], index + 1)
        elif heading.startswith(run_heading):
            set_code(cells[index + 1], run_source, index + 1)

    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


lab_1 = ROOT / "lab_1"
synchronize(
    lab_1 / "lab_1.ipynb",
    {
        "## 2. Подготовка данных": clean_source(lab_1 / "src" / "data.py", (), {}),
        "## 3. Метрики классификации": clean_source(lab_1 / "src" / "metrics.py", (), {}),
        "## 4. Модель и два варианта pooling": clean_source(lab_1 / "src" / "model.py", (), {}),
        "## 5. Обучение, оценка и сохранение результатов": clean_source(
            lab_1 / "src" / "train.py",
            ("data", "metrics", "model"),
            {"LAB_ROOT = Path(__file__).resolve().parents[1]": "LAB_ROOT = LAB_ROOT"},
        ),
    },
    '''# Параметры полного эксперимента видны прямо в ноутбуке.
EXPERIMENT_ARGS = argparse.Namespace(
    epochs=2,
    batch_size=32,
    train_per_class=3_000,
    validation_per_class=1_000,
    max_length=128,
    seed=42,
    pooling="both",
)

RUN_TRAINING = False

if RUN_TRAINING:
    run_experiment(EXPERIMENT_ARGS)
else:
    print("Обучение пропущено. Ниже загружаются сохранённые результаты.")
''',
    "## 6. Запуск эксперимента",
)


lab_2 = ROOT / "lab_2"
synchronize(
    lab_2 / "lab_2.ipynb",
    {
        "## 2. Подготовка диалоговых пар и датасетов": clean_source(lab_2 / "src" / "data.py", (), {}),
        "## 3. Метрики качества генерации": clean_source(lab_2 / "src" / "metrics.py", (), {}),
        "## 4. Общий цикл обучения и генерации": clean_source(lab_2 / "src" / "modeling.py", (), {}),
        "## 5. Эксперимент с двумя архитектурами": clean_source(
            lab_2 / "src" / "train.py",
            ("data", "metrics", "modeling"),
            {'default=Path(__file__).resolve().parents[1] / "artifacts"': 'default=LAB_ROOT / "artifacts"'},
        ),
    },
    '''EXPERIMENT_ARGS = argparse.Namespace(
    train_limit=6_000,
    validation_limit=200,
    validation_loss_limit=500,
    epochs=1,
    batch_size=8,
    gradient_accumulation=2,
    learning_rate=5e-5,
    max_input_length=64,
    max_target_length=48,
    generation_length=40,
    generation_batch_size=16,
    seed=42,
    output_dir=LAB_ROOT / "artifacts",
)

RUN_TRAINING = False

if RUN_TRAINING:
    run_experiment(EXPERIMENT_ARGS)
else:
    print("Обучение пропущено. Ниже загружаются сохранённые результаты.")
''',
    "## 6. Запуск эксперимента",
)


lab_3 = ROOT / "lab_3"
synchronize(
    lab_3 / "lab_3.ipynb",
    {
        "## 2. Подготовка обучающей и тестовой выборок": clean_source(lab_3 / "src" / "data.py", (), {}),
        "## 3. Метрики семантического поиска": clean_source(lab_3 / "src" / "evaluation.py", (), {}),
        "## 4. Базовая оценка, дообучение и повторная оценка": clean_source(
            lab_3 / "src" / "train.py",
            ("data", "evaluation"),
            {
                "LAB_ROOT = Path(__file__).resolve().parents[1]": "LAB_ROOT = LAB_ROOT",
                "PROJECT_ROOT = LAB_ROOT.parent": "PROJECT_ROOT = LAB_ROOT.parent",
            },
        ),
    },
    '''EXPERIMENT_ARGS = argparse.Namespace(
    data_dir=PROJECT_ROOT / "labs_data" / "lab3(base)" / "lab3",
    output_dir=LAB_ROOT / "artifacts",
    train_limit=12_000,
    test_positive_limit=1_000,
    pairwise_per_class=1_000,
    epochs=1.0,
    batch_size=16,
    encode_batch_size=64,
    learning_rate=2e-5,
    max_seq_length=256,
    temperature=0.01,
    seed=42,
)

RUN_TRAINING = False

if RUN_TRAINING:
    run_experiment(EXPERIMENT_ARGS)
else:
    print("Обучение пропущено. Ниже загружаются сохранённые результаты.")
''',
    "## 5. Запуск эксперимента",
)

print("Ноутбуки синхронизированы с исходным кодом.")
