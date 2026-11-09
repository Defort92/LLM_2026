from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import LABEL_NAMES, DatasetSizes, build_stratified_splits, tokenize_split
from metrics import ClassificationMetrics, calculate_metrics
from model import PoolingType, TextClassifier


LAB_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 42
    train_per_class: int = 3_000
    validation_per_class: int = 1_000
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнение mean pooling и CLS pooling на AG News."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-per-class", type=int, default=3_000)
    parser.add_argument("--validation-per-class", type=int, default=1_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls", "both"],
        default="both",
        help="Запустить одну стратегию pooling или сравнить обе.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_warmup_decay(
    optimizer: AdamW,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        # Сначала learning rate линейно растёт, затем плавно уменьшается до нуля.
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        remaining_steps = max(1, total_steps - warmup_steps)
        return max(
            0.0,
            float(total_steps - current_step) / float(remaining_steps),
        )

    return LambdaLR(optimizer, lr_lambda)


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def evaluate(
    model: TextClassifier,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, ClassificationMetrics, np.ndarray, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.inference_mode():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    metrics, matrix = calculate_metrics(targets, predictions)
    return total_loss / len(loader.dataset), metrics, matrix, targets, predictions


def train_one_pooling(
    pooling: PoolingType,
    config: ExperimentConfig,
    tokenized: dict[str, TensorDataset],
    raw_test_texts: list[str],
    device: torch.device,
) -> tuple[dict, list[dict]]:
    set_seed(config.seed)
    model = TextClassifier(
        model_name=config.model_name,
        num_classes=len(LABEL_NAMES),
        pooling=pooling,
        dropout=config.dropout,
    ).to(device)

    train_loader = make_loader(
        tokenized["train"], config.batch_size, shuffle=True, seed=config.seed
    )
    validation_loader = make_loader(
        tokenized["validation"], config.batch_size, shuffle=False, seed=config.seed
    )
    test_loader = make_loader(
        tokenized["test"], config.batch_size, shuffle=False, seed=config.seed
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = linear_warmup_decay(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=config.warmup_ratio,
    )
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    # Mixed precision уменьшает расход видеопамяти; GradScaler защищает fp16
    # градиенты от численного обнуления.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_validation_f1 = -1.0
    checkpoint_path = CHECKPOINT_DIR / f"{pooling}_best.pt"
    history: list[dict] = []
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"{pooling}: epoch {epoch}/{config.epochs}",
            leave=False,
        )

        for input_ids, attention_mask, labels in progress:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        validation_loss, validation_metrics, _, _, _ = evaluate(
            model, validation_loader, device, loss_fn
        )
        epoch_record = {
            "pooling": pooling,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **{
                f"validation_{name}": value
                for name, value in validation_metrics.as_dict().items()
            },
        }
        history.append(epoch_record)
        print(
            f"[{pooling}] epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_macro_f1={validation_metrics.f1_macro:.4f}"
        )

        if validation_metrics.f1_macro > best_validation_f1:
            # На test оценивается не последняя эпоха, а лучшая по validation F1.
            best_validation_f1 = validation_metrics.f1_macro
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pooling": pooling,
                    "config": asdict(config),
                    "validation_metrics": validation_metrics.as_dict(),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, matrix, targets, predictions = evaluate(
        model, test_loader, device, loss_fn
    )
    elapsed_seconds = time.perf_counter() - start_time

    report = classification_report(
        targets,
        predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    errors = []
    for index, (target, prediction) in enumerate(zip(targets, predictions)):
        if target != prediction:
            errors.append(
                {
                    "text": raw_test_texts[index],
                    "target": LABEL_NAMES[target],
                    "prediction": LABEL_NAMES[prediction],
                }
            )
        if len(errors) == 30:
            break

    result = {
        "pooling": pooling,
        "best_validation_f1_macro": best_validation_f1,
        "test_loss": test_loss,
        **test_metrics.as_dict(),
        "elapsed_seconds": elapsed_seconds,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "sample_errors": errors,
    }

    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, history


def save_plots(results: list[dict], history: list[dict]) -> None:
    plots_dir = ARTIFACTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    comparison_rows = []
    for result in results:
        for metric in ["precision_macro", "recall_macro", "f1_macro", "accuracy"]:
            comparison_rows.append(
                {
                    "Pooling": result["pooling"].upper(),
                    "Metric": metric.replace("_macro", "").replace("_", " ").title(),
                    "Value": result[metric],
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    plt.figure(figsize=(9, 5))
    chart = sns.barplot(
        data=comparison_df,
        x="Metric",
        y="Value",
        hue="Pooling",
    )
    chart.set_ylim(0, 1)
    chart.set_title("AG News: сравнение стратегий pooling")
    chart.set_ylabel("Значение метрики")
    for container in chart.containers:
        chart.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=180)
    plt.close()

    history_df = pd.DataFrame(history)
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=history_df,
        x="epoch",
        y="validation_f1_macro",
        hue="pooling",
        marker="o",
    )
    lower_bound = max(0.0, history_df["validation_f1_macro"].min() - 0.005)
    upper_bound = min(1.0, history_df["validation_f1_macro"].max() + 0.005)
    plt.ylim(lower_bound, upper_bound)
    plt.xticks(sorted(history_df["epoch"].unique()))
    plt.ylabel("Macro F1")
    plt.xlabel("Эпоха")
    plt.title("Macro F1 на validation по эпохам")
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_f1_history.png", dpi=180)
    plt.close()

    for result in results:
        matrix = np.asarray(result["confusion_matrix"])
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
        )
        plt.xlabel("Предсказанный класс")
        plt.ylabel("Истинный класс")
        plt.title(f"Матрица ошибок: {result['pooling'].upper()} pooling")
        plt.tight_layout()
        plt.savefig(
            plots_dir / f"confusion_matrix_{result['pooling']}.png",
            dpi=180,
        )
        plt.close()


def run_experiment(args: argparse.Namespace) -> None:
    """Запускает эксперимент с явно переданными параметрами."""

    config = ExperimentConfig(
        seed=args.seed,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    set_seed(config.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    raw = build_stratified_splits(
        seed=config.seed,
        sizes=DatasetSizes(
            train_per_class=config.train_per_class,
            validation_per_class=config.validation_per_class,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized = {
        split_name: tokenize_split(split, tokenizer, config.max_length)
        for split_name, split in raw.items()
    }
    print(
        "Dataset sizes: "
        + ", ".join(f"{name}={len(split)}" for name, split in raw.items())
    )

    pooling_strategies: list[PoolingType]
    if args.pooling == "both":
        pooling_strategies = ["mean", "cls"]
    else:
        pooling_strategies = [args.pooling]

    results: list[dict] = []
    complete_history: list[dict] = []
    for pooling in pooling_strategies:
        result, history = train_one_pooling(
            pooling=pooling,
            config=config,
            tokenized=tokenized,
            raw_test_texts=raw["test"]["text"],
            device=device,
        )
        results.append(result)
        complete_history.extend(history)

    payload = {
        "config": asdict(config),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "dataset": {
            "name": "fancyzhx/ag_news",
            "labels": LABEL_NAMES,
            "train_size": len(raw["train"]),
            "validation_size": len(raw["validation"]),
            "test_size": len(raw["test"]),
        },
        "results": results,
    }
    (ARTIFACTS_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(complete_history).to_csv(
        ARTIFACTS_DIR / "training_history.csv", index=False
    )

    comparison_columns = [
        "pooling",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "accuracy",
        "best_validation_f1_macro",
        "test_loss",
        "elapsed_seconds",
    ]
    pd.DataFrame(results)[comparison_columns].to_csv(
        ARTIFACTS_DIR / "comparison.csv", index=False
    )
    save_plots(results, complete_history)

    print("\nИтоговые метрики на test")
    print(pd.DataFrame(results)[comparison_columns].to_string(index=False))
    print(f"\nАртефакты сохранены в: {ARTIFACTS_DIR}")


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
