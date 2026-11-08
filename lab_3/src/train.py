from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from data import (
    build_pairwise_test_set,
    build_retrieval_test_set,
    build_training_dataset,
    load_pickle_examples,
)
from evaluation import evaluate_pairwise, evaluate_retrieval, ranking_examples


MODEL_NAME = "intfloat/multilingual-e5-small"
LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "labs_data" / "lab3(base)" / "lab3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laboratory work 3: fine-tune multilingual E5 for QA retrieval")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=LAB_ROOT / "artifacts")
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-positive-limit", type=int, default=1000)
    parser.add_argument("--pairwise-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = max_seq_length
    return model


def train_model(
    model: SentenceTransformer,
    dataset: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_dir = args.output_dir / "checkpoint"
    trainer_output = args.output_dir / "trainer"
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(trainer_output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # In Transformers v5 a float in warmup_steps is interpreted as a
        # fraction of all training steps; 0.1 means a 10% warmup.
        warmup_steps=0.1,
        bf16=use_bfloat16,
        fp16=torch.cuda.is_available() and not use_bfloat16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        report_to="none",
        run_name="lab-3-multilingual-e5-sberquad",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    loss = MultipleNegativesRankingLoss(model, scale=1.0 / args.temperature)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    started = perf_counter()
    train_result = trainer.train()
    seconds = round(perf_counter() - started, 2)
    model.save_pretrained(str(checkpoint_dir))

    summary = {
        "seconds": seconds,
        "train_runtime": round(float(train_result.metrics.get("train_runtime", seconds)), 4),
        "train_loss": round(float(train_result.metrics.get("train_loss", 0.0)), 6),
        "steps": int(train_result.metrics.get("global_step", trainer.state.global_step)),
    }
    return summary, trainer.state.log_history


def create_metric_plot(results: dict[str, Any], output_dir: Path) -> None:
    metric_keys = ["recall_at_5", "mrr_at_10", "map_at_10", "ndcg_at_10"]
    labels = ["Recall@5", "MRR@10", "MAP@10", "nDCG@10"]
    baseline = [results["baseline"]["retrieval"][key] for key in metric_keys]
    fine_tuned = [results["fine_tuned"]["retrieval"][key] for key in metric_keys]
    x = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, baseline, width, label="Baseline", color="#9aa6b2")
    axis.bar(x + width / 2, fine_tuned, width, label="Fine-tuned", color="#377eb8")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Значение метрики")
    axis.set_title("SberQuAD: точный поиск по 906 документам")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_training_plot(history: list[dict[str, Any]], output_dir: Path) -> None:
    points = [(entry["step"], entry["loss"]) for entry in history if "loss" in entry and "step" in entry]
    if not points:
        return
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", markersize=3)
    axis.set_xlabel("Шаг обучения")
    axis.set_ylabel("MultipleNegativesRankingLoss")
    axis.set_title("Динамика contrastive loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_rows = load_pickle_examples(args.data_dir / "trainval.pkl")
    test_rows = load_pickle_examples(args.data_dir / "test.pkl")
    train_dataset, train_metadata = build_training_dataset(train_rows, args.train_limit, args.seed)
    retrieval_queries, retrieval_documents, retrieval_metadata = build_retrieval_test_set(
        test_rows, args.test_positive_limit
    )
    pair_queries, pair_documents, pair_labels = build_pairwise_test_set(test_rows, args.pairwise_per_class)

    model = load_model(device, args.max_seq_length)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("\n=== Baseline evaluation ===", flush=True)
    baseline_retrieval, baseline_ranks, baseline_ids = evaluate_retrieval(
        model, retrieval_queries, retrieval_documents, args.encode_batch_size
    )
    baseline_pairwise = evaluate_pairwise(
        model, pair_queries, pair_documents, pair_labels, args.encode_batch_size
    )
    print(f"Baseline retrieval: {baseline_retrieval}", flush=True)
    print(f"Baseline pairwise: {baseline_pairwise}", flush=True)

    print("\n=== Fine-tuning ===", flush=True)
    training, history = train_model(model, train_dataset, args)

    print("\n=== Fine-tuned evaluation ===", flush=True)
    fine_tuned_retrieval, fine_tuned_ranks, fine_tuned_ids = evaluate_retrieval(
        model, retrieval_queries, retrieval_documents, args.encode_batch_size
    )
    fine_tuned_pairwise = evaluate_pairwise(
        model, pair_queries, pair_documents, pair_labels, args.encode_batch_size
    )
    print(f"Fine-tuned retrieval: {fine_tuned_retrieval}", flush=True)
    print(f"Fine-tuned pairwise: {fine_tuned_pairwise}", flush=True)

    retrieval_delta = {
        key: round(float(fine_tuned_retrieval[key]) - float(baseline_retrieval[key]), 6)
        for key in ["accuracy_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "map_at_10", "ndcg_at_10"]
    }
    pairwise_delta = {
        key: round(float(fine_tuned_pairwise[key]) - float(baseline_pairwise[key]), 6)
        for key in ["roc_auc", "spearman", "positive_cosine_mean", "negative_cosine_mean", "cosine_margin"]
    }

    results: dict[str, Any] = {
        "experiment": {
            "model": MODEL_NAME,
            "parameters": parameter_count,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "max_seq_length": args.max_seq_length,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "loss": "MultipleNegativesRankingLoss with in-batch negatives",
            "mixed_precision": "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float16" if device == "cuda" else "float32",
        },
        "data": {
            "source": "SberQuAD prepared pickle files supplied with the laboratory",
            "train": train_metadata,
            "retrieval_test": retrieval_metadata,
            "pairwise_test_pairs": len(pair_labels),
        },
        "baseline": {"retrieval": baseline_retrieval, "pairwise": baseline_pairwise},
        "fine_tuned": {"retrieval": fine_tuned_retrieval, "pairwise": fine_tuned_pairwise},
        "delta": {"retrieval": retrieval_delta, "pairwise": pairwise_delta},
        "training": training,
        "training_history": history,
    }

    examples = ranking_examples(
        retrieval_queries,
        retrieval_documents,
        baseline_ranks,
        baseline_ids,
        fine_tuned_ranks,
        fine_tuned_ids,
    )
    pd.DataFrame(examples).to_csv(args.output_dir / "ranking_changes.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    create_metric_plot(results, args.output_dir)
    create_training_plot(history, args.output_dir)
    print(f"\nDone. Results: {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
