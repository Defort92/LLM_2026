from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from data import (
    DecoderOnlyDataset,
    EncoderDecoderDataset,
    DialoguePair,
    decoder_prompt,
    load_dialogue_pairs,
    seq2seq_prompt,
)
from metrics import calculate_generation_metrics
from modeling import generate_in_batches, parameter_count, train_model


DECODER_MODEL = "distilbert/distilgpt2"
ENCODER_DECODER_MODEL = "google/flan-t5-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лабораторная №2: дообучение двух диалоговых архитектур")
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--validation-loss-limit", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-input-length", type=int, default=64)
    parser.add_argument("--max-target-length", type=int, default=48)
    parser.add_argument("--generation-length", type=int, default=40)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_wordnet() -> None:
    target = os.environ.get("NLTK_DATA")
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        if not nltk.download("wordnet", download_dir=target, quiet=False):
            raise RuntimeError("Не удалось загрузить WordNet, необходимый для METEOR")


def generate_decoder(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[DialoguePair],
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = [decoder_prompt(pair.message) for pair in pairs]
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(device)
    model.eval()
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # У decoder-only результата сначала идёт исходный промпт. Оставляем только
    # новые токены, сгенерированные после него.
    continuation = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(continuation, skip_special_tokens=True)


def generate_seq2seq(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[DialoguePair],
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = [seq2seq_prompt(pair.message) for pair in pairs]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(device)
    model.eval()
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def save_partial_results(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def run_architecture(
    architecture: str,
    model_name: str,
    train_pairs: list[DialoguePair],
    validation_pairs: list[DialoguePair],
    validation_loss_pairs: list[DialoguePair],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    print(f"\n=== {architecture}: {model_name} ===", flush=True)
    if architecture == "decoder-only":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        dataset_class = DecoderOnlyDataset
        generation_impl = generate_decoder
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        dataset_class = EncoderDecoderDataset
        generation_impl = generate_seq2seq

    counts = parameter_count(model)
    model.to(device)
    generation_args = (model, tokenizer, device, args.max_input_length, args.generation_length)

    def generate(batch: list[DialoguePair]) -> list[str]:
        return generation_impl(generation_args[0], generation_args[1], batch, *generation_args[2:])

    baseline_started = perf_counter()
    baseline_predictions = generate_in_batches(
        validation_pairs,
        args.generation_batch_size,
        generate,
        f"{architecture} zero-shot",
    )
    references = [pair.response for pair in validation_pairs]
    baseline_metrics = calculate_generation_metrics(baseline_predictions, references)
    baseline_seconds = round(perf_counter() - baseline_started, 2)
    print(f"Zero-shot metrics: {baseline_metrics}", flush=True)

    tokenizer.padding_side = "right"
    train_dataset = dataset_class(
        train_pairs,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )
    validation_dataset = dataset_class(
        validation_loss_pairs,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )
    checkpoint_dir = args.output_dir / "checkpoints" / architecture
    training = train_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
        output_dir=checkpoint_dir,
        tokenizer=tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        seed=args.seed,
    )

    fine_tuned_started = perf_counter()
    fine_tuned_predictions = generate_in_batches(
        validation_pairs,
        args.generation_batch_size,
        generate,
        f"{architecture} fine-tuned",
    )
    fine_tuned_metrics = calculate_generation_metrics(fine_tuned_predictions, references)
    fine_tuned_seconds = round(perf_counter() - fine_tuned_started, 2)
    print(f"Fine-tuned metrics: {fine_tuned_metrics}", flush=True)

    examples = pd.DataFrame(
        {
            "message": [pair.message for pair in validation_pairs],
            "reference": references,
            "zero_shot": baseline_predictions,
            "fine_tuned": fine_tuned_predictions,
        }
    )
    result = {
        "architecture": architecture,
        "model": model_name,
        "parameters": counts,
        "zero_shot": baseline_metrics,
        "fine_tuned": fine_tuned_metrics,
        "improvement": {
            "bleu_absolute": round(fine_tuned_metrics["bleu"] - baseline_metrics["bleu"], 4),
            "meteor_absolute": round(fine_tuned_metrics["meteor"] - baseline_metrics["meteor"], 4),
        },
        "generation_seconds": {"zero_shot": baseline_seconds, "fine_tuned": fine_tuned_seconds},
        "training": training,
    }

    del model, tokenizer, train_dataset, validation_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, examples


def create_plots(results: dict[str, Any], output_dir: Path) -> None:
    architectures = ["decoder-only", "encoder-decoder"]
    labels = ["Decoder-only\nDistilGPT2", "Encoder-decoder\nFLAN-T5-small"]
    x = np.arange(len(architectures))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, title in zip(axes, ["bleu", "meteor"], ["BLEU (0–100)", "METEOR (0–1)"], strict=True):
        baseline = [results[architecture]["zero_shot"][metric] for architecture in architectures]
        trained = [results[architecture]["fine_tuned"][metric] for architecture in architectures]
        axis.bar(x - width / 2, baseline, width, label="Zero-shot", color="#9aa6b2")
        axis.bar(x + width / 2, trained, width, label="После обучения", color="#377eb8")
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("DailyDialog: метрики до и после дообучения")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for architecture, label in zip(architectures, labels, strict=True):
        history = results[architecture]["training"]["history"]
        axis.plot(
            [item["epoch"] for item in history],
            [item["train_loss"] for item in history],
            marker="o",
            label=f"{label.replace(chr(10), ' ')}: train",
        )
        axis.plot(
            [item["epoch"] for item in history],
            [item["validation_loss"] for item in history],
            marker="s",
            linestyle="--",
            label=f"{label.replace(chr(10), ' ')}: validation",
        )
    axis.set_xlabel("Эпоха")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Функция потерь")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    """Запускает сравнение архитектур с явно переданными параметрами."""

    set_seed(args.seed)
    ensure_wordnet()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_pairs, train_metadata = load_dialogue_pairs("train", args.train_limit, args.seed)
    validation_pool, validation_metadata = load_dialogue_pairs(
        "validation",
        max(args.validation_limit, args.validation_loss_limit),
        args.seed,
    )
    validation_pairs = validation_pool[: args.validation_limit]
    validation_loss_pairs = validation_pool[: args.validation_loss_limit]

    results: dict[str, Any] = {
        "experiment": {
            "seed": args.seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "train_limit": len(train_pairs),
            "validation_generation_limit": len(validation_pairs),
            "validation_loss_limit": len(validation_loss_pairs),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "generation_length": args.generation_length,
            "generation": "greedy decoding; do_sample=False; num_beams=1",
            "mixed_precision": (
                "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else "float16" if device.type == "cuda"
                else "float32"
            ),
        },
        "data": {"train": train_metadata, "validation": validation_metadata},
    }

    for architecture, model_name in [
        ("decoder-only", DECODER_MODEL),
        ("encoder-decoder", ENCODER_DECODER_MODEL),
    ]:
        architecture_result, examples = run_architecture(
            architecture,
            model_name,
            train_pairs,
            validation_pairs,
            validation_loss_pairs,
            args,
            device,
        )
        results[architecture] = architecture_result
        examples.to_csv(args.output_dir / f"examples_{architecture}.csv", index=False, encoding="utf-8-sig")
        save_partial_results(results, args.output_dir)

    create_plots(results, args.output_dir)
    save_partial_results(results, args.output_dir)
    print(f"\nГотово. Результаты: {args.output_dir / 'results.json'}", flush=True)


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
