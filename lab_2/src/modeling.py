from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


def parameter_count(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def validation_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            loss = model(**batch).loss
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def train_model(
    model: torch.nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
    output_dir: Path,
    tokenizer: Any,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer_steps_per_epoch = (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)
    # У T5 возможны переполнения во float16. На современных GPU используем
    # bfloat16: у него шире диапазон значений и не нужен динамический GradScaler.
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bfloat16)

    model.to(device)
    history: list[dict[str, float | int]] = []
    started = perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=True)

        for batch_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with _autocast(device):
                loss = model(**batch).loss
                scaled_loss = loss / gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Некорректный loss на эпохе {epoch}, batch={batch_index}: {float(loss.detach().cpu())}"
                )
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())

            # Градиенты копятся несколько batch, имитируя больший batch без
            # соответствующего увеличения потребления видеопамяти.
            should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # При переполнении fp16 GradScaler пропускает optimizer.step().
                # Scheduler тоже должен пропустить этот шаг, чтобы не рассинхронизироваться.
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{running_loss / batch_index:.3f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = validation_loss(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "history": history,
        "seconds": round(perf_counter() - started, 2),
        "optimizer_steps": total_optimizer_steps,
    }


@torch.no_grad()
def generate_in_batches(
    items: list[Any],
    batch_size: int,
    generation_function: Callable[[list[Any]], list[str]],
    description: str,
) -> list[str]:
    predictions: list[str] = []
    for start in tqdm(range(0, len(items), batch_size), desc=description, leave=True):
        predictions.extend(generation_function(items[start : start + batch_size]))
    return predictions
