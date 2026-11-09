from __future__ import annotations

from statistics import mean

from nltk.tokenize import wordpunct_tokenize
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_bleu


def calculate_generation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Количество предсказаний и эталонных ответов должно совпадать")
    if not predictions:
        raise ValueError("Нужно передать хотя бы одно предсказание")

    normalized_predictions = [prediction.strip() for prediction in predictions]
    normalized_references = [reference.strip() for reference in references]
    # В DailyDialog пробелы возле пунктуации уже расставлены. force отключает
    # предупреждение SacreBLEU, но не меняет значение метрики.
    bleu = corpus_bleu(normalized_predictions, [normalized_references], force=True).score

    meteor_values: list[float] = []
    for prediction, reference in zip(normalized_predictions, normalized_references, strict=True):
        prediction_tokens = wordpunct_tokenize(prediction.lower())
        reference_tokens = wordpunct_tokenize(reference.lower())
        if not prediction_tokens:
            meteor_values.append(0.0)
        else:
            meteor_values.append(meteor_score([reference_tokens], prediction_tokens))

    return {
        "bleu": round(float(bleu), 4),
        "meteor": round(float(mean(meteor_values)), 4),
        "examples": len(predictions),
    }
