from __future__ import annotations

import json
import re
from typing import Any

from medical_distill.utils import get_nested_value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def exact_match(prediction: Any, reference: Any) -> bool:
    return normalize_text(prediction) == normalize_text(reference)


def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = 0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def resolve_field(row: dict[str, Any], field: str, compare_field: str | None) -> Any:
    value = row.get(field)
    if compare_field:
        value = get_nested_value(value, compare_field)
    return value


def compute_basic_metrics(
    rows: list[dict[str, Any]],
    prediction_field: str,
    reference_field: str,
    compare_field: str | None = None,
    abstain_values: list[str] | None = None,
    positive_labels: list[str] | None = None,
) -> dict[str, Any]:
    abstain_values = abstain_values or []
    positive_labels = positive_labels or []
    abstain_set = {normalize_text(value) for value in abstain_values}
    positive_set = {normalize_text(value) for value in positive_labels}

    exact_matches = 0
    token_f1_sum = 0.0
    abstentions = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for row in rows:
        prediction = resolve_field(row, prediction_field, compare_field)
        reference = resolve_field(row, reference_field, compare_field)

        if exact_match(prediction, reference):
            exact_matches += 1
        token_f1_sum += token_f1(prediction, reference)

        normalized_prediction = normalize_text(prediction)
        normalized_reference = normalize_text(reference)
        if normalized_prediction in abstain_set:
            abstentions += 1

        if positive_set:
            prediction_is_positive = normalized_prediction in positive_set
            reference_is_positive = normalized_reference in positive_set
            if prediction_is_positive and reference_is_positive:
                true_positive += 1
            elif prediction_is_positive and not reference_is_positive:
                false_positive += 1
            elif not prediction_is_positive and reference_is_positive:
                false_negative += 1

    total = len(rows)
    metrics: dict[str, Any] = {
        "count": total,
        "exact_match": exact_matches / total if total else 0.0,
        "avg_token_f1": token_f1_sum / total if total else 0.0,
        "abstain_rate": abstentions / total if total else 0.0,
    }

    if positive_set:
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics["positive_precision"] = precision
        metrics["positive_recall"] = recall
        metrics["positive_f1"] = f1

    return metrics

