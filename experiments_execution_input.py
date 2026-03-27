from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence

from model_with_sae import ModelWithSAEModule
from prompts.experiments_execution_prompt import (
    build_input_activation_context,
    build_input_expansion_context,
)

InputTestType = Literal["activation", "expansion"]


def _extract_designed_sentences(experiment_item: Dict[str, Any]) -> List[str]:
    raw = experiment_item.get("designed_sentences")
    if not isinstance(raw, list):
        raise ValueError("Each input-side experiment item must contain list field 'designed_sentences'.")
    sentences: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sentences.append(item.strip())
    return sentences


def _extract_test_type(experiment_item: Dict[str, Any]) -> InputTestType:
    raw = str(experiment_item.get("test_type", "activation")).strip().lower()
    if raw not in ("activation", "expansion"):
        return "activation"
    return "expansion" if raw == "expansion" else "activation"


def _extract_reference_hypothesis(experiment_item: Dict[str, Any]) -> str:
    return str(experiment_item.get("reference_hypothesis", "")).strip()


def _extract_max_token(trace: Dict[str, Any]) -> str:
    tokens = trace.get("tokens")
    max_token_index = trace.get("max_token_index")
    if not isinstance(tokens, list) or not isinstance(max_token_index, int):
        return ""
    if max_token_index < 0 or max_token_index >= len(tokens):
        return ""
    token = tokens[max_token_index]
    return token if isinstance(token, str) else str(token)


def _run_sentence_batch(
    *,
    module: ModelWithSAEModule,
    sentences: Sequence[str],
    non_zero_threshold: float,
) -> Dict[str, Any]:
    sentence_results: List[Dict[str, Any]] = []
    non_zero_count = 0
    activation_sum = 0.0
    activation_max = 0.0

    for sentence_index, sentence in enumerate(sentences, start=1):
        trace = module.get_activation_trace(sentence)
        activation_value = float(trace.get("summary_activation", 0.0) or 0.0)
        activation_sum += activation_value
        activation_max = max(activation_max, activation_value)
        is_non_zero = activation_value > non_zero_threshold
        if is_non_zero:
            non_zero_count += 1

        sentence_results.append(
            {
                "sentence_index": sentence_index,
                "sentence": sentence,
                "summary_activation": activation_value,
                "summary_activation_mean": float(trace.get("summary_activation_mean", 0.0) or 0.0),
                "summary_activation_sum": float(trace.get("summary_activation_sum", 0.0) or 0.0),
                "max_token_index": int(trace.get("max_token_index", 0) or 0),
                "max_token": _extract_max_token(trace),
                "is_non_zero": is_non_zero,
            }
        )

    total_sentences = len(sentences)
    non_zero_rate = (non_zero_count / total_sentences) if total_sentences > 0 else 0.0
    mean_activation = (activation_sum / total_sentences) if total_sentences > 0 else 0.0
    return {
        "sentence_results": sentence_results,
        "non_zero_count": non_zero_count,
        "total_sentences": total_sentences,
        "score_non_zero_rate": non_zero_rate,
        "mean_summary_activation": mean_activation,
        "max_summary_activation": activation_max,
    }


def execute_input_side_experiments(
    *,
    input_side_experiments: Sequence[Dict[str, Any]],
    module: ModelWithSAEModule,
    non_zero_threshold: float = 0.0,
) -> Dict[str, Any]:
    hypothesis_results: List[Dict[str, Any]] = []

    for hypothesis_index, item in enumerate(input_side_experiments, start=1):
        hypothesis_text = str(item.get("hypothesis", "")).strip()
        test_type = _extract_test_type(item)
        reference_hypothesis = _extract_reference_hypothesis(item)
        sentences = _extract_designed_sentences(item)

        metrics = _run_sentence_batch(
            module=module,
            sentences=sentences,
            non_zero_threshold=non_zero_threshold,
        )

        if test_type == "expansion":
            input_test_context = build_input_expansion_context(
                hypothesis=hypothesis_text,
                reference_hypothesis=reference_hypothesis,
                expansion_sentences=sentences,
            )
        else:
            input_test_context = build_input_activation_context(
                hypothesis=hypothesis_text,
                designed_sentences=sentences,
            )

        hypothesis_results.append(
            {
                "hypothesis_index": hypothesis_index,
                "hypothesis": hypothesis_text,
                "test_type": test_type,
                "reference_hypothesis": reference_hypothesis,
                "designed_sentences": sentences,
                "input_test_context": input_test_context,
                "sentence_results": metrics["sentence_results"],
                "non_zero_count": metrics["non_zero_count"],
                "total_sentences": metrics["total_sentences"],
                "score_non_zero_rate": metrics["score_non_zero_rate"],
                "mean_summary_activation": metrics["mean_summary_activation"],
                "max_summary_activation": metrics["max_summary_activation"],
                "is_full_activation": bool(
                    metrics["total_sentences"] > 0 and metrics["non_zero_count"] == metrics["total_sentences"]
                ),
            }
        )

    if hypothesis_results:
        overall_score = sum(item["score_non_zero_rate"] for item in hypothesis_results) / len(hypothesis_results)
    else:
        overall_score = 0.0

    unique_test_types = sorted({str(item.get("test_type", "activation")) for item in hypothesis_results})
    input_test_mode = unique_test_types[0] if len(unique_test_types) == 1 else "mixed"

    return {
        "side": "input",
        "non_zero_threshold": non_zero_threshold,
        "input_test_mode": input_test_mode,
        "hypothesis_results": hypothesis_results,
        "overall_score_non_zero_rate": overall_score,
    }
