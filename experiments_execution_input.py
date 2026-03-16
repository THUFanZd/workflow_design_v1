from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from model_with_sae import ModelWithSAEModule
from prompts.experiments_execution_prompt import (
    build_input_activation_context,
    build_input_boundary_context,
)


def _load_compare_generate_boundary_contexts():
    compare_path = (
        Path(__file__).resolve().parent
        / "explanation_quality_evaluation"
        / "input-side-evaluation"
        / "compare_explanations_with_llm.py"
    )
    spec = importlib.util.spec_from_file_location("compare_explanations_with_llm_shared", str(compare_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {compare_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, "generate_boundary_contexts")


def generate_boundary_contexts(*args: Any, **kwargs: Any) -> List[str]:
    compare_generate_boundary_contexts = _load_compare_generate_boundary_contexts()
    return compare_generate_boundary_contexts(*args, **kwargs)


def _extract_designed_sentences(experiment_item: Dict[str, Any]) -> List[str]:
    raw = experiment_item.get("designed_sentences")
    if not isinstance(raw, list):
        raise ValueError("Each input-side experiment item must contain a list field 'designed_sentences'.")
    sentences: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sentences.append(item.strip())
    return sentences


def _extract_boundary_sentences(experiment_item: Dict[str, Any]) -> List[str]:
    raw = experiment_item.get("boundary_sentences")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Field 'boundary_sentences' must be a list when provided.")
    sentences: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sentences.append(item.strip())
    return sentences


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
        sentences = _extract_designed_sentences(item)
        boundary_sentences = _extract_boundary_sentences(item)

        activation_metrics = _run_sentence_batch(
            module=module,
            sentences=sentences,
            non_zero_threshold=non_zero_threshold,
        )
        boundary_metrics = _run_sentence_batch(
            module=module,
            sentences=boundary_sentences,
            non_zero_threshold=non_zero_threshold,
        )
        boundary_non_activation_count = (
            boundary_metrics["total_sentences"] - boundary_metrics["non_zero_count"]
        )
        boundary_non_activation_rate = (
            boundary_non_activation_count / boundary_metrics["total_sentences"]
            if boundary_metrics["total_sentences"] > 0
            else None
        )

        hypothesis_results.append(
            {
                "hypothesis_index": hypothesis_index,
                "hypothesis": hypothesis_text,
                "designed_sentences": sentences,
                "boundary_sentences": boundary_sentences,
                "input_activation_context": build_input_activation_context(
                    hypothesis=hypothesis_text,
                    designed_sentences=sentences,
                ),
                "input_boundary_context": build_input_boundary_context(
                    hypothesis=hypothesis_text,
                    boundary_sentences=boundary_sentences,
                ),
                "sentence_results": activation_metrics["sentence_results"],
                "non_zero_count": activation_metrics["non_zero_count"],
                "total_sentences": activation_metrics["total_sentences"],
                "score_non_zero_rate": activation_metrics["score_non_zero_rate"],
                "mean_summary_activation": activation_metrics["mean_summary_activation"],
                "max_summary_activation": activation_metrics["max_summary_activation"],
                "boundary_sentence_results": boundary_metrics["sentence_results"],
                "boundary_non_zero_count": boundary_metrics["non_zero_count"],
                "total_boundary_sentences": boundary_metrics["total_sentences"],
                "score_boundary_non_zero_rate": boundary_metrics["score_non_zero_rate"],
                "boundary_non_activation_count": boundary_non_activation_count,
                "score_boundary_non_activation_rate": boundary_non_activation_rate,
                "mean_boundary_summary_activation": boundary_metrics["mean_summary_activation"],
                "max_boundary_summary_activation": boundary_metrics["max_summary_activation"],
            }
        )

    if hypothesis_results:
        overall_score = sum(item["score_non_zero_rate"] for item in hypothesis_results) / len(hypothesis_results)
        boundary_values = [
            item["score_boundary_non_activation_rate"]
            for item in hypothesis_results
            if item.get("score_boundary_non_activation_rate") is not None
        ]
        overall_boundary_score = (
            sum(boundary_values) / len(boundary_values) if boundary_values else None
        )
    else:
        overall_score = 0.0
        overall_boundary_score = None

    return {
        "side": "input",
        "non_zero_threshold": non_zero_threshold,
        "hypothesis_results": hypothesis_results,
        "overall_score_non_zero_rate": overall_score,
        "overall_score_boundary_non_activation_rate": overall_boundary_score,
    }
