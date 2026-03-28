from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from function import build_default_sae_path, build_feature_dir
from function import TokenUsageAccumulator, call_llm_stream, extract_json_object, read_api_key
from model_with_sae import ModelWithSAEModule
from neuronpedia_feature_api import extract_explanations, fetch_feature_json
from experiments_design import generate_boundary_contexts
from prompts.experiments_design_prompt import build_system_prompt, build_user_prompt
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from support_info.llm_api_info import base_url as DEFAULT_BASE_URL
from support_info.llm_api_info import model_name as DEFAULT_MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _resolve_workflow_final_result_path(workflow_path: Path) -> Path:
    if workflow_path.is_file():
        return workflow_path
    if not workflow_path.is_dir():
        raise FileNotFoundError(f"Workflow path does not exist: {workflow_path}")

    final_dir = workflow_path / "final_result"
    candidates = sorted(final_dir.glob("*-final-result.json")) if final_dir.exists() else []
    if not candidates:
        candidates = sorted(workflow_path.glob("**/*-final-result.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find '*-final-result.json' under workflow path: {workflow_path}"
        )
    return candidates[-1]


def _resolve_workflow_path_from_args(args: argparse.Namespace) -> Path:
    if args.workflow_path:
        return Path(str(args.workflow_path))

    missing: List[str] = []
    if args.layer_id is None:
        missing.append("--layer-id")
    if args.feature_id is None:
        missing.append("--feature-id")
    if args.timestamp is None:
        missing.append("--timestamp")
    if missing:
        raise ValueError(
            "Missing workflow locator arguments. Provide --workflow-path, or provide all of: "
            + ", ".join(missing)
        )

    return (
        build_feature_dir(
            layer_id=str(args.layer_id).strip(),
            feature_id=str(args.feature_id).strip(),
            logs_root=Path(str(args.logs_root)),
        )
        / str(args.timestamp).strip()
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_candidate_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def _normalize_cached_input_hypothesis(
    cached: Dict[str, Any],
    *,
    source: str,
) -> Optional[Dict[str, Any]]:
    hypothesis = str(cached.get("hypothesis", "")).strip()
    if not hypothesis:
        return None
    score_non_zero_raw = cached.get("score_non_zero_rate")
    score_boundary_raw = cached.get("score_boundary_non_activation_rate")
    score_non_zero = _safe_float(score_non_zero_raw, 0.0) if score_non_zero_raw is not None else None
    score_boundary = _safe_float(score_boundary_raw, 0.0) if score_boundary_raw is not None else None
    combined_raw = cached.get("combined_score")
    combined_score = (
        _safe_float(combined_raw, 0.0)
        if combined_raw is not None
        else _safe_float(score_non_zero, 0.0) + _safe_float(score_boundary, 0.0)
    )
    return {
        "source": source,
        "hypothesis_index": _safe_int(cached.get("hypothesis_index"), 0),
        "hypothesis": hypothesis,
        "round_index": _safe_int(cached.get("round_index"), 0),
        "round_id": str(cached.get("round_id", "")).strip() or None,
        "score_name": "combined_input_score",
        "score_value": combined_score,
        "score_non_zero_rate": score_non_zero,
        "score_boundary_non_activation_rate": score_boundary,
        "combined_score": combined_score,
    }


def _load_input_hypothesis_cache(
    *,
    final_result_payload: Dict[str, Any],
    workflow_timestamp_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    candidate_paths: List[Path] = []
    configured_path = str(final_result_payload.get("input_side_hypothesis_cache_path", "")).strip()
    if configured_path:
        candidate_paths.append(_resolve_candidate_path(configured_path, base_dir=PROJECT_ROOT))

    layer_id = str(final_result_payload.get("layer_id", "")).strip()
    feature_id = str(final_result_payload.get("feature_id", "")).strip()
    if layer_id and feature_id:
        candidate_paths.append(
            workflow_timestamp_dir / f"layer{layer_id}-feature{feature_id}-input-side-hypotheses-cache.json"
        )

    seen: set[str] = set()
    deduped_paths: List[Path] = []
    for candidate in candidate_paths:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_paths.append(candidate)

    for path in deduped_paths:
        if not path.exists():
            continue
        payload = _load_json(path)
        if isinstance(payload, dict):
            return payload, path
    return None, None


def _pick_best_input_hypothesis_from_workflow(
    *,
    final_result_payload: Dict[str, Any],
    workflow_timestamp_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Path]]:
    best_raw = final_result_payload.get("input_side_best_hypothesis")
    if isinstance(best_raw, dict):
        normalized = _normalize_cached_input_hypothesis(
            best_raw,
            source="workflow_final_result.input_side_best_hypothesis",
        )
        if normalized is not None:
            return normalized, best_raw, None

    cache_payload, cache_path = _load_input_hypothesis_cache(
        final_result_payload=final_result_payload,
        workflow_timestamp_dir=workflow_timestamp_dir,
    )
    if isinstance(cache_payload, dict):
        cache_best = cache_payload.get("best_hypothesis")
        if isinstance(cache_best, dict):
            normalized = _normalize_cached_input_hypothesis(
                cache_best,
                source="workflow_input_side_cache.best_hypothesis",
            )
            if normalized is not None:
                return normalized, cache_best, cache_path
    return None, None, cache_path


def _pick_best_hypothesis_from_refined(
    refined_payload: Dict[str, Any],
    *,
    side: str,
) -> Optional[Dict[str, Any]]:
    refined_section = refined_payload.get("refined_hypotheses", {})
    if not isinstance(refined_section, dict):
        return None
    items_raw = refined_section.get(side, [])
    items = [item for item in items_raw if isinstance(item, dict)]
    if not items:
        return None
    items.sort(
        key=lambda item: (
            -_safe_float(item.get("score_value"), 0.0),
            _safe_int(item.get("hypothesis_index"), 10**9),
        )
    )
    best = items[0]
    return {
        "source": "refined_hypotheses",
        "hypothesis_index": _safe_int(best.get("hypothesis_index"), 0),
        "hypothesis": str(best.get("refined_hypothesis", "")).strip()
        or str(best.get("original_hypothesis", "")).strip(),
        "score_name": str(best.get("score_name", "")).strip(),
        "score_value": _safe_float(best.get("score_value"), 0.0),
    }


def _pick_best_hypothesis_from_execution(
    execution_payload: Dict[str, Any],
    *,
    side: str,
) -> Optional[Dict[str, Any]]:
    if side == "input":
        section = execution_payload.get("input_side_execution", {})
    else:
        section = execution_payload.get("output_side_execution", {})
    if not isinstance(section, dict):
        return None
    if side == "input":
        score_name = "score_non_zero_rate"
    else:
        score_name = str(section.get("output_score_name", "score_blind_accuracy")).strip() or "score_blind_accuracy"
    results_raw = section.get("hypothesis_results", [])
    results = [item for item in results_raw if isinstance(item, dict)]
    if not results:
        return None
    def _item_score(item: Dict[str, Any]) -> float:
        return _safe_float(item.get(score_name, item.get("score_primary", item.get("score", 0.0))), 0.0)

    results.sort(
        key=lambda item: (
            -_item_score(item),
            _safe_int(item.get("hypothesis_index"), 10**9),
        )
    )
    best = results[0]
    return {
        "source": "experiments_execution",
        "hypothesis_index": _safe_int(best.get("hypothesis_index"), 0),
        "hypothesis": str(best.get("hypothesis", "")).strip(),
        "score_name": score_name,
        "score_value": _item_score(best),
    }


def _pick_best_hypothesis_from_final_result(
    final_result_payload: Dict[str, Any],
    *,
    side: str,
) -> Optional[Dict[str, Any]]:
    key = "input_side_final_hypotheses" if side == "input" else "output_side_final_hypotheses"
    hypotheses_raw = final_result_payload.get(key, [])
    hypotheses = [str(item).strip() for item in hypotheses_raw if str(item).strip()]
    if not hypotheses:
        return None
    return {
        "source": "final_result_fallback",
        "hypothesis_index": 1,
        "hypothesis": hypotheses[0],
        "score_name": "unknown",
        "score_value": None,
    }


def _pick_best_hypothesis(
    *,
    final_result_payload: Dict[str, Any],
    refined_payload: Optional[Dict[str, Any]],
    execution_payload: Optional[Dict[str, Any]],
    side: str,
) -> Dict[str, Any]:
    if execution_payload is not None:
        result = _pick_best_hypothesis_from_execution(execution_payload, side=side)
        if result is not None:
            return result
    if refined_payload is not None:
        result = _pick_best_hypothesis_from_refined(refined_payload, side=side)
        if result is not None:
            return result
    result = _pick_best_hypothesis_from_final_result(final_result_payload, side=side)
    if result is not None:
        return result
    raise ValueError(f"No {side}-side hypothesis found in workflow artifacts.")


def _run_command(cmd: Sequence[str], *, cwd: Path) -> None:
    process = subprocess.run(list(cmd), cwd=str(cwd), text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit={process.returncode}): {' '.join(cmd)}")


def _log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [final-eval] {message}", flush=True)


def _run_command_with_progress(
    cmd: Sequence[str],
    *,
    cwd: Path,
    step_name: str,
    heartbeat_seconds: int = 60,
) -> None:
    _log_progress(f"Start step: {step_name}")
    _log_progress(f"Command: {' '.join(cmd)}")

    started_at = time.monotonic()
    process = subprocess.Popen(list(cmd), cwd=str(cwd), text=True)
    last_heartbeat = started_at

    while True:
        return_code = process.poll()
        if return_code is not None:
            elapsed = int(time.monotonic() - started_at)
            if return_code != 0:
                raise RuntimeError(
                    f"Step failed: {step_name} (exit={return_code}, elapsed={elapsed}s). "
                    f"Command: {' '.join(cmd)}"
                )
            _log_progress(f"Finished step: {step_name} (elapsed={elapsed}s)")
            return

        now = time.monotonic()
        if heartbeat_seconds > 0 and (now - last_heartbeat) >= heartbeat_seconds:
            elapsed = int(now - started_at)
            _log_progress(f"Still running: {step_name} (elapsed={elapsed}s)")
            last_heartbeat = now
        time.sleep(1.0)


def _neuronpedia_input_eval_cache_path(*, logs_root: Path, layer_id: str, feature_id: int) -> Path:
    return (
        build_feature_dir(
            layer_id=str(layer_id).strip(),
            feature_id=str(feature_id).strip(),
            logs_root=logs_root,
        )
        / "neuronpedia-input-eval-cache.json"
    )


def _build_neuronpedia_input_eval_cache_signature(
    *,
    model_id: str,
    source: str,
    feature_id: int,
    width: str,
    sae_identity: str,
    max_explanations: int,
    non_activation_context_count: int,
    non_zero_threshold: float,
    activation_count_per_reference: int,
    boundary_count_per_reference: int,
    input_llm_model: str,
    input_base_url: str,
    input_llm_temperature: float,
    input_llm_max_tokens: int,
) -> str:
    payload = {
        "input_reference_generation_mode": "workflow_isomorphic_llm_v1",
        "model_id": str(model_id).strip(),
        "source": str(source).strip(),
        "feature_id": int(feature_id),
        "width": str(width).strip(),
        "sae_identity": str(sae_identity).strip(),
        "max_explanations": int(max_explanations),
        "non_activation_context_count": int(non_activation_context_count),
        "non_zero_threshold": float(non_zero_threshold),
        "activation_count_per_reference": int(activation_count_per_reference),
        "boundary_count_per_reference": int(boundary_count_per_reference),
        "input_llm_model": str(input_llm_model).strip(),
        "input_base_url": str(input_base_url).strip(),
        "input_llm_temperature": float(input_llm_temperature),
        "input_llm_max_tokens": int(input_llm_max_tokens),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_neuronpedia_input_eval_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": {}}
    payload = _load_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    return payload


def _save_neuronpedia_input_eval_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_source(layer_id: str, sae_name: str, width: str) -> str:
    return f"{layer_id}-{sae_name}-{width}"


def _normalize_sentence(text: str) -> str:
    stripped = text.strip().strip('"').strip("'")
    stripped = re.sub(r"^\d+[\).\s-]+", "", stripped)
    return " ".join(stripped.split())


def _parse_sentence_list(raw_output: str, expected_count: int) -> List[str]:
    parsed = extract_json_object(raw_output)
    sentences: List[str] = []

    if isinstance(parsed, dict):
        candidate = parsed.get("sentences")
        if isinstance(candidate, list):
            sentences = [
                _normalize_sentence(item)
                for item in candidate
                if isinstance(item, str) and _normalize_sentence(item)
            ]

    if not sentences:
        lines = [line for line in raw_output.splitlines() if line.strip()]
        sentences = [_normalize_sentence(line) for line in lines if _normalize_sentence(line)]

    if not sentences:
        raise ValueError(f"Failed to parse sentences from output: {raw_output}")

    if len(sentences) < expected_count:
        raise ValueError(
            f"Expected {expected_count} sentences, but only parsed {len(sentences)}: {raw_output}"
        )
    return sentences[:expected_count]


def _generate_activation_sentences_with_llm(
    *,
    client: OpenAI,
    model: str,
    reference_explanation: str,
    sentence_count: int,
    max_tokens: int,
    temperature: float,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    reference_index: int,
) -> List[str]:
    if sentence_count <= 0:
        return []

    system_prompt = build_system_prompt("input")
    user_prompt = build_user_prompt(
        side="input",
        hypothesis=reference_explanation,
        num_sentences=sentence_count,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw_output, usage_obj = call_llm_stream(
        client,
        model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage_counts = token_counter.add(usage_obj)
    designed_sentences = _parse_sentence_list(raw_output, expected_count=sentence_count)
    llm_calls.append(
        {
            "call_type": "input_activation_sentence_generation",
            "reference_index": reference_index,
            "reference_explanation": reference_explanation,
            "messages": messages,
            "raw_output": raw_output,
            "usage": usage_counts,
        }
    )
    return designed_sentences


def _generate_sentence_sets_for_reference(
    *,
    client: OpenAI,
    model: str,
    reference_index: int,
    reference_explanation: str,
    activation_count: int,
    boundary_count: int,
    max_tokens: int,
    temperature: float,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    designed = _generate_activation_sentences_with_llm(
        client=client,
        model=model,
        reference_explanation=reference_explanation,
        sentence_count=max(activation_count, 0),
        max_tokens=max_tokens,
        temperature=temperature,
        token_counter=token_counter,
        llm_calls=llm_calls,
        reference_index=reference_index,
    )
    boundary = generate_boundary_contexts(
        client=client,
        model=model,
        explanation=reference_explanation,
        boundary_case_count=max(boundary_count, 0),
        max_tokens=max_tokens,
        temperature=0.0,
        token_counter=token_counter,
        llm_calls=llm_calls,
        call_metadata={
            "call_type": "input_boundary_context_generation",
            "reference_index": reference_index,
            "reference_explanation": reference_explanation,
        },
    ) if boundary_count > 0 else []
    return designed, boundary


def _extract_max_token(trace: Dict[str, Any]) -> str:
    tokens = trace.get("tokens")
    max_token_index = trace.get("max_token_index")
    if not isinstance(tokens, list) or not isinstance(max_token_index, int):
        return ""
    if max_token_index < 0 or max_token_index >= len(tokens):
        return ""
    token = tokens[max_token_index]
    return token if isinstance(token, str) else str(token)


def _run_sentence_batch_with_sae(
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


def _extract_workflow_input_scores(
    *,
    selected_input: Dict[str, Any],
    selected_input_cache_raw: Optional[Dict[str, Any]],
    execution_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(selected_input_cache_raw, dict):
        non_zero = selected_input_cache_raw.get("score_non_zero_rate")
        boundary = selected_input_cache_raw.get("score_boundary_non_activation_rate")
        if non_zero is not None and boundary is not None:
            return {
                "source": "workflow_input_side_cache.best_hypothesis",
                "score_non_zero_rate": _safe_float(non_zero, 0.0),
                "score_boundary_non_activation_rate": _safe_float(boundary, 0.0),
                "combined_input_score": _safe_float(non_zero, 0.0) + _safe_float(boundary, 0.0),
                "designed_sentences": list(selected_input_cache_raw.get("designed_sentences", []))
                if isinstance(selected_input_cache_raw.get("designed_sentences"), list)
                else [],
                "boundary_sentences": list(selected_input_cache_raw.get("boundary_sentences", []))
                if isinstance(selected_input_cache_raw.get("boundary_sentences"), list)
                else [],
            }

    non_zero_sel = selected_input.get("score_non_zero_rate")
    boundary_sel = selected_input.get("score_boundary_non_activation_rate")
    if non_zero_sel is not None and boundary_sel is not None:
        return {
            "source": str(selected_input.get("source", "selected_input")),
            "score_non_zero_rate": _safe_float(non_zero_sel, 0.0),
            "score_boundary_non_activation_rate": _safe_float(boundary_sel, 0.0),
            "combined_input_score": _safe_float(non_zero_sel, 0.0) + _safe_float(boundary_sel, 0.0),
            "designed_sentences": [],
            "boundary_sentences": [],
        }

    if isinstance(execution_payload, dict):
        input_side = execution_payload.get("input_side_execution", {})
        if isinstance(input_side, dict):
            items_raw = input_side.get("hypothesis_results", [])
            items = [item for item in items_raw if isinstance(item, dict)]
            target_index = _safe_int(selected_input.get("hypothesis_index"), 0)
            target_text = str(selected_input.get("hypothesis", "")).strip()
            for item in items:
                hypothesis_index = _safe_int(item.get("hypothesis_index"), 0)
                hypothesis_text = str(item.get("hypothesis", "")).strip()
                if target_index and hypothesis_index != target_index:
                    continue
                if target_text and hypothesis_text and hypothesis_text != target_text:
                    continue
                non_zero = item.get("score_non_zero_rate")
                boundary = item.get("score_boundary_non_activation_rate")
                if non_zero is None or boundary is None:
                    continue
                return {
                    "source": "round_execution_input_side",
                    "score_non_zero_rate": _safe_float(non_zero, 0.0),
                    "score_boundary_non_activation_rate": _safe_float(boundary, 0.0),
                    "combined_input_score": _safe_float(non_zero, 0.0) + _safe_float(boundary, 0.0),
                    "designed_sentences": list(item.get("designed_sentences", []))
                    if isinstance(item.get("designed_sentences"), list)
                    else [],
                    "boundary_sentences": list(item.get("boundary_sentences", []))
                    if isinstance(item.get("boundary_sentences"), list)
                    else [],
                }

    raise ValueError(
        "Failed to recover workflow cached input-side SAE scores "
        "(score_non_zero_rate / score_boundary_non_activation_rate)."
    )


def _evaluate_neuronpedia_input_with_sae(
    *,
    model_id: str,
    source: str,
    feature_id: int,
    width: str,
    sae_identity: str,
    module: ModelWithSAEModule,
    non_zero_threshold: float,
    max_explanations: int,
    non_activation_context_count: int,
    activation_count_per_reference: int,
    boundary_count_per_reference: int,
    input_llm_model: str,
    input_base_url: str,
    input_api_key_file: str,
    input_llm_temperature: float,
    input_llm_max_tokens: int,
    neuronpedia_api_key: Optional[str],
    neuronpedia_timeout: int,
    cache_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    signature = _build_neuronpedia_input_eval_cache_signature(
        model_id=model_id,
        source=source,
        feature_id=feature_id,
        width=width,
        sae_identity=sae_identity,
        max_explanations=max_explanations,
        non_activation_context_count=non_activation_context_count,
        non_zero_threshold=non_zero_threshold,
        activation_count_per_reference=activation_count_per_reference,
        boundary_count_per_reference=boundary_count_per_reference,
        input_llm_model=input_llm_model,
        input_base_url=input_base_url,
        input_llm_temperature=input_llm_temperature,
        input_llm_max_tokens=input_llm_max_tokens,
    )
    cache_payload = _load_neuronpedia_input_eval_cache(cache_path)
    cache_entries = cache_payload.get("entries", {})
    if isinstance(cache_entries, dict):
        cached = cache_entries.get(signature)
        if isinstance(cached, dict):
            cached_result = dict(cached.get("result", cached))
            if cached_result:
                cached_result["cache_signature"] = signature
                return cached_result, True

    payload = fetch_feature_json(
        model_id=model_id,
        source=source,
        feature_id=str(feature_id),
        api_key=neuronpedia_api_key,
        timeout=neuronpedia_timeout,
    )
    reference_explanations = extract_explanations(payload, limit=max(1, int(max_explanations)))
    if not reference_explanations:
        raise ValueError("No explanation found in Neuronpedia response.")

    input_api_key = read_api_key(input_api_key_file)
    llm_client = OpenAI(
        base_url=input_base_url,
        api_key=input_api_key,
    )
    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    details: List[Dict[str, Any]] = []
    for ref_index, explanation in enumerate(reference_explanations, start=1):
        designed_sentences, boundary_sentences = _generate_sentence_sets_for_reference(
            client=llm_client,
            model=input_llm_model,
            reference_index=ref_index,
            reference_explanation=explanation,
            activation_count=activation_count_per_reference,
            boundary_count=boundary_count_per_reference,
            max_tokens=input_llm_max_tokens,
            temperature=input_llm_temperature,
            token_counter=token_counter,
            llm_calls=llm_calls,
        )
        activation_metrics = _run_sentence_batch_with_sae(
            module=module,
            sentences=designed_sentences,
            non_zero_threshold=non_zero_threshold,
        )
        boundary_metrics = _run_sentence_batch_with_sae(
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
        score_non_zero_rate = _safe_float(activation_metrics.get("score_non_zero_rate"), 0.0)
        score_boundary_non_activation_rate = (
            _safe_float(boundary_non_activation_rate, 0.0)
            if boundary_non_activation_rate is not None
            else None
        )
        combined_input_score = (
            score_non_zero_rate + score_boundary_non_activation_rate
            if score_boundary_non_activation_rate is not None
            else None
        )
        details.append(
            {
                "reference_explanation": explanation,
                "designed_sentences": designed_sentences,
                "boundary_sentences": boundary_sentences,
                "score_non_zero_rate": score_non_zero_rate,
                "score_boundary_non_activation_rate": score_boundary_non_activation_rate,
                "combined_input_score": combined_input_score,
                "activation_metrics": activation_metrics,
                "boundary_metrics": boundary_metrics,
            }
        )

    non_zero_values = [
        _safe_float(item.get("score_non_zero_rate"), 0.0)
        for item in details
        if item.get("score_non_zero_rate") is not None
    ]
    boundary_values = [
        _safe_float(item.get("score_boundary_non_activation_rate"), 0.0)
        for item in details
        if item.get("score_boundary_non_activation_rate") is not None
    ]
    combined_values = [
        _safe_float(item.get("combined_input_score"), 0.0)
        for item in details
        if item.get("combined_input_score") is not None
    ]
    result = {
        "reference_explanations": reference_explanations,
        "reference_count": len(reference_explanations),
        "reference_details": details,
        "neuronpedia_mean_score_non_zero_rate": (
            (sum(non_zero_values) / len(non_zero_values)) if non_zero_values else None
        ),
        "neuronpedia_mean_score_boundary_non_activation_rate": (
            (sum(boundary_values) / len(boundary_values)) if boundary_values else None
        ),
        "neuronpedia_mean_combined_input_score": (
            (sum(combined_values) / len(combined_values)) if combined_values else None
        ),
        "activation_candidate_pool": [],
        "boundary_candidate_pool": [],
        "reference_sentence_generation_mode": "workflow_isomorphic_llm_v1",
        "input_llm_model": input_llm_model,
        "input_base_url": input_base_url,
        "input_llm_temperature": input_llm_temperature,
        "input_llm_max_tokens": input_llm_max_tokens,
        "llm_calls": llm_calls,
        "llm_token_usage": token_counter.as_dict(),
    }
    cache_entries[signature] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "input_reference_generation_mode": "workflow_isomorphic_llm_v1",
            "model_id": str(model_id).strip(),
            "source": str(source).strip(),
            "feature_id": int(feature_id),
            "width": str(width).strip(),
            "sae_identity": str(sae_identity).strip(),
            "max_explanations": int(max_explanations),
            "non_activation_context_count": int(non_activation_context_count),
            "non_zero_threshold": float(non_zero_threshold),
            "activation_count_per_reference": int(activation_count_per_reference),
            "boundary_count_per_reference": int(boundary_count_per_reference),
            "input_llm_model": str(input_llm_model).strip(),
            "input_base_url": str(input_base_url).strip(),
            "input_llm_temperature": float(input_llm_temperature),
            "input_llm_max_tokens": int(input_llm_max_tokens),
        },
        "result": result,
    }
    cache_payload["entries"] = cache_entries
    _save_neuronpedia_input_eval_cache(cache_path, cache_payload)
    result["cache_signature"] = signature
    return result, False


def _extract_logit_summary(logit_payload: Dict[str, Any]) -> Dict[str, Any]:
    runs = [item for item in logit_payload.get("runs", []) if isinstance(item, dict)]
    positive = [
        _safe_float(item.get("scores", {}).get("positive_topk_increase_ratio"), 0.0)
        for item in runs
    ]
    negative = [
        _safe_float(item.get("scores", {}).get("negative_topk_decrease_ratio"), 0.0)
        for item in runs
    ]
    return {
        "run_count": len(runs),
        "mean_positive_topk_increase_ratio": (sum(positive) / len(positive)) if positive else None,
        "mean_negative_topk_decrease_ratio": (sum(negative) / len(negative)) if negative else None,
    }


def _write_summary_markdown(path: Path, *, payload: Dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    selected = payload.get("selected_hypotheses", {})
    input_eval = payload.get("input_evaluation", {})
    output_eval = payload.get("output_evaluation", {})

    lines: List[str] = []
    lines.append("# Final Explanation Evaluation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- generated_at: {metadata.get('generated_at')}")
    lines.append(f"- model_id: {metadata.get('model_id')}")
    lines.append(f"- layer_id: {metadata.get('layer_id')}")
    lines.append(f"- feature_id: {metadata.get('feature_id')}")
    lines.append(f"- evaluation_timestamp: {metadata.get('evaluation_timestamp')}")
    lines.append(f"- run_mode: {metadata.get('run_mode')}")
    lines.append(f"- workflow_final_result: {metadata.get('workflow_final_result_path')}")
    lines.append("")
    lines.append("## Selected Input Hypothesis")
    lines.append(f"- source: {selected.get('input', {}).get('source')}")
    lines.append(f"- score_name: {selected.get('input', {}).get('score_name')}")
    lines.append(f"- score_value: {selected.get('input', {}).get('score_value')}")
    lines.append("```text")
    lines.append(str(selected.get("input", {}).get("hypothesis", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Selected Output Hypothesis")
    lines.append(f"- source: {selected.get('output', {}).get('source')}")
    lines.append(f"- score_name: {selected.get('output', {}).get('score_name')}")
    lines.append(f"- score_value: {selected.get('output', {}).get('score_value')}")
    lines.append("```text")
    lines.append(str(selected.get("output", {}).get("hypothesis", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Input-side Metrics")
    lines.append(f"- status: {input_eval.get('status')}")
    lines.append(f"- method: {input_eval.get('method')}")
    lines.append(f"- used_workflow_cached_scores: {input_eval.get('used_workflow_cached_scores')}")
    lines.append(f"- used_neuronpedia_input_eval_cache: {input_eval.get('used_neuronpedia_input_eval_cache')}")
    lines.append(f"- neuronpedia_input_eval_cache_path: {input_eval.get('neuronpedia_input_eval_cache_path')}")
    lines.append(f"- workflow_score_non_zero_rate: {input_eval.get('workflow_score_non_zero_rate')}")
    lines.append(
        f"- workflow_score_boundary_non_activation_rate: {input_eval.get('workflow_score_boundary_non_activation_rate')}"
    )
    lines.append(
        f"- neuronpedia_mean_score_non_zero_rate: {input_eval.get('neuronpedia_mean_score_non_zero_rate')}"
    )
    lines.append(
        "- neuronpedia_mean_score_boundary_non_activation_rate: "
        f"{input_eval.get('neuronpedia_mean_score_boundary_non_activation_rate')}"
    )
    lines.append(
        f"- relative_quality_score_non_zero_rate: {input_eval.get('relative_quality_score_non_zero_rate')}"
    )
    lines.append(
        "- relative_quality_score_boundary_non_activation_rate: "
        f"{input_eval.get('relative_quality_score_boundary_non_activation_rate')}"
    )
    lines.append(f"- workflow_combined_input_score: {input_eval.get('workflow_combined_input_score')}")
    lines.append(f"- neuronpedia_mean_combined_input_score: {input_eval.get('neuronpedia_mean_combined_input_score')}")
    lines.append(f"- relative_quality_combined_input_score: {input_eval.get('relative_quality_combined_input_score')}")
    lines.append(f"- reference_count: {input_eval.get('reference_count')}")
    lines.append("")
    lines.append("## Output-side Metrics")
    lines.append(f"- mode: {output_eval.get('mode')}")
    lines.append(f"- score_summary: {json.dumps(output_eval.get('summary', {}), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Paths")
    lines.append(f"- input_result_json: {payload.get('paths', {}).get('input_result_json')}")
    lines.append(f"- output_result_json: {payload.get('paths', {}).get('output_result_json')}")
    lines.append(f"- final_summary_json: {payload.get('paths', {}).get('final_summary_json')}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read workflow_runner outputs from logs, select top-scoring final input/output hypotheses, "
            "run final evaluation, and save summary metrics."
        )
    )
    parser.add_argument(
        "--workflow-path",
        default=None,
        help=(
            "Path to workflow final-result json file, or workflow timestamp directory. "
            "If omitted, path is composed as logs/layer-{layer-id}/feature-{feature-id}/{timestamp}."
        ),
    )
    parser.add_argument("--layer-id", default=None, help="Layer id used to compose workflow path.")
    parser.add_argument("--feature-id", type=int, default=None, help="Feature id used to compose workflow path.")
    parser.add_argument("--timestamp", default=None, help="Workflow timestamp directory name used to compose workflow path.")
    parser.add_argument("--logs-root", default=str(PROJECT_ROOT / "logs"), help="Logs root directory for composed workflow path.")
    parser.add_argument("--width", default="16k", help="SAE width used for source string.")
    parser.add_argument("--sae-name", default="gemmascope-res", help="SAE name used by evaluation outputs.")
    parser.add_argument(
        "--input-output-root",
        default=str(
            PROJECT_ROOT / "explanation_quality_evaluation" / "input-side-evaluation" / "outputs"
        ),
    )
    parser.add_argument(
        "--output-output-root",
        default=str(
            PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"
        ),
    )

    parser.add_argument("--input-max-explanations", type=int, default=3)
    parser.add_argument("--input-non-activation-context-count", type=int, default=5)
    parser.add_argument("--input-llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--input-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--input-llm-temperature", type=float, default=0.0)
    parser.add_argument("--input-llm-max-tokens", type=int, default=10000)
    parser.add_argument("--input-disable-boundary-score", action="store_true", help="Deprecated no-op.")
    parser.add_argument(
        "--force-run-input-eval",
        action="store_true",
        help="Deprecated no-op. Input-side evaluation now always runs when run-mode includes input.",
    )
    parser.add_argument("--input-non-zero-threshold", type=float, default=0.0)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--sae-path", default=None, help="Optional SAE path override for input-side SAE scoring.")
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument("--sae-canonical-map", default=str(PROJECT_ROOT / "support_info" / "canonical_map.txt"))
    parser.add_argument("--sae-device", default="auto")

    parser.add_argument("--output-eval-mode", choices=["blind", "logit"], default="blind")
    parser.add_argument("--output-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--output-openai-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-openai-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--run-mode",
        choices=["both", "input", "output", "none"],
        default="both",
        help="Control which side(s) to run. Default is both.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Progress heartbeat interval in seconds while a child process is running.",
    )

    parser.add_argument("--blind-trials", type=int, default=1)
    parser.add_argument("--blind-seed", type=int, default=42)
    parser.add_argument("--blind-num-choices", type=int, default=3)
    parser.add_argument(
        "--blind-use-checkpoint-fallback",
        dest="blind_use_checkpoint_fallback",
        action="store_true",
        help="If cached intervention_output is missing, allow checkpoint generation fallback for blind evaluation.",
    )
    parser.add_argument(
        "--no-blind-checkpoint-fallback",
        dest="blind_use_checkpoint_fallback",
        action="store_false",
        help="Do not generate intervention_output from checkpoint when cache is missing.",
    )
    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--logit-top-k", type=int, default=5)
    parser.add_argument("--logit-target-kl", type=float, nargs="*", default=[0.25, 0.5, -0.25, -0.5])
    parser.add_argument("--logit-judge-max-tokens", type=int, default=10000)
    parser.set_defaults(blind_use_checkpoint_fallback=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log_progress("Resolving workflow artifacts")
    workflow_path = _resolve_workflow_path_from_args(args)
    final_result_path = _resolve_workflow_final_result_path(workflow_path)
    final_result = _load_json(final_result_path)

    layer_id = str(final_result.get("layer_id", "")).strip()
    feature_id_str = str(final_result.get("feature_id", "")).strip()
    model_id = str(final_result.get("model_id", "gemma-2-2b")).strip()
    executed_rounds = _safe_int(final_result.get("executed_rounds"), 0)
    if not layer_id or not feature_id_str:
        raise ValueError(f"Invalid workflow final result metadata: {final_result_path}")
    feature_id = int(feature_id_str)
    if args.layer_id is not None and str(args.layer_id).strip() != layer_id:
        raise ValueError(
            f"Layer id mismatch: args.layer_id={args.layer_id} vs final_result.layer_id={layer_id}"
        )
    if args.feature_id is not None and int(args.feature_id) != feature_id:
        raise ValueError(
            f"Feature id mismatch: args.feature_id={args.feature_id} vs final_result.feature_id={feature_id}"
        )

    workflow_timestamp_dir = final_result_path.parents[1]
    evaluation_timestamp = str(args.timestamp).strip() if args.timestamp else workflow_timestamp_dir.name
    if not str(args.timestamp or "").strip():
        _log_progress(
            "No --timestamp provided; using workflow directory name as evaluation timestamp: "
            f"{evaluation_timestamp}"
        )

    refined_payload: Optional[Dict[str, Any]] = None
    execution_payload: Optional[Dict[str, Any]] = None
    target_round = executed_rounds if executed_rounds > 0 else 0
    round_dir = workflow_timestamp_dir / f"round_{target_round}"
    if executed_rounds > 0:
        refined_path = round_dir / f"layer{layer_id}-feature{feature_id}-refined-hypotheses.json"
        if refined_path.exists():
            refined_payload = _load_json(refined_path)
    execution_path = round_dir / f"layer{layer_id}-feature{feature_id}-experiments-execution.json"
    if execution_path.exists():
        execution_payload = _load_json(execution_path)

    selected_input_from_workflow, selected_input_cache_raw, input_cache_path = _pick_best_input_hypothesis_from_workflow(
        final_result_payload=final_result,
        workflow_timestamp_dir=workflow_timestamp_dir,
    )
    if selected_input_from_workflow is not None:
        selected_input = selected_input_from_workflow
    else:
        selected_input = _pick_best_hypothesis(
            final_result_payload=final_result,
            refined_payload=refined_payload,
            execution_payload=execution_payload,
            side="input",
        )
    selected_output = _pick_best_hypothesis(
        final_result_payload=final_result,
        refined_payload=refined_payload,
        execution_payload=execution_payload,
        side="output",
    )
    _log_progress(
        f"Selected hypotheses (input idx={selected_input.get('hypothesis_index')}, "
        f"output idx={selected_output.get('hypothesis_index')})"
    )

    source = _build_source(layer_id=layer_id, sae_name=str(args.sae_name), width=str(args.width))
    output_blind_script = (
        PROJECT_ROOT
        / "explanation_quality_evaluation"
        / "output-side-evaluation"
        / "intervention_blind_score.py"
    )
    output_logit_script = (
        PROJECT_ROOT
        / "explanation_quality_evaluation"
        / "output-side-evaluation"
        / "intervention_logit_topk_score.py"
    )

    run_input = args.run_mode in ("both", "input")
    run_output = args.run_mode in ("both", "output")
    if args.run_mode == "none":
        _log_progress("Run mode is 'none': skip input-side and output-side evaluation execution.")

    input_result_path: Optional[Path] = None
    input_result: Dict[str, Any] = {}
    input_eval_status = "skipped"
    used_cached_input_scores = False
    used_neuronpedia_input_eval_cache = False
    neuronpedia_input_eval_cache_path: Optional[Path] = None
    if run_input:
        workflow_input_scores = _extract_workflow_input_scores(
            selected_input=selected_input,
            selected_input_cache_raw=selected_input_cache_raw,
            execution_payload=execution_payload,
        )
        used_cached_input_scores = True

        workflow_designed = list(workflow_input_scores.get("designed_sentences", []))
        workflow_boundary = list(workflow_input_scores.get("boundary_sentences", []))
        boundary_count = (
            len(workflow_boundary)
            if workflow_boundary
            else max(1, int(args.input_non_activation_context_count))
        )
        activation_count = (
            len(workflow_designed)
            if workflow_designed
            else boundary_count
        )

        _log_progress("Initializing SAE module for Neuronpedia input-side scoring")
        sae_release = str(args.sae_release).strip() if args.sae_release else "gemma-scope-2b-pt-res"
        sae_path = str(args.sae_path).strip() if args.sae_path else build_default_sae_path(
            layer_id=layer_id,
            width=str(args.width),
            release=sae_release,
            average_l0=args.sae_average_l0,
            canonical_map_path=args.sae_canonical_map,
        )[0]
        module = ModelWithSAEModule(
            llm_name=str(args.model_checkpoint_path),
            sae_path=sae_path,
            sae_layer=int(layer_id),
            feature_index=int(feature_id),
            device=str(args.device),
        )
        sae_identity = (
            f"sae_path:{sae_path}"
            if args.sae_path
            else (
                "sae_release:"
                f"{sae_release}|average_l0:{str(args.sae_average_l0)}|canonical_map:{str(args.sae_canonical_map)}"
            )
        )
        neuronpedia_input_eval_cache_path = _neuronpedia_input_eval_cache_path(
            logs_root=Path(str(args.logs_root)),
            layer_id=layer_id,
            feature_id=feature_id,
        )

        _log_progress("Scoring Neuronpedia reference explanations with workflow-isomorphic LLM+SAE input-side method")
        input_api_key_file = (
            str(args.input_api_key_file).strip()
            if str(args.input_api_key_file or "").strip()
            else str(DEFAULT_API_KEY_FILE)
        )
        neuronpedia_eval, used_neuronpedia_input_eval_cache = _evaluate_neuronpedia_input_with_sae(
            model_id=model_id,
            source=source,
            feature_id=feature_id,
            width=str(args.width),
            sae_identity=sae_identity,
            module=module,
            non_zero_threshold=float(args.input_non_zero_threshold),
            max_explanations=int(args.input_max_explanations),
            non_activation_context_count=int(args.input_non_activation_context_count),
            activation_count_per_reference=activation_count,
            boundary_count_per_reference=boundary_count,
            input_llm_model=str(args.input_llm_model),
            input_base_url=str(args.input_base_url),
            input_api_key_file=input_api_key_file,
            input_llm_temperature=float(args.input_llm_temperature),
            input_llm_max_tokens=int(args.input_llm_max_tokens),
            neuronpedia_api_key=args.neuronpedia_api_key,
            neuronpedia_timeout=int(args.neuronpedia_timeout),
            cache_path=neuronpedia_input_eval_cache_path,
        )

        workflow_non_zero = _safe_float(workflow_input_scores.get("score_non_zero_rate"), 0.0)
        workflow_boundary_rate = _safe_float(workflow_input_scores.get("score_boundary_non_activation_rate"), 0.0)
        workflow_combined = _safe_float(workflow_input_scores.get("combined_input_score"), 0.0)
        np_non_zero = neuronpedia_eval.get("neuronpedia_mean_score_non_zero_rate")
        np_boundary_rate = neuronpedia_eval.get("neuronpedia_mean_score_boundary_non_activation_rate")
        np_combined = neuronpedia_eval.get("neuronpedia_mean_combined_input_score")

        input_result = {
            "method": "workflow_style_sae_only",
            "workflow_score_non_zero_rate": workflow_non_zero,
            "workflow_score_boundary_non_activation_rate": workflow_boundary_rate,
            "workflow_combined_input_score": workflow_combined,
            "neuronpedia_mean_score_non_zero_rate": np_non_zero,
            "neuronpedia_mean_score_boundary_non_activation_rate": np_boundary_rate,
            "neuronpedia_mean_combined_input_score": np_combined,
            "relative_quality_score_non_zero_rate": (
                (workflow_non_zero / _safe_float(np_non_zero, 0.0))
                if np_non_zero is not None and _safe_float(np_non_zero, 0.0) > 0
                else None
            ),
            "relative_quality_score_boundary_non_activation_rate": (
                (workflow_boundary_rate / _safe_float(np_boundary_rate, 0.0))
                if np_boundary_rate is not None and _safe_float(np_boundary_rate, 0.0) > 0
                else None
            ),
            "relative_quality_combined_input_score": (
                (workflow_combined / _safe_float(np_combined, 0.0))
                if np_combined is not None and _safe_float(np_combined, 0.0) > 0
                else None
            ),
            "reference_count": neuronpedia_eval.get("reference_count"),
            "reference_details": neuronpedia_eval.get("reference_details"),
            "workflow_input_score_source": workflow_input_scores.get("source"),
            "workflow_input_score_details": workflow_input_scores,
            "neuronpedia_evaluation": neuronpedia_eval,
            "used_neuronpedia_input_eval_cache": used_neuronpedia_input_eval_cache,
            "neuronpedia_input_eval_cache_path": (
                str(neuronpedia_input_eval_cache_path) if neuronpedia_input_eval_cache_path is not None else None
            ),
        }
        input_result_path = (
            Path(args.input_output_root)
            / str(args.sae_name)
            / f"layer-{layer_id}"
            / f"feature-{feature_id}"
            / evaluation_timestamp
            / "result.json"
        )
        input_result_path.parent.mkdir(parents=True, exist_ok=True)
        input_result_path.write_text(json.dumps(input_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        input_eval_status = "completed"
    else:
        _log_progress("Skip input-side evaluation.")

    output_result_path: Optional[Path] = None
    output_payload: Dict[str, Any] = {}
    output_summary: Dict[str, Any] = {}
    if run_output:
        if args.output_eval_mode == "blind":
            output_cmd: List[str] = [
                sys.executable,
                str(output_blind_script),
                "--layer-id",
                layer_id,
                "--feature-id",
                str(feature_id),
                "--width",
                str(args.width),
                "--sae-name",
                str(args.sae_name),
                "--output-root",
                str(args.output_output_root),
                "--timestamp",
                evaluation_timestamp,
                "--explanation",
                str(selected_output["hypothesis"]),
                "--prefer-existing",
                "--trials",
                str(args.blind_trials),
                "--seed",
                str(args.blind_seed),
                "--num-choices",
                str(args.blind_num_choices),
                "--openai-model",
                str(args.output_openai_model),
                "--openai-base-url",
                str(args.output_openai_base_url),
                "--device",
                str(args.device),
                "--model-checkpoint-path",
                str(args.model_checkpoint_path),
            ]
            if args.output_api_key_file:
                output_cmd.extend(["--api-key-file", str(args.output_api_key_file)])
            if args.sae_release:
                output_cmd.extend(["--sae-release", str(args.sae_release)])
            if args.sae_average_l0:
                output_cmd.extend(["--sae-average-l0", str(args.sae_average_l0)])
            if args.sae_canonical_map:
                output_cmd.extend(["--sae-canonical-map", str(args.sae_canonical_map)])
            if args.blind_use_checkpoint_fallback:
                output_cmd.append("--use-checkpoint")
            _run_command_with_progress(
                output_cmd,
                cwd=PROJECT_ROOT,
                step_name="output-side evaluation (blind)",
                heartbeat_seconds=int(args.heartbeat_seconds),
            )
            output_result_path = (
                Path(args.output_output_root)
                / str(args.sae_name)
                / f"layer-{layer_id}"
                / f"feature-{feature_id}"
                / evaluation_timestamp
                / "intervention_blind_score.json"
            )
            output_payload = _load_json(output_result_path)
            output_summary = {
                "score_blind_accuracy": output_payload.get("score", {}).get("score"),
                "blind_judge_successes": output_payload.get("score", {}).get("successes"),
                "blind_judge_trials": output_payload.get("score", {}).get("trials"),
            }
        else:
            output_cmd = [
                sys.executable,
                str(output_logit_script),
                "--layer-id",
                layer_id,
                "--feature-id",
                str(feature_id),
                "--width",
                str(args.width),
                "--sae-name",
                str(args.sae_name),
                "--output-root",
                str(args.output_output_root),
                "--timestamp",
                evaluation_timestamp,
                "--explanation",
                str(selected_output["hypothesis"]),
                "--top-k",
                str(args.logit_top_k),
                "--judge-max-tokens",
                str(args.logit_judge_max_tokens),
                "--openai-model",
                str(args.output_openai_model),
                "--openai-base-url",
                str(args.output_openai_base_url),
                "--prefer-existing",
            ]
            output_cmd.extend(["--target-kl", *[str(float(kl)) for kl in args.logit_target_kl]])
            if args.output_api_key_file:
                output_cmd.extend(["--api-key-file", str(args.output_api_key_file)])
            if args.sae_release:
                output_cmd.extend(["--sae-release", str(args.sae_release)])
            if args.sae_average_l0:
                output_cmd.extend(["--sae-average-l0", str(args.sae_average_l0)])
            if args.sae_canonical_map:
                output_cmd.extend(["--sae-canonical-map", str(args.sae_canonical_map)])
            _run_command_with_progress(
                output_cmd,
                cwd=PROJECT_ROOT,
                step_name="output-side evaluation (logit)",
                heartbeat_seconds=int(args.heartbeat_seconds),
            )
            output_result_path = (
                Path(args.output_output_root)
                / str(args.sae_name)
                / f"layer-{layer_id}"
                / f"feature-{feature_id}"
                / evaluation_timestamp
                / "intervention_logit_topk_score.json"
            )
            output_payload = _load_json(output_result_path)
            output_summary = _extract_logit_summary(output_payload)
    else:
        _log_progress("Skip output-side evaluation.")

    final_summary_path = (
        workflow_timestamp_dir
        / "final_result"
        / f"layer{layer_id}-feature{feature_id}-final-evaluation.json"
    )
    final_summary_md_path = final_summary_path.with_suffix(".md")
    _log_progress("Building final merged summary payload")

    summary_payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_id": model_id,
            "layer_id": layer_id,
            "feature_id": feature_id,
            "evaluation_timestamp": evaluation_timestamp,
            "run_mode": str(args.run_mode),
            "workflow_final_result_path": str(final_result_path),
        },
        "selected_hypotheses": {
            "input": selected_input,
            "output": selected_output,
        },
        "input_evaluation": {
            "status": input_eval_status,
            "method": input_result.get("method") if run_input else None,
            "used_workflow_cached_scores": used_cached_input_scores,
            "used_neuronpedia_input_eval_cache": used_neuronpedia_input_eval_cache if run_input else None,
            "workflow_score_non_zero_rate": (
                input_result.get("workflow_score_non_zero_rate") if run_input else None
            ),
            "workflow_score_boundary_non_activation_rate": (
                input_result.get("workflow_score_boundary_non_activation_rate") if run_input else None
            ),
            "workflow_combined_input_score": (
                input_result.get("workflow_combined_input_score") if run_input else None
            ),
            "neuronpedia_mean_score_non_zero_rate": (
                input_result.get("neuronpedia_mean_score_non_zero_rate") if run_input else None
            ),
            "neuronpedia_mean_score_boundary_non_activation_rate": (
                input_result.get("neuronpedia_mean_score_boundary_non_activation_rate") if run_input else None
            ),
            "neuronpedia_mean_combined_input_score": (
                input_result.get("neuronpedia_mean_combined_input_score") if run_input else None
            ),
            "relative_quality_score_non_zero_rate": (
                input_result.get("relative_quality_score_non_zero_rate") if run_input else None
            ),
            "relative_quality_score_boundary_non_activation_rate": (
                input_result.get("relative_quality_score_boundary_non_activation_rate") if run_input else None
            ),
            "relative_quality_combined_input_score": (
                input_result.get("relative_quality_combined_input_score") if run_input else None
            ),
            "reference_count": input_result.get("reference_count") if run_input else None,
            "neuronpedia_input_eval_cache_path": (
                input_result.get("neuronpedia_input_eval_cache_path") if run_input else None
            ),
            "raw_result": (
                {
                    "input_result": input_result,
                    "selected_input_cache_entry": selected_input_cache_raw,
                }
                if run_input and used_cached_input_scores
                else (input_result if run_input else None)
            ),
        },
        "output_evaluation": {
            "mode": str(args.output_eval_mode),
            "status": "completed" if run_output else "skipped",
            "summary": output_summary if run_output else {},
            "raw_result": output_payload if run_output else None,
        },
        "paths": {
            "input_result_json": str(input_result_path) if input_result_path is not None else None,
            "output_result_json": str(output_result_path) if output_result_path is not None else None,
            "final_summary_json": str(final_summary_path),
            "final_summary_md": str(final_summary_md_path),
        },
    }
    _log_progress("Writing final summary json and markdown")
    final_summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(final_summary_md_path, payload=summary_payload)

    print(
        json.dumps(
            {
                "selected_input_hypothesis": selected_input,
                "selected_output_hypothesis": selected_output,
                "input_metrics": summary_payload["input_evaluation"],
                "output_metrics": summary_payload["output_evaluation"]["summary"],
                "summary_json": str(final_summary_path),
                "summary_md": str(final_summary_md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
