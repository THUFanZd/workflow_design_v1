from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from experiments_design import _write_markdown_log as write_experiments_markdown
from experiments_design import design_hypothesis_experiments
from experiments_execution import _write_markdown_log as write_execution_markdown
from experiments_execution import execute_hypothesis_experiments
from experiments_execution_output import KL_DIV_VALUES_DEFAULT
from function import (
    DEFAULT_CANONICAL_MAP_PATH,
    TokenUsageAccumulator,
    build_default_sae_path,
    build_round_dir,
    normalize_round_id,
)
from hypothesis_merge import merge_refined_hypotheses
from hypothesis_memory import build_hypothesis_memory, write_hypothesis_memory_markdown
from hypothesis_refinement import refine_hypotheses
from initial_hypothesis_generation import generate_initial_hypotheses
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from support_info.llm_api_info import base_url as DEFAULT_BASE_URL
from support_info.llm_api_info import model_name as DEFAULT_MODEL_NAME
from model_with_sae import ModelWithSAEModule
from neuronpedia_feature_api import fetch_and_parse_feature_observation


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _round_id_from_index(round_index: int) -> str:
    return normalize_round_id(None, round_index=round_index)


def _round_dir(*, layer_id: str, feature_id: str, timestamp: str, round_index: int) -> Path:
    return build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        round_id=_round_id_from_index(round_index),
        round_index=round_index,
    )


def _artifact_json_path(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    round_index: int,
    kind: str,
) -> Path:
    stem = f"layer{layer_id}-feature{feature_id}"
    filename_map = {
        "observation_input": f"{stem}-observation-input.json",
        "initial_hypotheses": f"{stem}-initial-hypotheses.json",
        "experiments": f"{stem}-experiments.json",
        "experiments_execution": f"{stem}-experiments-execution.json",
        "memory": f"{stem}-memory.json",
        "refined_hypotheses": f"{stem}-refined-hypotheses.json",
        "merged_hypotheses": f"{stem}-merged-hypotheses.json",
    }
    if kind not in filename_map:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    return _round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        round_index=round_index,
    ) / filename_map[kind]


def _load_json_or_raise(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a dict: {path}")
    return payload


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _result_usage(result: Dict[str, Any]) -> Dict[str, int]:
    usage = result.get("token_usage")
    if isinstance(usage, dict):
        return usage
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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


def _is_input_hypothesis_full_score(item: Dict[str, Any]) -> bool:
    score_non_zero = item.get("score_non_zero_rate")
    score_boundary = item.get("score_boundary_non_activation_rate")
    if score_non_zero is None or score_boundary is None:
        return False
    return _safe_float(score_non_zero, 0.0) >= 1.0 and _safe_float(score_boundary, 0.0) >= 1.0


def _frozen_input_indices_from_execution(execution_result: Dict[str, Any]) -> Set[int]:
    input_exec = execution_result.get("input_side_execution", {})
    if not isinstance(input_exec, dict):
        return set()
    hypothesis_results_raw = input_exec.get("hypothesis_results", [])
    hypothesis_results = (
        [item for item in hypothesis_results_raw if isinstance(item, dict)]
        if isinstance(hypothesis_results_raw, list)
        else []
    )
    frozen: Set[int] = set()
    for item in hypothesis_results:
        index = _safe_int(item.get("hypothesis_index"), 0)
        if index <= 0:
            continue
        if _is_input_hypothesis_full_score(item):
            frozen.add(index)
    return frozen


def _normalize_hypothesis_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value]


def _copy_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _build_filtered_hypotheses_for_active_input(
    *,
    hypotheses_result: Dict[str, Any],
    active_input_indices: Sequence[int],
) -> Dict[str, Any]:
    payload = dict(hypotheses_result)
    input_hypotheses = _normalize_hypothesis_list(hypotheses_result.get("input_side_hypotheses", []))
    input_reasons_raw = hypotheses_result.get("input_side_hypothesis_reasons", [])
    input_reasons = (
        [str(item) if item is not None else "" for item in input_reasons_raw]
        if isinstance(input_reasons_raw, list)
        else []
    )

    filtered_hypotheses: List[str] = []
    filtered_reasons: List[str] = []
    for index in active_input_indices:
        if 1 <= index <= len(input_hypotheses):
            filtered_hypotheses.append(input_hypotheses[index - 1])
            filtered_reasons.append(input_reasons[index - 1] if index - 1 < len(input_reasons) else "")

    payload["input_side_hypotheses"] = filtered_hypotheses
    payload["input_side_hypothesis_reasons"] = filtered_reasons
    return payload


def _merge_input_experiments_from_active_and_frozen(
    *,
    current_hypotheses: Dict[str, Any],
    active_indices: Sequence[int],
    frozen_indices: Set[int],
    active_experiments_result: Dict[str, Any],
    previous_experiments_result: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(active_experiments_result)
    input_hypotheses = _normalize_hypothesis_list(current_hypotheses.get("input_side_hypotheses", []))
    total_input = len(input_hypotheses)
    active_items_raw = active_experiments_result.get("input_side_experiments", [])
    active_items = [item for item in active_items_raw if isinstance(item, dict)] if isinstance(active_items_raw, list) else []
    previous_items_raw = previous_experiments_result.get("input_side_experiments", [])
    previous_items = (
        [item for item in previous_items_raw if isinstance(item, dict)] if isinstance(previous_items_raw, list) else []
    )

    merged_items: List[Optional[Dict[str, Any]]] = [None] * total_input
    for index in frozen_indices:
        if 1 <= index <= len(previous_items):
            carried = _copy_dict(previous_items[index - 1])
            if 1 <= index <= total_input:
                carried["hypothesis"] = input_hypotheses[index - 1]
            merged_items[index - 1] = carried

    for position, index in enumerate(active_indices):
        if not (1 <= index <= total_input):
            continue
        if position >= len(active_items):
            raise ValueError(
                "Active input experiments count does not match expected active hypotheses count."
            )
        generated = _copy_dict(active_items[position])
        generated["hypothesis"] = input_hypotheses[index - 1]
        merged_items[index - 1] = generated

    if any(item is None for item in merged_items):
        raise ValueError("Failed to assemble merged input experiments for all hypotheses.")

    merged["input_side_experiments"] = [item for item in merged_items if isinstance(item, dict)]
    output_hypotheses = _normalize_hypothesis_list(current_hypotheses.get("output_side_hypotheses", []))
    merged["num_hypothesis"] = max(len(input_hypotheses), len(output_hypotheses))
    return merged


def _merge_input_execution_from_active_and_frozen(
    *,
    current_hypotheses: Dict[str, Any],
    active_indices: Sequence[int],
    frozen_indices: Set[int],
    active_execution_result: Dict[str, Any],
    previous_execution_result: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(active_execution_result)
    input_hypotheses = _normalize_hypothesis_list(current_hypotheses.get("input_side_hypotheses", []))
    total_input = len(input_hypotheses)

    active_input_exec = active_execution_result.get("input_side_execution", {})
    active_items_raw = active_input_exec.get("hypothesis_results", []) if isinstance(active_input_exec, dict) else []
    active_items = [item for item in active_items_raw if isinstance(item, dict)] if isinstance(active_items_raw, list) else []

    previous_input_exec = previous_execution_result.get("input_side_execution", {})
    previous_items_raw = previous_input_exec.get("hypothesis_results", []) if isinstance(previous_input_exec, dict) else []
    previous_items = (
        [item for item in previous_items_raw if isinstance(item, dict)] if isinstance(previous_items_raw, list) else []
    )

    merged_items: List[Optional[Dict[str, Any]]] = [None] * total_input
    for index in frozen_indices:
        if 1 <= index <= len(previous_items):
            carried = _copy_dict(previous_items[index - 1])
            carried["hypothesis_index"] = index
            if 1 <= index <= total_input:
                carried["hypothesis"] = input_hypotheses[index - 1]
            merged_items[index - 1] = carried

    for position, index in enumerate(active_indices):
        if not (1 <= index <= total_input):
            continue
        if position >= len(active_items):
            raise ValueError(
                "Active input execution count does not match expected active hypotheses count."
            )
        generated = _copy_dict(active_items[position])
        generated["hypothesis_index"] = index
        generated["hypothesis"] = input_hypotheses[index - 1]
        merged_items[index - 1] = generated

    if any(item is None for item in merged_items):
        raise ValueError("Failed to assemble merged input execution for all hypotheses.")

    boundary_values = [
        _safe_float(item.get("score_boundary_non_activation_rate"), 0.0)
        for item in merged_items
        if isinstance(item, dict) and item.get("score_boundary_non_activation_rate") is not None
    ]
    non_zero_values = [
        _safe_float(item.get("score_non_zero_rate"), 0.0)
        for item in merged_items
        if isinstance(item, dict)
    ]
    merged_input_execution = dict(active_input_exec) if isinstance(active_input_exec, dict) else {}
    merged_input_execution["hypothesis_results"] = [item for item in merged_items if isinstance(item, dict)]
    merged_input_execution["overall_score_non_zero_rate"] = (
        (sum(non_zero_values) / len(non_zero_values)) if non_zero_values else 0.0
    )
    merged_input_execution["overall_score_boundary_non_activation_rate"] = (
        (sum(boundary_values) / len(boundary_values)) if boundary_values else None
    )
    merged["input_side_execution"] = merged_input_execution
    return merged


def _format_elapsed(total_seconds: float) -> str:
    total = max(0, int(total_seconds))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _log_stage(message: str, workflow_start_time: float) -> None:
    elapsed = _format_elapsed(time.perf_counter() - workflow_start_time)
    print(f"[elapsed {elapsed}] {message}")


def _build_input_hypothesis_selection_payload(
    *,
    model_id: str,
    layer_id: str,
    feature_id: str,
    timestamp: str,
    round_executions: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []

    for round_index in sorted(round_executions.keys()):
        execution_result = round_executions.get(round_index, {})
        if not isinstance(execution_result, dict):
            continue
        input_side = execution_result.get("input_side_execution", {})
        if not isinstance(input_side, dict):
            continue

        round_id = _clean_text(execution_result.get("round_id")) or _round_id_from_index(round_index)
        hypothesis_results_raw = input_side.get("hypothesis_results", [])
        hypothesis_results = (
            [item for item in hypothesis_results_raw if isinstance(item, dict)]
            if isinstance(hypothesis_results_raw, list)
            else []
        )
        for item in hypothesis_results:
            score_non_zero_rate = _safe_float(item.get("score_non_zero_rate"), 0.0)
            boundary_score_raw = item.get("score_boundary_non_activation_rate")
            boundary_score_for_ranking = (
                _safe_float(boundary_score_raw, 0.0) if boundary_score_raw is not None else 0.0
            )
            combined_score = score_non_zero_rate + boundary_score_for_ranking
            entries.append(
                {
                    "round_index": round_index,
                    "round_id": round_id,
                    "hypothesis_index": _safe_int(item.get("hypothesis_index"), 0),
                    "hypothesis": _clean_text(item.get("hypothesis")),
                    "score_non_zero_rate": score_non_zero_rate,
                    "score_boundary_non_activation_rate": (
                        _safe_float(boundary_score_raw, 0.0) if boundary_score_raw is not None else None
                    ),
                    "score_boundary_non_activation_rate_for_ranking": boundary_score_for_ranking,
                    "combined_score": combined_score,
                    "non_zero_count": _safe_int(item.get("non_zero_count"), 0),
                    "total_sentences": _safe_int(item.get("total_sentences"), 0),
                    "boundary_non_activation_count": _safe_int(item.get("boundary_non_activation_count"), 0),
                    "total_boundary_sentences": _safe_int(item.get("total_boundary_sentences"), 0),
                    "designed_sentences": (
                        list(item.get("designed_sentences", []))
                        if isinstance(item.get("designed_sentences"), list)
                        else []
                    ),
                    "boundary_sentences": (
                        list(item.get("boundary_sentences", []))
                        if isinstance(item.get("boundary_sentences"), list)
                        else []
                    ),
                    "sentence_results": (
                        list(item.get("sentence_results", []))
                        if isinstance(item.get("sentence_results"), list)
                        else []
                    ),
                    "boundary_sentence_results": (
                        list(item.get("boundary_sentence_results", []))
                        if isinstance(item.get("boundary_sentence_results"), list)
                        else []
                    ),
                    "source_execution_round_path": str(
                        _artifact_json_path(
                            layer_id=layer_id,
                            feature_id=feature_id,
                            timestamp=timestamp,
                            round_index=round_index,
                            kind="experiments_execution",
                        )
                    ),
                }
            )

    ranked_entries = sorted(
        entries,
        key=lambda item: (
            -_safe_float(item.get("combined_score"), 0.0),
            -_safe_int(item.get("round_index"), -1),
            _safe_int(item.get("hypothesis_index"), 10**9),
        ),
    )
    best_hypothesis = ranked_entries[0] if ranked_entries else None

    return {
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": timestamp,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "score_formula": "score_non_zero_rate + score_boundary_non_activation_rate",
        "total_candidates": len(ranked_entries),
        "best_hypothesis": best_hypothesis,
        "hypotheses": ranked_entries,
    }


def _extract_reason_map(hypotheses_result: Dict[str, Any]) -> Dict[str, Any]:
    input_reasons = hypotheses_result.get("input_side_hypothesis_reasons", [])
    output_reasons = hypotheses_result.get("output_side_hypothesis_reasons", [])
    return {
        "input": input_reasons if isinstance(input_reasons, list) else [],
        "output": output_reasons if isinstance(output_reasons, list) else [],
    }


def _to_next_round_hypotheses(
    *,
    refinement_result: Dict[str, Any],
    round_index: int,
) -> Dict[str, Any]:
    return {
        "model_id": _clean_text(refinement_result.get("model_id")),
        "layer_id": _clean_text(refinement_result.get("layer_id")),
        "feature_id": _clean_text(refinement_result.get("feature_id")),
        "timestamp": _clean_text(refinement_result.get("timestamp")),
        "round_id": _round_id_from_index(round_index),
        "input_side_hypotheses": list(refinement_result.get("input_side_hypotheses", [])),
        "input_side_hypothesis_reasons": list(refinement_result.get("input_side_hypothesis_reasons", [])),
        "output_side_hypotheses": list(refinement_result.get("output_side_hypotheses", [])),
        "output_side_hypothesis_reasons": list(refinement_result.get("output_side_hypothesis_reasons", [])),
        "llm_model": refinement_result.get("llm_model"),
        "generation_mode": "refined",
    }


def _same_hypotheses(before: Dict[str, Any], after_refine: Dict[str, Any]) -> bool:
    before_input = [str(x).strip() for x in before.get("input_side_hypotheses", []) if str(x).strip()]
    before_output = [str(x).strip() for x in before.get("output_side_hypotheses", []) if str(x).strip()]
    after_input = [str(x).strip() for x in after_refine.get("input_side_hypotheses", []) if str(x).strip()]
    after_output = [str(x).strip() for x in after_refine.get("output_side_hypotheses", []) if str(x).strip()]
    return before_input == after_input and before_output == after_output


def _should_run(*, start_round: int, start_step: int, round_index: int, step_index: int) -> bool:
    if round_index < start_round:
        return False
    if round_index > start_round:
        return True
    return step_index >= start_step


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAE explanation workflow with baseline phase and refinement rounds (resume supported).",
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id")
    parser.add_argument("--layer-id", required=True, help="Layer id")
    parser.add_argument("--feature-id", required=True, help="Feature id")
    parser.add_argument("--timestamp", default=None, help="Timestamp directory under logs/{layer_id}/{feature_id}/")
    parser.add_argument("--max-rounds", type=int, default=1, help="Maximum refinement rounds (round_1..round_n).")
    parser.add_argument(
        "--start-round",
        type=int,
        default=0,
        help="Round index to start real execution from. round_0 is baseline (not counted in max-rounds).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help=(
            "When --start-round=0, valid steps are 1..5 "
            "(observation, initial, design, execution, memory). "
            "When --start-round>=1 and merge is disabled, valid steps are 1..4 "
            "(refinement, design, execution, memory). "
            "When --start-round>=1 and merge is enabled, valid steps are 1..5 "
            "(refinement, merge, design, execution, memory)."
        ),
    )
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="Reuse artifacts before start point from logs/{layer_id}/{feature_id}/{timestamp}/{round_id}.",
    )

    parser.add_argument("--num-hypothesis", type=int, default=3, help="Hypothesis count n for each side")
    parser.add_argument(
        "--side",
        choices=["input", "output", "both"],
        default="both",
        help="Run only input-side or output-side hypothesis iteration, or both.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["single_call", "iterative"],
        default="single_call",
        help="Initial hypothesis generation mode.",
    )
    parser.add_argument(
        "--num-input-sentences-per-hypothesis",
        type=int,
        default=5,
        help="Input-side designed activation and boundary sentences per hypothesis.",
    )
    parser.add_argument("--top-m", type=int, default=None, help="Refine top-m hypotheses per side (default=all).")
    parser.add_argument(
        "--enable-hypothesis-merge",
        action="store_true",
        help=(
            "After each refinement, add an LLM semantic merge step. "
            "Default off to keep original workflow behavior."
        ),
    )
    parser.add_argument(
        "--history-rounds",
        type=int,
        default=1,
        help=(
            "Use at most previous n historical memory rounds during refinement. "
            "Default: all available historical rounds."
        ),
    )
    parser.add_argument(
        "--history-scope",
        choices=["same_hypothesis", "all_hypotheses"],
        default="same_hypothesis",
        help="Historical memory scope used in refinement.",
    )

    parser.add_argument("--width", default="16k", help="Neuronpedia source width")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)

    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10000)

    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--sae-path", default=None, help="SAE path or sae-lens URI")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument(
        "--sae-canonical-map",
        default=str(DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt for default SAE resolution.",
    )
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--input-non-zero-threshold", type=float, default=0.0)
    parser.add_argument("--output-max-new-tokens", type=int, default=25)
    parser.add_argument("--output-generation-temperature", type=float, default=0.75)
    parser.add_argument("--output-judge-num-choices", type=int, default=3)
    parser.add_argument("--output-judge-trials", type=int, default=1)
    parser.add_argument("--output-judge-seed", type=int, default=42)
    parser.add_argument("--output-judge-temperature", type=float, default=0.0)
    parser.add_argument("--output-judge-max-tokens", type=int, default=10000)
    parser.add_argument("--output-kl-values", type=float, nargs="*", default=KL_DIV_VALUES_DEFAULT)
    parser.add_argument(
        "--output-intervention-method",
        choices=["blind", "logit"],
        default="blind",
        help="Output-side intervention scoring method used in experiments execution.",
    )
    parser.add_argument("--output-logit-top-k", type=int, default=5)
    parser.add_argument("--output-logit-kl-tolerance", type=float, default=0.1)
    parser.add_argument("--output-logit-kl-max-steps", type=int, default=12)
    parser.add_argument("--output-logit-force-refresh-kl-cache", action="store_true")
    parser.add_argument("--output-logit-include-special-tokens", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    workflow_start_time = time.perf_counter()
    merge_enabled = bool(args.enable_hypothesis_merge)

    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be >= 0.")
    if args.start_round < 0:
        raise ValueError("--start-round must be >= 0.")
    if args.start_round == 0 and args.start_step not in (1, 2, 3, 4, 5):
        raise ValueError("When --start-round=0, --start-step must be one of 1/2/3/4/5.")
    if args.start_round >= 1:
        valid_steps = (1, 2, 3, 4, 5) if merge_enabled else (1, 2, 3, 4)
        if args.start_step not in valid_steps:
            if merge_enabled:
                raise ValueError("When --start-round>=1 with merge enabled, --start-step must be one of 1/2/3/4/5.")
            raise ValueError("When --start-round>=1 with merge disabled, --start-step must be one of 1/2/3/4.")
    if args.start_round > args.max_rounds and args.start_round > 0:
        raise ValueError("--start-round cannot be greater than --max-rounds.")
    if args.history_rounds is not None and args.history_rounds < 0:
        raise ValueError("--history-rounds must be >= 0 when provided.")
    if (args.start_round != 0 or args.start_step != 1) and not args.reuse_from_logs:
        raise ValueError("Resume from middle requires --reuse-from-logs.")
    if args.reuse_from_logs and args.timestamp is None:
        raise ValueError("When --reuse-from-logs is set, --timestamp is required.")

    layer_id = str(args.layer_id)
    feature_id = str(args.feature_id)
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    total_tokens = TokenUsageAccumulator()
    run_tokens = TokenUsageAccumulator()
    executed_steps: List[str] = []
    loaded_steps: List[str] = []

    def track_usage(result: Dict[str, Any], *, executed: bool) -> None:
        usage = _result_usage(result)
        total_tokens.add(usage)
        if executed:
            run_tokens.add(usage)

    observation_path = _artifact_json_path(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_index=0,
        kind="observation_input",
    )
    _log_stage("collect observation...", workflow_start_time)
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=1):
        observation = fetch_and_parse_feature_observation(
            model_id=args.model_id,
            layer_id=layer_id,
            feature_id=feature_id,
            width=args.width,
            selection_method=args.selection_method,
            m=args.observation_m,
            n=args.observation_n,
            api_key=args.neuronpedia_api_key,
            timeout=args.neuronpedia_timeout,
            timestamp=ts,
            round_id=_round_id_from_index(0),
        )
        executed_steps.append("round_0_step_1_observation")
    else:
        observation = _load_json_or_raise(observation_path)
        loaded_steps.append("round_0_step_1_observation")

    initial_path = _artifact_json_path(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_index=0,
        kind="initial_hypotheses",
    )
    _log_stage("generate initial hypotheses...", workflow_start_time)
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=2):
        initial_result = generate_initial_hypotheses(
            observation=observation,
            model_id=args.model_id,
            layer_id=layer_id,
            feature_id=feature_id,
            num_hypothesis=args.num_hypothesis,
            generation_mode=args.generation_mode,
            timestamp=ts,
            round_id=_round_id_from_index(0),
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        executed_steps.append("round_0_step_2_initial_hypotheses")
        track_usage(initial_result, executed=True)
    else:
        initial_result = _load_json_or_raise(initial_path)
        loaded_steps.append("round_0_step_2_initial_hypotheses")
        track_usage(initial_result, executed=False)

    current_hypotheses = initial_result
    last_executed_hypotheses = initial_result
    previous_execution_result: Optional[Dict[str, Any]] = None
    previous_memory_result: Optional[Dict[str, Any]] = None
    previous_experiments_result: Optional[Dict[str, Any]] = None
    round_memories: Dict[int, Dict[str, Any]] = {}
    round_refinements: Dict[int, Dict[str, Any]] = {}
    round_merges: Dict[int, Dict[str, Any]] = {}
    round_executions: Dict[int, Dict[str, Any]] = {}
    module: Optional[ModelWithSAEModule] = None
    converged = False
    converged_round: Optional[int] = None
    last_round_executed = 0
    last_output_intervention_method = str(args.output_intervention_method)
    last_output_score_name = "score_blind_accuracy"
    last_output_logit_top_k = args.output_logit_top_k

    def update_last_output_meta(
        execution_result_payload: Dict[str, Any],
        *,
        default_method: str,
        default_score_name: str,
        default_logit_top_k: int,
    ) -> tuple[str, str, int]:
        output_exec_meta = execution_result_payload.get("output_side_execution", {})
        if not isinstance(output_exec_meta, dict):
            return default_method, default_score_name, default_logit_top_k
        method = str(output_exec_meta.get("output_intervention_method", default_method))
        score_name = str(output_exec_meta.get("output_score_name", default_score_name))
        logit_top_k = default_logit_top_k
        if output_exec_meta.get("logit_top_k") is not None:
            try:
                logit_top_k = int(output_exec_meta.get("logit_top_k"))
            except (TypeError, ValueError):
                pass
        return method, score_name, logit_top_k

    # Baseline phase (round_0): initial hypotheses -> design -> execution -> memory.
    baseline_round_id = _round_id_from_index(0)
    baseline_experiments_path = _artifact_json_path(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_index=0,
        kind="experiments",
    )
    _log_stage("design baseline experiments...", workflow_start_time)
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=3):
        baseline_experiments_result = design_hypothesis_experiments(
            hypotheses_result=current_hypotheses,
            num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
            run_side=args.side,
            round_id=baseline_round_id,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        executed_steps.append("round_0_step_3_experiments_design")
        track_usage(baseline_experiments_result, executed=True)
    else:
        baseline_experiments_result = _load_json_or_raise(baseline_experiments_path)
        loaded_steps.append("round_0_step_3_experiments_design")
        track_usage(baseline_experiments_result, executed=False)

    baseline_execution_path = _artifact_json_path(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_index=0,
        kind="experiments_execution",
    )
    _log_stage("execute baseline experiments...", workflow_start_time)
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=4):
        if module is None:
            sae_path = args.sae_path or build_default_sae_path(
                layer_id=layer_id,
                width=args.width,
                release=args.sae_release,
                average_l0=args.sae_average_l0,
                canonical_map_path=args.sae_canonical_map,
            )[0]
            module = ModelWithSAEModule(
                llm_name=args.model_checkpoint_path,
                sae_path=sae_path,
                sae_layer=int(layer_id),
                feature_index=int(feature_id),
                device=args.device,
            )
        baseline_execution_result = execute_hypothesis_experiments(
            experiments_result=baseline_experiments_result,
            module=module,
            run_side=args.side,
            round_id=baseline_round_id,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            input_non_zero_threshold=args.input_non_zero_threshold,
            output_judge_num_choices=args.output_judge_num_choices,
            output_judge_trials=args.output_judge_trials,
            output_judge_seed=args.output_judge_seed,
            output_max_new_tokens=args.output_max_new_tokens,
            output_generation_temperature=args.output_generation_temperature,
            output_judge_temperature=args.output_judge_temperature,
            output_judge_max_tokens=args.output_judge_max_tokens,
            output_kl_values=args.output_kl_values,
            output_intervention_method=args.output_intervention_method,
            output_logit_top_k=args.output_logit_top_k,
            output_logit_kl_tolerance=args.output_logit_kl_tolerance,
            output_logit_kl_max_steps=args.output_logit_kl_max_steps,
            output_logit_force_refresh_kl_cache=args.output_logit_force_refresh_kl_cache,
            output_logit_include_special_tokens=args.output_logit_include_special_tokens,
        )
        executed_steps.append("round_0_step_4_experiments_execution")
        track_usage(baseline_execution_result, executed=True)
    else:
        baseline_execution_result = _load_json_or_raise(baseline_execution_path)
        loaded_steps.append("round_0_step_4_experiments_execution")
        track_usage(baseline_execution_result, executed=False)
    (
        last_output_intervention_method,
        last_output_score_name,
        last_output_logit_top_k,
    ) = update_last_output_meta(
        baseline_execution_result,
        default_method=last_output_intervention_method,
        default_score_name=last_output_score_name,
        default_logit_top_k=last_output_logit_top_k,
    )
    round_executions[0] = baseline_execution_result
    last_executed_hypotheses = current_hypotheses
    previous_execution_result = baseline_execution_result
    previous_experiments_result = baseline_experiments_result

    baseline_memory_path = _artifact_json_path(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_index=0,
        kind="memory",
    )
    baseline_memory_md_path = baseline_memory_path.with_suffix(".md")
    _log_stage("build baseline memory...", workflow_start_time)
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=5):
        baseline_memory_result = build_hypothesis_memory(
            initial_hypotheses_result=current_hypotheses,
            experiments_result=baseline_experiments_result,
            execution_result=baseline_execution_result,
            hypothesis_reasons=_extract_reason_map(current_hypotheses),
            round_index=0,
            round_id=baseline_round_id,
        )
        _save_json(baseline_memory_path, baseline_memory_result)
        write_hypothesis_memory_markdown(baseline_memory_md_path, memory=baseline_memory_result)
        executed_steps.append("round_0_step_5_memory")
    else:
        baseline_memory_result = _load_json_or_raise(baseline_memory_path)
        loaded_steps.append("round_0_step_5_memory")
    round_memories[0] = baseline_memory_result
    previous_memory_result = baseline_memory_result

    for round_index in range(1, args.max_rounds + 1):
        round_id = _round_id_from_index(round_index)
        round_design_step_index = 3 if merge_enabled else 2
        round_execution_step_index = 4 if merge_enabled else 3
        round_memory_step_index = 5 if merge_enabled else 4
        _log_stage(f"round {round_index}...", workflow_start_time)

        if previous_execution_result is None:
            prev_execution_path = _artifact_json_path(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=ts,
                round_index=round_index - 1,
                kind="experiments_execution",
            )
            previous_execution_result = _load_json_or_raise(prev_execution_path)
        if previous_memory_result is None:
            prev_memory_path = _artifact_json_path(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=ts,
                round_index=round_index - 1,
                kind="memory",
            )
            previous_memory_result = _load_json_or_raise(prev_memory_path)
            round_memories[round_index - 1] = previous_memory_result
        if previous_experiments_result is None:
            prev_experiments_path = _artifact_json_path(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=ts,
                round_index=round_index - 1,
                kind="experiments",
            )
            previous_experiments_result = _load_json_or_raise(prev_experiments_path)

        refine_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="refined_hypotheses",
        )
        _log_stage("refine hypotheses...", workflow_start_time)
        if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=1):
            historical_memories: List[Dict[str, Any]] = []
            history_end = round_index - 1
            history_start = 0
            if args.history_rounds is not None:
                history_start = max(0, history_end - args.history_rounds)
            for hist_round in range(history_start, history_end):
                if hist_round not in round_memories:
                    hist_memory_path = _artifact_json_path(
                        layer_id=layer_id,
                        feature_id=feature_id,
                        timestamp=ts,
                        round_index=hist_round,
                        kind="memory",
                    )
                    round_memories[hist_round] = _load_json_or_raise(hist_memory_path)
                historical_memories.append(round_memories[hist_round])

            top_m = args.top_m
            if top_m is None:
                top_m = len(list(current_hypotheses.get("input_side_hypotheses", [])))
            refinement_result = refine_hypotheses(
                current_memory=previous_memory_result,
                current_execution_result=previous_execution_result,
                historical_memories=historical_memories,
                run_side=args.side,
                model_id=str(current_hypotheses.get("model_id", args.model_id)),
                layer_id=layer_id,
                feature_id=feature_id,
                top_m=top_m,
                history_scope=args.history_scope,
                timestamp=ts,
                round_id=round_id,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_api_key_file=args.llm_api_key_file,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            executed_steps.append(f"{round_id}_step_1_refinement")
            track_usage(refinement_result, executed=True)
        else:
            refinement_result = _load_json_or_raise(refine_path)
            loaded_steps.append(f"{round_id}_step_1_refinement")
            track_usage(refinement_result, executed=False)
        round_refinements[round_index] = refinement_result

        merged_result: Optional[Dict[str, Any]] = None
        if merge_enabled:
            merge_path = _artifact_json_path(
                layer_id=layer_id,
                feature_id=feature_id,
                timestamp=ts,
                round_index=round_index,
                kind="merged_hypotheses",
            )
            _log_stage("merge refined hypotheses...", workflow_start_time)
            if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=2):
                merged_result = merge_refined_hypotheses(
                    refined_hypotheses_result=refinement_result,
                    model_id=str(refinement_result.get("model_id", args.model_id)),
                    layer_id=layer_id,
                    feature_id=feature_id,
                    run_side=args.side,
                    timestamp=ts,
                    round_id=round_id,
                    llm_base_url=args.llm_base_url,
                    llm_model=args.llm_model,
                    llm_api_key_file=args.llm_api_key_file,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                executed_steps.append(f"{round_id}_step_2_merge")
                track_usage(merged_result, executed=True)
            else:
                merged_result = _load_json_or_raise(merge_path)
                loaded_steps.append(f"{round_id}_step_2_merge")
                track_usage(merged_result, executed=False)
            round_merges[round_index] = merged_result

        hypotheses_after_refine = merged_result if merged_result is not None else refinement_result
        hypotheses_before_refine = current_hypotheses
        converged_this_round = _same_hypotheses(current_hypotheses, hypotheses_after_refine)
        current_hypotheses = _to_next_round_hypotheses(
            refinement_result=hypotheses_after_refine,
            round_index=round_index,
        )
        last_executed_hypotheses = current_hypotheses

        frozen_input_indices: Set[int] = set()
        if args.side in ("input", "both") and isinstance(previous_execution_result, dict):
            frozen_candidates = _frozen_input_indices_from_execution(previous_execution_result)
            before_input = _normalize_hypothesis_list(hypotheses_before_refine.get("input_side_hypotheses", []))
            after_input = _normalize_hypothesis_list(current_hypotheses.get("input_side_hypotheses", []))
            for idx in frozen_candidates:
                if 1 <= idx <= len(before_input) and 1 <= idx <= len(after_input):
                    if before_input[idx - 1] == after_input[idx - 1]:
                        frozen_input_indices.add(idx)

        total_input_hypotheses = len(_normalize_hypothesis_list(current_hypotheses.get("input_side_hypotheses", [])))
        active_input_indices = [
            idx for idx in range(1, total_input_hypotheses + 1) if idx not in frozen_input_indices
        ]
        hypotheses_for_design = _build_filtered_hypotheses_for_active_input(
            hypotheses_result=current_hypotheses,
            active_input_indices=active_input_indices,
        )

        experiments_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="experiments",
        )
        _log_stage("design experiments...", workflow_start_time)
        if _should_run(
            start_round=args.start_round,
            start_step=args.start_step,
            round_index=round_index,
            step_index=round_design_step_index,
        ):
            active_experiments_result = design_hypothesis_experiments(
                hypotheses_result=hypotheses_for_design,
                num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
                run_side=args.side,
                round_id=round_id,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_api_key_file=args.llm_api_key_file,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if frozen_input_indices:
                if previous_experiments_result is None:
                    raise ValueError("Missing previous experiments result for frozen input hypotheses.")
                experiments_result = _merge_input_experiments_from_active_and_frozen(
                    current_hypotheses=current_hypotheses,
                    active_indices=active_input_indices,
                    frozen_indices=frozen_input_indices,
                    active_experiments_result=active_experiments_result,
                    previous_experiments_result=previous_experiments_result,
                )
                _save_json(experiments_path, experiments_result)
                write_experiments_markdown(
                    experiments_path.with_suffix(".md"),
                    result=experiments_result,
                    llm_calls=active_experiments_result.get("llm_calls", []),
                )
            else:
                experiments_result = active_experiments_result
            executed_steps.append(f"{round_id}_step_{round_design_step_index}_experiments_design")
            track_usage(active_experiments_result, executed=True)
        else:
            experiments_result = _load_json_or_raise(experiments_path)
            loaded_steps.append(f"{round_id}_step_{round_design_step_index}_experiments_design")
            track_usage(experiments_result, executed=False)

        execution_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="experiments_execution",
        )
        _log_stage("execute experiments...", workflow_start_time)
        if _should_run(
            start_round=args.start_round,
            start_step=args.start_step,
            round_index=round_index,
            step_index=round_execution_step_index,
        ):
            if module is None:
                sae_path = args.sae_path or build_default_sae_path(
                    layer_id=layer_id,
                    width=args.width,
                    release=args.sae_release,
                    average_l0=args.sae_average_l0,
                    canonical_map_path=args.sae_canonical_map,
                )[0]
                module = ModelWithSAEModule(
                    llm_name=args.model_checkpoint_path,
                    sae_path=sae_path,
                    sae_layer=int(layer_id),
                    feature_index=int(feature_id),
                    device=args.device,
                )
            experiments_for_execution = dict(experiments_result)
            if frozen_input_indices:
                active_input_experiments = []
                all_input_experiments = experiments_result.get("input_side_experiments", [])
                if isinstance(all_input_experiments, list):
                    for idx in active_input_indices:
                        if 1 <= idx <= len(all_input_experiments) and isinstance(all_input_experiments[idx - 1], dict):
                            active_input_experiments.append(dict(all_input_experiments[idx - 1]))
                experiments_for_execution["input_side_experiments"] = active_input_experiments

            active_execution_result = execute_hypothesis_experiments(
                experiments_result=experiments_for_execution,
                module=module,
                run_side=args.side,
                round_id=round_id,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_api_key_file=args.llm_api_key_file,
                input_non_zero_threshold=args.input_non_zero_threshold,
                output_judge_num_choices=args.output_judge_num_choices,
                output_judge_trials=args.output_judge_trials,
                output_judge_seed=args.output_judge_seed,
                output_max_new_tokens=args.output_max_new_tokens,
                output_generation_temperature=args.output_generation_temperature,
                output_judge_temperature=args.output_judge_temperature,
                output_judge_max_tokens=args.output_judge_max_tokens,
                output_kl_values=args.output_kl_values,
                output_intervention_method=args.output_intervention_method,
                output_logit_top_k=args.output_logit_top_k,
                output_logit_kl_tolerance=args.output_logit_kl_tolerance,
                output_logit_kl_max_steps=args.output_logit_kl_max_steps,
                output_logit_force_refresh_kl_cache=args.output_logit_force_refresh_kl_cache,
                output_logit_include_special_tokens=args.output_logit_include_special_tokens,
            )
            if frozen_input_indices:
                if previous_execution_result is None:
                    raise ValueError("Missing previous execution result for frozen input hypotheses.")
                execution_result = _merge_input_execution_from_active_and_frozen(
                    current_hypotheses=current_hypotheses,
                    active_indices=active_input_indices,
                    frozen_indices=frozen_input_indices,
                    active_execution_result=active_execution_result,
                    previous_execution_result=previous_execution_result,
                )
                _save_json(execution_path, execution_result)
                write_execution_markdown(
                    execution_path.with_suffix(".md"),
                    result=execution_result,
                    llm_calls=execution_result.get("llm_calls", []),
                )
            else:
                execution_result = active_execution_result
            executed_steps.append(f"{round_id}_step_{round_execution_step_index}_experiments_execution")
            track_usage(active_execution_result, executed=True)
        else:
            execution_result = _load_json_or_raise(execution_path)
            loaded_steps.append(f"{round_id}_step_{round_execution_step_index}_experiments_execution")
            track_usage(execution_result, executed=False)
        (
            last_output_intervention_method,
            last_output_score_name,
            last_output_logit_top_k,
        ) = update_last_output_meta(
            execution_result,
            default_method=last_output_intervention_method,
            default_score_name=last_output_score_name,
            default_logit_top_k=last_output_logit_top_k,
        )
        round_executions[round_index] = execution_result

        memory_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="memory",
        )
        memory_md_path = memory_path.with_suffix(".md")
        _log_stage("build memory...", workflow_start_time)
        if _should_run(
            start_round=args.start_round,
            start_step=args.start_step,
            round_index=round_index,
            step_index=round_memory_step_index,
        ):
            memory_result = build_hypothesis_memory(
                initial_hypotheses_result=current_hypotheses,
                experiments_result=experiments_result,
                execution_result=execution_result,
                hypothesis_reasons=_extract_reason_map(current_hypotheses),
                round_index=round_index,
                round_id=round_id,
            )
            _save_json(memory_path, memory_result)
            write_hypothesis_memory_markdown(memory_md_path, memory=memory_result)
            executed_steps.append(f"{round_id}_step_{round_memory_step_index}_memory")
        else:
            memory_result = _load_json_or_raise(memory_path)
            loaded_steps.append(f"{round_id}_step_{round_memory_step_index}_memory")
        round_memories[round_index] = memory_result
        previous_experiments_result = experiments_result
        previous_execution_result = execution_result
        previous_memory_result = memory_result
        last_round_executed = round_index

        if converged_this_round:
            converged = True
            converged_round = round_index
            break

    if last_round_executed > 0:
        final_hypotheses_source = last_executed_hypotheses
    else:
        final_hypotheses_source = initial_result
    final_input_hypotheses = list(final_hypotheses_source.get("input_side_hypotheses", []))
    final_output_hypotheses = list(final_hypotheses_source.get("output_side_hypotheses", []))
    final_input_reasons = list(final_hypotheses_source.get("input_side_hypothesis_reasons", []))
    final_output_reasons = list(final_hypotheses_source.get("output_side_hypothesis_reasons", []))

    ts_dir = Path("logs") / layer_id / feature_id / ts
    ts_dir.mkdir(parents=True, exist_ok=True)

    input_hypothesis_cache_path = (
        ts_dir / f"layer{layer_id}-feature{feature_id}-input-side-hypotheses-cache.json"
    )
    input_hypothesis_cache = _build_input_hypothesis_selection_payload(
        model_id=_clean_text(final_hypotheses_source.get("model_id") or args.model_id),
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_executions=round_executions,
    )
    _save_json(input_hypothesis_cache_path, input_hypothesis_cache)
    best_input_hypothesis = input_hypothesis_cache.get("best_hypothesis")
    if not isinstance(best_input_hypothesis, dict):
        fallback_hypothesis = final_input_hypotheses[0] if final_input_hypotheses else ""
        best_input_hypothesis = {
            "source": "final_input_hypotheses_fallback",
            "round_index": last_round_executed if last_round_executed > 0 else 0,
            "round_id": _round_id_from_index(last_round_executed if last_round_executed > 0 else 0),
            "hypothesis_index": 1 if fallback_hypothesis else 0,
            "hypothesis": fallback_hypothesis,
            "score_non_zero_rate": None,
            "score_boundary_non_activation_rate": None,
            "combined_score": None,
        }

    workflow_memory_md = ts_dir / f"layer{layer_id}-feature{feature_id}-workflow-memory.md"
    memory_lines: List[str] = []
    memory_lines.append("# SAE Workflow Memory (All Rounds)")
    memory_lines.append("")
    memory_lines.append("## Metadata")
    memory_lines.append(f"- model_id: {args.model_id}")
    memory_lines.append(f"- layer_id: {layer_id}")
    memory_lines.append(f"- feature_id: {feature_id}")
    memory_lines.append(f"- timestamp: {ts}")
    memory_lines.append(f"- max_rounds: {args.max_rounds}")
    memory_lines.append("")
    memory_lines.append("## Round Memories")
    if round_memories:
        for round_index in sorted(round_memories.keys()):
            round_id = _round_id_from_index(round_index)
            memory_lines.append(f"### {round_id}")
            memory_lines.append("```json")
            memory_lines.append(json.dumps(round_memories[round_index], ensure_ascii=False, indent=2))
            memory_lines.append("```")
            memory_lines.append("")
    else:
        memory_lines.append("- no iterative memory generated (max_rounds=0)")
    workflow_memory_md.write_text("\n".join(memory_lines) + "\n", encoding="utf-8")

    final_dir = ts_dir / "final_result"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_result: Dict[str, Any] = {
        "model_id": _clean_text(final_hypotheses_source.get("model_id") or args.model_id),
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "max_rounds": args.max_rounds,
        "executed_rounds": last_round_executed,
        "converged": converged,
        "converged_round": converged_round,
        "input_side_final_hypotheses": final_input_hypotheses,
        "input_side_final_reasons": final_input_reasons,
        "output_side_final_hypotheses": final_output_hypotheses,
        "output_side_final_reasons": final_output_reasons,
        "run_side": args.side,
        "history_rounds": args.history_rounds,
        "enable_hypothesis_merge": merge_enabled,
        "hypothesis_merge_mode": "llm_semantic" if merge_enabled else "off",
        "merged_rounds": sorted(round_merges.keys()),
        "output_intervention_method": last_output_intervention_method,
        "output_score_name": last_output_score_name,
        "output_logit_top_k": last_output_logit_top_k,
        "input_side_hypothesis_cache_path": str(input_hypothesis_cache_path),
        "input_side_hypothesis_scoring_formula": input_hypothesis_cache.get("score_formula"),
        "input_side_best_hypothesis": best_input_hypothesis,
        "token_usage_total": total_tokens.as_dict(),
        "token_usage_this_run": run_tokens.as_dict(),
        "executed_steps": executed_steps,
        "loaded_steps": loaded_steps,
    }

    final_json_path = final_dir / f"layer{layer_id}-feature{feature_id}-final-result.json"
    _save_json(final_json_path, final_result)
    final_md_path = final_dir / f"layer{layer_id}-feature{feature_id}-final-result.md"

    lines: List[str] = []
    lines.append("# SAE Workflow Final Result")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- model_id: {final_result['model_id']}")
    lines.append(f"- layer_id: {layer_id}")
    lines.append(f"- feature_id: {feature_id}")
    lines.append(f"- timestamp: {ts}")
    lines.append(f"- max_rounds: {args.max_rounds}")
    lines.append(f"- executed_rounds: {last_round_executed}")
    lines.append(f"- converged: {converged}")
    lines.append(f"- converged_round: {converged_round}")
    lines.append(f"- enable_hypothesis_merge: {final_result.get('enable_hypothesis_merge')}")
    lines.append(f"- hypothesis_merge_mode: {final_result.get('hypothesis_merge_mode')}")
    lines.append(f"- merged_rounds: {final_result.get('merged_rounds')}")
    lines.append(f"- output_intervention_method: {final_result.get('output_intervention_method')}")
    lines.append(f"- output_score_name: {final_result.get('output_score_name')}")
    lines.append(f"- output_logit_top_k: {final_result.get('output_logit_top_k')}")
    lines.append(f"- input_side_hypothesis_cache_path: {final_result.get('input_side_hypothesis_cache_path')}")
    lines.append("")
    lines.append("## Token Usage (Workflow)")
    lines.append(f"- total_prompt_tokens: {final_result['token_usage_total']['prompt_tokens']}")
    lines.append(f"- total_completion_tokens: {final_result['token_usage_total']['completion_tokens']}")
    lines.append(f"- total_tokens: {final_result['token_usage_total']['total_tokens']}")
    lines.append(f"- this_run_prompt_tokens: {final_result['token_usage_this_run']['prompt_tokens']}")
    lines.append(f"- this_run_completion_tokens: {final_result['token_usage_this_run']['completion_tokens']}")
    lines.append(f"- this_run_total_tokens: {final_result['token_usage_this_run']['total_tokens']}")
    lines.append("")
    lines.append("## Input-side Final Hypotheses")
    for idx, hypothesis in enumerate(final_input_hypotheses, start=1):
        lines.append(f"{idx}. {hypothesis}")
    lines.append("")
    lines.append("## Best Input Hypothesis For Final Evaluation")
    lines.append(f"- round_id: {best_input_hypothesis.get('round_id')}")
    lines.append(f"- hypothesis_index: {best_input_hypothesis.get('hypothesis_index')}")
    lines.append(f"- score_non_zero_rate: {best_input_hypothesis.get('score_non_zero_rate')}")
    lines.append(
        f"- score_boundary_non_activation_rate: {best_input_hypothesis.get('score_boundary_non_activation_rate')}"
    )
    lines.append(f"- combined_score: {best_input_hypothesis.get('combined_score')}")
    lines.append("```text")
    lines.append(str(best_input_hypothesis.get("hypothesis", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Output-side Final Hypotheses")
    for idx, hypothesis in enumerate(final_output_hypotheses, start=1):
        lines.append(f"{idx}. {hypothesis}")
    lines.append("")
    lines.append("## Workflow Steps")
    lines.append("- executed:")
    for item in executed_steps:
        lines.append(f"  - {item}")
    lines.append("- loaded:")
    for item in loaded_steps:
        lines.append(f"  - {item}")
    lines.append("")
    final_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "final_result_json_path": str(final_json_path),
                "final_result_md_path": str(final_md_path),
                "workflow_memory_md_path": str(workflow_memory_md),
                "converged": converged,
                "converged_round": converged_round,
                "enable_hypothesis_merge": final_result.get("enable_hypothesis_merge"),
                "hypothesis_merge_mode": final_result.get("hypothesis_merge_mode"),
                "merged_rounds": final_result.get("merged_rounds"),
                "output_intervention_method": final_result.get("output_intervention_method"),
                "output_score_name": final_result.get("output_score_name"),
                "input_side_hypothesis_cache_path": final_result.get("input_side_hypothesis_cache_path"),
                "input_side_best_hypothesis": best_input_hypothesis,
                "input_final_hypothesis_count": len(final_input_hypotheses),
                "output_final_hypothesis_count": len(final_output_hypotheses),
                "token_usage_total": total_tokens.as_dict(),
                "token_usage_this_run": run_tokens.as_dict(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
