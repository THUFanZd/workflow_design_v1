from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict

from function import (
    TokenUsageAccumulator,
    build_default_sae_path,
    build_feature_dir,
    build_round_dir,
    extract_usage_counts,
    normalize_round_id,
)
from support_info.llm_api_info import (
    api_key_file as DEFAULT_API_KEY_FILE,
    base_url as DEFAULT_BASE_URL,
    model_name as DEFAULT_MODEL_NAME,
)

if TYPE_CHECKING:
    from model_with_sae import ModelWithSAEModule

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


RunSideType = Literal["input", "output", "both"]


@dataclass
class WorkflowConfig:
    args: Any
    layer_id: str
    feature_id: str
    timestamp: str
    merge_enabled: bool
    generation_model: str
    judge_model: str
    activation_max_rounds: int
    expansion_max_rounds: int
    scheduled_input_total_rounds: int
    effective_max_rounds: int


class WorkflowState(TypedDict, total=False):
    config: WorkflowConfig
    workflow_start_time: float
    total_tokens: Dict[str, int]
    run_tokens: Dict[str, int]
    executed_steps: List[str]
    loaded_steps: List[str]
    observation: Dict[str, Any]
    current_hypotheses: Dict[str, Any]
    last_executed_hypotheses: Dict[str, Any]
    previous_round_hypotheses: Optional[Dict[str, Any]]
    previous_execution_result: Optional[Dict[str, Any]]
    previous_memory_result: Optional[Dict[str, Any]]
    previous_experiments_result: Optional[Dict[str, Any]]
    round_memories: Dict[int, Dict[str, Any]]
    round_refinements: Dict[int, Dict[str, Any]]
    round_merges: Dict[int, Dict[str, Any]]
    round_executions: Dict[int, Dict[str, Any]]
    module: Optional[Any]
    converged: bool
    converged_round: Optional[int]
    last_round_executed: int
    last_output_intervention_method: str
    last_output_score_name: str
    last_output_logit_top_k: int
    current_input_test_mode: str
    input_activation_rounds_done: int
    input_expansion_rounds_done: int
    current_round: int
    refinement_result: Optional[Dict[str, Any]]
    merged_result: Optional[Dict[str, Any]]
    experiments_result: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    memory_result: Optional[Dict[str, Any]]
    final_result: Optional[Dict[str, Any]]


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _round_id_from_index(round_index: int) -> str:
    return normalize_round_id(None, round_index=round_index)


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
    return build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=timestamp,
        round_id=_round_id_from_index(round_index),
        round_index=round_index,
    ) / filename_map[kind]


def _load_json_or_raise(path: Path) -> Dict[str, Any]:
    target_path = path
    if not target_path.exists():
        parts = list(path.parts)
        if len(parts) >= 5 and parts[0] == "logs":
            layer_part = parts[1]
            feature_part = parts[2]
            if layer_part.startswith("layer-") and feature_part.startswith("feature-"):
                layer_id = layer_part[len("layer-") :]
                feature_id = feature_part[len("feature-") :]
                old_path = Path("logs") / layer_id / feature_id / Path(*parts[3:])
                if old_path.exists():
                    target_path = old_path
                else:
                    legacy_path = Path("logs") / f"{layer_id}_{feature_id}" / Path(*parts[3:])
                    if legacy_path.exists():
                        target_path = legacy_path
    if not target_path.exists():
        raise FileNotFoundError(f"Cannot find required file: {path}")

    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a dict: {target_path}")
    return payload


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _result_usage(result: Dict[str, Any]) -> Dict[str, int]:
    usage = result.get("token_usage")
    if isinstance(usage, dict):
        return usage
    return extract_usage_counts(usage)


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


def _normalize_hypothesis_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value]


def _format_elapsed(total_seconds: float) -> str:
    total = max(0, int(total_seconds))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _extract_reason_map(hypotheses_result: Dict[str, Any]) -> Dict[str, Any]:
    input_reasons = hypotheses_result.get("input_side_hypothesis_reasons", [])
    output_reasons = hypotheses_result.get("output_side_hypothesis_reasons", [])
    return {
        "input": input_reasons if isinstance(input_reasons, list) else [],
        "output": output_reasons if isinstance(output_reasons, list) else [],
    }


def _to_next_round_hypotheses(*, refinement_result: Dict[str, Any], round_index: int) -> Dict[str, Any]:
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
            test_type = _clean_text(item.get("test_type")) or "activation"
            test_type_rank = 1 if test_type == "expansion" else 0
            entries.append(
                {
                    "round_index": round_index,
                    "round_id": round_id,
                    "hypothesis_index": _safe_int(item.get("hypothesis_index"), 0),
                    "hypothesis": _clean_text(item.get("hypothesis")),
                    "test_type": test_type,
                    "score_non_zero_rate": score_non_zero_rate,
                    "combined_score": score_non_zero_rate,
                    "test_type_rank": test_type_rank,
                    "non_zero_count": _safe_int(item.get("non_zero_count"), 0),
                    "total_sentences": _safe_int(item.get("total_sentences"), 0),
                    "designed_sentences": (
                        list(item.get("designed_sentences", []))
                        if isinstance(item.get("designed_sentences"), list)
                        else []
                    ),
                    "reference_hypothesis": _clean_text(item.get("reference_hypothesis")),
                    "sentence_results": (
                        list(item.get("sentence_results", []))
                        if isinstance(item.get("sentence_results"), list)
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
            -_safe_float(item.get("score_non_zero_rate"), 0.0),
            -_safe_int(item.get("test_type_rank"), 0),
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
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "score_formula": "score_non_zero_rate (tie-breaker: prefer expansion rounds, then newer rounds)",
        "total_candidates": len(ranked_entries),
        "best_hypothesis": best_hypothesis,
        "hypotheses": ranked_entries,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SAE explanation workflow via LangGraph orchestration.",
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id")
    parser.add_argument("--layer-id", required=True, help="Layer id")
    parser.add_argument("--feature-id", required=True, help="Feature id")
    parser.add_argument("--timestamp", default=None, help="Timestamp directory under logs/layer-{layer_id}/feature-{feature_id}/")
    parser.add_argument("--input-activation-max-rounds", type=int, default=1, help="Input-side activation test rounds p (includes baseline round_0).")
    parser.add_argument("--input-expansion-max-rounds", type=int, default=1, help="Input-side expansion test rounds q.")
    parser.add_argument("--start-round", type=int, default=0, help="Round index to start real execution from. round_0 is the baseline round.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help=(
            "When --start-round=0, valid steps are 1..5 (observation, initial, design, execution, memory). "
            "When --start-round>=1 and merge is disabled, valid steps are 1..4 (refinement, design, execution, memory). "
            "When --start-round>=1 and merge is enabled, valid steps are 1..5 (refinement, merge, design, execution, memory)."
        ),
    )
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="Reuse artifacts before start point from logs/layer-{layer_id}/feature-{feature_id}/{timestamp}/{round_id}.",
    )
    parser.add_argument("--num-hypothesis", type=int, default=3, help="Hypothesis count n for each side")
    parser.add_argument("--side", choices=["input", "output", "both"], default="both", help="Run only input-side or output-side hypothesis iteration, or both.")
    parser.add_argument("--generation-mode", choices=["single_call", "iterative"], default="single_call", help="Initial hypothesis generation mode.")
    parser.add_argument("--num-input-sentences-per-hypothesis", type=int, default=5, help="Input-side designed activation or expansion sentences per hypothesis.")
    parser.add_argument("--top-m", type=int, default=None, help="Refine top-m hypotheses per side (default=all).")
    parser.add_argument("--enable-hypothesis-merge", action="store_true", help="After each refinement, add an LLM semantic merge step.")
    parser.add_argument("--history-rounds", type=int, default=1, help="Use at most previous n historical memory rounds during refinement.")
    parser.add_argument("--history-scope", choices=["same_hypothesis", "all_hypotheses"], default="same_hypothesis", help="Historical memory scope used in refinement.")
    parser.add_argument("--width", default="16k", help="Neuronpedia source width")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-generation-model", "--llm-model", dest="llm_generation_model", default=DEFAULT_MODEL_NAME, help="LLM model used for hypothesis generation/refinement/experiment design.")
    parser.add_argument("--llm-judge-model", default=None, help="LLM model used for judge-style calls (output-side intervention judge). Defaults to generation model.")
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--sae-path", default=None, help="SAE path or sae-lens URI")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument("--sae-canonical-map", default="support_info/canonical_map.txt", help="Path to canonical_map.txt for default SAE resolution.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-non-zero-threshold", type=float, default=0.0)
    parser.add_argument("--output-max-new-tokens", type=int, default=25)
    parser.add_argument("--output-generation-temperature", type=float, default=0.75)
    parser.add_argument("--output-judge-num-choices", type=int, default=3)
    parser.add_argument("--output-judge-trials", type=int, default=1)
    parser.add_argument("--output-judge-seed", type=int, default=42)
    parser.add_argument("--output-judge-temperature", type=float, default=0.0)
    parser.add_argument("--output-judge-max-tokens", type=int, default=20000)
    parser.add_argument("--output-kl-values", type=float, nargs="*", default=[0.25, 0.5, -0.25, -0.5])
    parser.add_argument("--output-intervention-method", choices=["blind", "logit"], default="blind", help="Output-side intervention scoring method used in experiments execution.")
    parser.add_argument("--output-logit-top-k", type=int, default=5)
    parser.add_argument("--output-logit-kl-tolerance", type=float, default=0.1)
    parser.add_argument("--output-logit-kl-max-steps", type=int, default=12)
    parser.add_argument("--output-logit-force-refresh-kl-cache", action="store_true")
    parser.add_argument("--output-logit-include-special-tokens", action="store_true")
    return parser


def _usage_zero() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(base: Dict[str, int], extra: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": _safe_int(base.get("prompt_tokens"), 0) + _safe_int(extra.get("prompt_tokens"), 0),
        "completion_tokens": _safe_int(base.get("completion_tokens"), 0) + _safe_int(extra.get("completion_tokens"), 0),
        "total_tokens": _safe_int(base.get("total_tokens"), 0) + _safe_int(extra.get("total_tokens"), 0),
    }


def _track_usage(state: WorkflowState, result: Dict[str, Any], *, executed: bool) -> Dict[str, Any]:
    usage = _result_usage(result)
    updates: Dict[str, Any] = {
        "total_tokens": _merge_usage(state.get("total_tokens", _usage_zero()), usage),
    }
    if executed:
        updates["run_tokens"] = _merge_usage(state.get("run_tokens", _usage_zero()), usage)
    return updates


def _log_stage(state: WorkflowState, message: str) -> None:
    elapsed = _format_elapsed(time.perf_counter() - state["workflow_start_time"])
    print(f"[elapsed {elapsed}] {message}")


def _update_last_output_meta(
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


def _ensure_module(state: WorkflowState) -> Any:
    from model_with_sae import ModelWithSAEModule

    module = state.get("module")
    if module is not None:
        return module
    config = state["config"]
    args = config.args
    sae_path = args.sae_path or build_default_sae_path(
        layer_id=config.layer_id,
        width=args.width,
        release=args.sae_release,
        average_l0=args.sae_average_l0,
        canonical_map_path=args.sae_canonical_map,
    )[0]
    return ModelWithSAEModule(
        llm_name=args.model_checkpoint_path,
        sae_path=sae_path,
        sae_layer=int(config.layer_id),
        feature_index=int(config.feature_id),
        device=args.device,
    )


def _update_input_mode_after_execution(
    state: WorkflowState,
    *,
    experiments_result: Dict[str, Any],
    execution_result: Dict[str, Any],
    current_input_test_mode: str,
    input_activation_rounds_done: int,
    input_expansion_rounds_done: int,
) -> Dict[str, Any]:
    args = state["config"].args
    if args.side not in ("input", "both"):
        return {
            "current_input_test_mode": current_input_test_mode,
            "input_activation_rounds_done": input_activation_rounds_done,
            "input_expansion_rounds_done": input_expansion_rounds_done,
        }

    used_mode = _clean_text(
        experiments_result.get("input_test_mode")
        or execution_result.get("input_side_execution", {}).get("input_test_mode")
        or current_input_test_mode
    ) or current_input_test_mode
    full_activation = (
        _safe_float(
            execution_result.get("input_side_execution", {}).get("overall_score_non_zero_rate"),
            0.0,
        )
        >= 1.0
    )

    if used_mode == "expansion":
        input_expansion_rounds_done += 1
        current_input_test_mode = "expansion"
    else:
        input_activation_rounds_done += 1
        if full_activation or input_activation_rounds_done >= state["config"].activation_max_rounds:
            current_input_test_mode = "expansion"
        else:
            current_input_test_mode = "activation"

    return {
        "current_input_test_mode": current_input_test_mode,
        "input_activation_rounds_done": input_activation_rounds_done,
        "input_expansion_rounds_done": input_expansion_rounds_done,
    }


def _should_stop_before_round(state: WorkflowState) -> bool:
    config = state["config"]
    round_index = state.get("current_round", 1)
    if round_index > config.effective_max_rounds:
        return True
    if config.args.side in ("input", "both"):
        return (
            state.get("current_input_test_mode", "activation") == "expansion"
            and state.get("input_expansion_rounds_done", 0) >= config.expansion_max_rounds
        )
    return False


def _ensure_previous_round_context(state: WorkflowState, round_index: int) -> Dict[str, Any]:
    config = state["config"]
    previous_execution_result = state.get("previous_execution_result")
    previous_memory_result = state.get("previous_memory_result")
    previous_experiments_result = state.get("previous_experiments_result")
    round_memories = dict(state.get("round_memories", {}))

    if previous_execution_result is None:
        previous_execution_result = _load_json_or_raise(
            _artifact_json_path(
                layer_id=config.layer_id,
                feature_id=config.feature_id,
                timestamp=config.timestamp,
                round_index=round_index - 1,
                kind="experiments_execution",
            )
        )
    if previous_memory_result is None:
        previous_memory_result = _load_json_or_raise(
            _artifact_json_path(
                layer_id=config.layer_id,
                feature_id=config.feature_id,
                timestamp=config.timestamp,
                round_index=round_index - 1,
                kind="memory",
            )
        )
        round_memories[round_index - 1] = previous_memory_result
    if previous_experiments_result is None:
        previous_experiments_result = _load_json_or_raise(
            _artifact_json_path(
                layer_id=config.layer_id,
                feature_id=config.feature_id,
                timestamp=config.timestamp,
                round_index=round_index - 1,
                kind="experiments",
            )
        )

    return {
        "previous_execution_result": previous_execution_result,
        "previous_memory_result": previous_memory_result,
        "previous_experiments_result": previous_experiments_result,
        "round_memories": round_memories,
    }


def collect_observation_node(state: WorkflowState) -> Dict[str, Any]:
    from neuronpedia_feature_api import fetch_and_parse_feature_observation

    config = state["config"]
    args = config.args
    _log_stage(state, "collect observation...")
    observation_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=0,
        kind="observation_input",
    )
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=1):
        observation = fetch_and_parse_feature_observation(
            model_id=args.model_id,
            layer_id=config.layer_id,
            feature_id=config.feature_id,
            width=args.width,
            selection_method=args.selection_method,
            m=args.observation_m,
            n=args.observation_n,
            api_key=args.neuronpedia_api_key,
            timeout=args.neuronpedia_timeout,
            timestamp=config.timestamp,
            round_id=_round_id_from_index(0),
        )
        return {
            "observation": observation,
            "executed_steps": [*state.get("executed_steps", []), "round_0_step_1_observation"],
        }

    observation = _load_json_or_raise(observation_path)
    return {
        "observation": observation,
        "loaded_steps": [*state.get("loaded_steps", []), "round_0_step_1_observation"],
    }


def generate_initial_hypotheses_node(state: WorkflowState) -> Dict[str, Any]:
    from initial_hypothesis_generation import generate_initial_hypotheses

    config = state["config"]
    args = config.args
    _log_stage(state, "generate initial hypotheses...")
    initial_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=0,
        kind="initial_hypotheses",
    )
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=2):
        initial_result = generate_initial_hypotheses(
            observation=state["observation"],
            model_id=args.model_id,
            layer_id=config.layer_id,
            feature_id=config.feature_id,
            num_hypothesis=args.num_hypothesis,
            generation_mode=args.generation_mode,
            run_side=args.side,
            timestamp=config.timestamp,
            round_id=_round_id_from_index(0),
            llm_base_url=args.llm_base_url,
            llm_model=config.generation_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        updates = _track_usage(state, initial_result, executed=True)
        updates.update(
            {
                "current_hypotheses": initial_result,
                "last_executed_hypotheses": initial_result,
                "executed_steps": [*state.get("executed_steps", []), "round_0_step_2_initial_hypotheses"],
            }
        )
        return updates

    initial_result = _load_json_or_raise(initial_path)
    updates = _track_usage(state, initial_result, executed=False)
    updates.update(
        {
            "current_hypotheses": initial_result,
            "last_executed_hypotheses": initial_result,
            "loaded_steps": [*state.get("loaded_steps", []), "round_0_step_2_initial_hypotheses"],
        }
    )
    return updates


def design_baseline_experiments_node(state: WorkflowState) -> Dict[str, Any]:
    from experiments_design import design_hypothesis_experiments

    config = state["config"]
    args = config.args
    _log_stage(state, "design baseline experiments...")
    baseline_experiments_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=0,
        kind="experiments",
    )
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=3):
        baseline_experiments_result = design_hypothesis_experiments(
            hypotheses_result=state["current_hypotheses"],
            num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
            run_side=args.side,
            input_test_mode=state.get("current_input_test_mode", "activation"),
            round_id=_round_id_from_index(0),
            llm_base_url=args.llm_base_url,
            llm_model=config.generation_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        updates = _track_usage(state, baseline_experiments_result, executed=True)
        updates.update(
            {
                "experiments_result": baseline_experiments_result,
                "executed_steps": [*state.get("executed_steps", []), "round_0_step_3_experiments_design"],
            }
        )
        return updates

    baseline_experiments_result = _load_json_or_raise(baseline_experiments_path)
    updates = _track_usage(state, baseline_experiments_result, executed=False)
    updates.update(
        {
            "experiments_result": baseline_experiments_result,
            "loaded_steps": [*state.get("loaded_steps", []), "round_0_step_3_experiments_design"],
        }
    )
    return updates


def execute_baseline_experiments_node(state: WorkflowState) -> Dict[str, Any]:
    from experiments_execution import execute_hypothesis_experiments

    config = state["config"]
    args = config.args
    _log_stage(state, "execute baseline experiments...")
    baseline_execution_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=0,
        kind="experiments_execution",
    )
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=4):
        module = _ensure_module(state)
        baseline_execution_result = execute_hypothesis_experiments(
            experiments_result=state["experiments_result"],
            module=module,
            run_side=args.side,
            round_id=_round_id_from_index(0),
            llm_base_url=args.llm_base_url,
            output_judge_llm_model=config.judge_model,
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
        updates = _track_usage(state, baseline_execution_result, executed=True)
        updates.update(
            {
                "module": module,
                "execution_result": baseline_execution_result,
                "executed_steps": [*state.get("executed_steps", []), "round_0_step_4_experiments_execution"],
            }
        )
    else:
        baseline_execution_result = _load_json_or_raise(baseline_execution_path)
        updates = _track_usage(state, baseline_execution_result, executed=False)
        updates.update(
            {
                "execution_result": baseline_execution_result,
                "loaded_steps": [*state.get("loaded_steps", []), "round_0_step_4_experiments_execution"],
            }
        )

    method, score_name, logit_top_k = _update_last_output_meta(
        baseline_execution_result,
        default_method=state.get("last_output_intervention_method", str(args.output_intervention_method)),
        default_score_name=state.get("last_output_score_name", "score_blind_accuracy"),
        default_logit_top_k=state.get("last_output_logit_top_k", args.output_logit_top_k),
    )
    round_executions = dict(state.get("round_executions", {}))
    round_executions[0] = baseline_execution_result
    updates.update(
        {
            "round_executions": round_executions,
            "last_executed_hypotheses": state["current_hypotheses"],
            "previous_execution_result": baseline_execution_result,
            "previous_experiments_result": state["experiments_result"],
            "last_output_intervention_method": method,
            "last_output_score_name": score_name,
            "last_output_logit_top_k": logit_top_k,
        }
    )
    updates.update(
        _update_input_mode_after_execution(
            state,
            experiments_result=state["experiments_result"],
            execution_result=baseline_execution_result,
            current_input_test_mode=state.get("current_input_test_mode", "activation"),
            input_activation_rounds_done=state.get("input_activation_rounds_done", 0),
            input_expansion_rounds_done=state.get("input_expansion_rounds_done", 0),
        )
    )
    return updates


def build_baseline_memory_node(state: WorkflowState) -> Dict[str, Any]:
    from hypothesis_memory import build_hypothesis_memory, write_hypothesis_memory_markdown

    config = state["config"]
    args = config.args
    _log_stage(state, "build baseline memory...")
    baseline_memory_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=0,
        kind="memory",
    )
    baseline_memory_md_path = baseline_memory_path.with_suffix(".md")
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=0, step_index=5):
        baseline_memory_result = build_hypothesis_memory(
            initial_hypotheses_result=state["current_hypotheses"],
            experiments_result=state["experiments_result"],
            execution_result=state["execution_result"],
            hypothesis_reasons=_extract_reason_map(state["current_hypotheses"]),
            round_index=0,
            round_id=_round_id_from_index(0),
        )
        _save_json(baseline_memory_path, baseline_memory_result)
        write_hypothesis_memory_markdown(baseline_memory_md_path, memory=baseline_memory_result)
        updates: Dict[str, Any] = {
            "executed_steps": [*state.get("executed_steps", []), "round_0_step_5_memory"],
        }
    else:
        baseline_memory_result = _load_json_or_raise(baseline_memory_path)
        updates = {
            "loaded_steps": [*state.get("loaded_steps", []), "round_0_step_5_memory"],
        }

    round_memories = dict(state.get("round_memories", {}))
    round_memories[0] = baseline_memory_result
    updates.update(
        {
            "memory_result": baseline_memory_result,
            "round_memories": round_memories,
            "previous_memory_result": baseline_memory_result,
            "current_round": 1,
        }
    )
    return updates


def refine_hypotheses_node(state: WorkflowState) -> Dict[str, Any]:
    from hypothesis_refinement import refine_hypotheses

    config = state["config"]
    args = config.args
    round_index = state["current_round"]
    _log_stage(state, "refine hypotheses...")
    context_updates = _ensure_previous_round_context(state, round_index)
    previous_memory_result = context_updates["previous_memory_result"]
    previous_execution_result = context_updates["previous_execution_result"]
    round_memories = context_updates["round_memories"]

    refine_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=round_index,
        kind="refined_hypotheses",
    )
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=1):
        historical_memories: List[Dict[str, Any]] = []
        history_end = round_index - 1
        history_start = 0
        if args.history_rounds is not None:
            history_start = max(0, history_end - args.history_rounds)
        for hist_round in range(history_start, history_end):
            if hist_round not in round_memories:
                round_memories[hist_round] = _load_json_or_raise(
                    _artifact_json_path(
                        layer_id=config.layer_id,
                        feature_id=config.feature_id,
                        timestamp=config.timestamp,
                        round_index=hist_round,
                        kind="memory",
                    )
                )
            historical_memories.append(round_memories[hist_round])

        top_m = args.top_m
        if top_m is None:
            top_m = len(list(state["current_hypotheses"].get("input_side_hypotheses", [])))
        refinement_result = refine_hypotheses(
            current_memory=previous_memory_result,
            current_execution_result=previous_execution_result,
            historical_memories=historical_memories,
            run_side=args.side,
            model_id=str(state["current_hypotheses"].get("model_id", args.model_id)),
            layer_id=config.layer_id,
            feature_id=config.feature_id,
            top_m=top_m,
            history_scope=args.history_scope,
            timestamp=config.timestamp,
            round_id=_round_id_from_index(round_index),
            llm_base_url=args.llm_base_url,
            llm_model=config.generation_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        updates = _track_usage(state, refinement_result, executed=True)
        updates.update(
            {
                "executed_steps": [*state.get("executed_steps", []), f"round_{round_index}_step_1_refinement"],
            }
        )
    else:
        refinement_result = _load_json_or_raise(refine_path)
        updates = _track_usage(state, refinement_result, executed=False)
        updates.update(
            {
                "loaded_steps": [*state.get("loaded_steps", []), f"round_{round_index}_step_1_refinement"],
            }
        )

    round_refinements = dict(state.get("round_refinements", {}))
    round_refinements[round_index] = refinement_result
    updates.update(
        {
            **context_updates,
            "refinement_result": refinement_result,
            "merged_result": None,
            "round_refinements": round_refinements,
        }
    )
    return updates


def merge_hypotheses_node(state: WorkflowState) -> Dict[str, Any]:
    from hypothesis_merge import merge_refined_hypotheses

    config = state["config"]
    args = config.args
    round_index = state["current_round"]
    if not config.merge_enabled:
        return {"merged_result": None}

    _log_stage(state, "merge refined hypotheses...")
    merge_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=round_index,
        kind="merged_hypotheses",
    )
    if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=2):
        merged_result = merge_refined_hypotheses(
            refined_hypotheses_result=state["refinement_result"],
            model_id=str(state["refinement_result"].get("model_id", args.model_id)),
            layer_id=config.layer_id,
            feature_id=config.feature_id,
            run_side=args.side,
            timestamp=config.timestamp,
            round_id=_round_id_from_index(round_index),
            llm_base_url=args.llm_base_url,
            llm_model=config.generation_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        updates = _track_usage(state, merged_result, executed=True)
        updates.update(
            {
                "executed_steps": [*state.get("executed_steps", []), f"round_{round_index}_step_2_merge"],
            }
        )
    else:
        merged_result = _load_json_or_raise(merge_path)
        updates = _track_usage(state, merged_result, executed=False)
        updates.update(
            {
                "loaded_steps": [*state.get("loaded_steps", []), f"round_{round_index}_step_2_merge"],
            }
        )

    round_merges = dict(state.get("round_merges", {}))
    round_merges[round_index] = merged_result
    updates.update(
        {
            "merged_result": merged_result,
            "round_merges": round_merges,
        }
    )
    return updates


def prepare_round_hypotheses_node(state: WorkflowState) -> Dict[str, Any]:
    round_index = state["current_round"]
    hypotheses_after_refine = state["merged_result"] if state.get("merged_result") is not None else state["refinement_result"]
    hypotheses_before_refine = state["current_hypotheses"]
    converged_this_round = _same_hypotheses(hypotheses_before_refine, hypotheses_after_refine)
    current_hypotheses = _to_next_round_hypotheses(
        refinement_result=hypotheses_after_refine,
        round_index=round_index,
    )
    return {
        "current_hypotheses": current_hypotheses,
        "last_executed_hypotheses": current_hypotheses,
        "previous_round_hypotheses": hypotheses_before_refine,
        "converged": bool(state.get("converged", False) or (
            state["config"].args.side not in ("input", "both") and converged_this_round
        )),
        "converged_round": round_index if state["config"].args.side not in ("input", "both") and converged_this_round else state.get("converged_round"),
    }


def design_experiments_node(state: WorkflowState) -> Dict[str, Any]:
    from experiments_design import design_hypothesis_experiments

    config = state["config"]
    args = config.args
    round_index = state["current_round"]
    round_design_step_index = 3 if config.merge_enabled else 2
    _log_stage(state, "design experiments...")
    experiments_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=round_index,
        kind="experiments",
    )
    input_test_mode_for_round = state.get("current_input_test_mode", "activation") if args.side in ("input", "both") else "activation"
    previous_input_hypotheses = _normalize_hypothesis_list(
        state.get("previous_round_hypotheses", {}).get("input_side_hypotheses", [])
    )
    if _should_run(
        start_round=args.start_round,
        start_step=args.start_step,
        round_index=round_index,
        step_index=round_design_step_index,
    ):
        experiments_result = design_hypothesis_experiments(
            hypotheses_result=state["current_hypotheses"],
            num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
            run_side=args.side,
            input_test_mode=input_test_mode_for_round,
            previous_input_hypotheses=previous_input_hypotheses,
            round_id=_round_id_from_index(round_index),
            llm_base_url=args.llm_base_url,
            llm_model=config.generation_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        updates = _track_usage(state, experiments_result, executed=True)
        updates.update(
            {
                "executed_steps": [
                    *state.get("executed_steps", []),
                    f"round_{round_index}_step_{round_design_step_index}_experiments_design",
                ],
            }
        )
    else:
        experiments_result = _load_json_or_raise(experiments_path)
        updates = _track_usage(state, experiments_result, executed=False)
        updates.update(
            {
                "loaded_steps": [
                    *state.get("loaded_steps", []),
                    f"round_{round_index}_step_{round_design_step_index}_experiments_design",
                ],
            }
        )

    updates.update({"experiments_result": experiments_result})
    return updates


def execute_experiments_node(state: WorkflowState) -> Dict[str, Any]:
    from experiments_execution import execute_hypothesis_experiments

    config = state["config"]
    args = config.args
    round_index = state["current_round"]
    round_execution_step_index = 4 if config.merge_enabled else 3
    _log_stage(state, "execute experiments...")
    execution_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=round_index,
        kind="experiments_execution",
    )
    if _should_run(
        start_round=args.start_round,
        start_step=args.start_step,
        round_index=round_index,
        step_index=round_execution_step_index,
    ):
        module = _ensure_module(state)
        execution_result = execute_hypothesis_experiments(
            experiments_result=state["experiments_result"],
            module=module,
            run_side=args.side,
            round_id=_round_id_from_index(round_index),
            llm_base_url=args.llm_base_url,
            output_judge_llm_model=config.judge_model,
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
        updates = _track_usage(state, execution_result, executed=True)
        updates.update(
            {
                "module": module,
                "executed_steps": [
                    *state.get("executed_steps", []),
                    f"round_{round_index}_step_{round_execution_step_index}_experiments_execution",
                ],
            }
        )
    else:
        execution_result = _load_json_or_raise(execution_path)
        updates = _track_usage(state, execution_result, executed=False)
        updates.update(
            {
                "loaded_steps": [
                    *state.get("loaded_steps", []),
                    f"round_{round_index}_step_{round_execution_step_index}_experiments_execution",
                ],
            }
        )

    method, score_name, logit_top_k = _update_last_output_meta(
        execution_result,
        default_method=state.get("last_output_intervention_method", str(args.output_intervention_method)),
        default_score_name=state.get("last_output_score_name", "score_blind_accuracy"),
        default_logit_top_k=state.get("last_output_logit_top_k", args.output_logit_top_k),
    )
    round_executions = dict(state.get("round_executions", {}))
    round_executions[round_index] = execution_result
    updates.update(
        {
            "execution_result": execution_result,
            "round_executions": round_executions,
            "last_output_intervention_method": method,
            "last_output_score_name": score_name,
            "last_output_logit_top_k": logit_top_k,
        }
    )
    return updates


def build_memory_node(state: WorkflowState) -> Dict[str, Any]:
    from hypothesis_memory import build_hypothesis_memory, write_hypothesis_memory_markdown

    config = state["config"]
    args = config.args
    round_index = state["current_round"]
    round_memory_step_index = 5 if config.merge_enabled else 4
    _log_stage(state, "build memory...")
    memory_path = _artifact_json_path(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=config.timestamp,
        round_index=round_index,
        kind="memory",
    )
    memory_md_path = memory_path.with_suffix(".md")
    if _should_run(
        start_round=args.start_round,
        start_step=args.start_step,
        round_index=round_index,
        step_index=round_memory_step_index,
    ):
        memory_result = build_hypothesis_memory(
            initial_hypotheses_result=state["current_hypotheses"],
            experiments_result=state["experiments_result"],
            execution_result=state["execution_result"],
            hypothesis_reasons=_extract_reason_map(state["current_hypotheses"]),
            round_index=round_index,
            round_id=_round_id_from_index(round_index),
        )
        _save_json(memory_path, memory_result)
        write_hypothesis_memory_markdown(memory_md_path, memory=memory_result)
        updates: Dict[str, Any] = {
            "executed_steps": [
                *state.get("executed_steps", []),
                f"round_{round_index}_step_{round_memory_step_index}_memory",
            ],
        }
    else:
        memory_result = _load_json_or_raise(memory_path)
        updates = {
            "loaded_steps": [
                *state.get("loaded_steps", []),
                f"round_{round_index}_step_{round_memory_step_index}_memory",
            ],
        }

    round_memories = dict(state.get("round_memories", {}))
    round_memories[round_index] = memory_result
    updates.update(
        {
            "memory_result": memory_result,
            "round_memories": round_memories,
            "previous_experiments_result": state["experiments_result"],
            "previous_execution_result": state["execution_result"],
            "previous_memory_result": memory_result,
            "last_round_executed": round_index,
        }
    )
    updates.update(
        _update_input_mode_after_execution(
            state,
            experiments_result=state["experiments_result"],
            execution_result=state["execution_result"],
            current_input_test_mode=state.get("current_input_test_mode", "activation"),
            input_activation_rounds_done=state.get("input_activation_rounds_done", 0),
            input_expansion_rounds_done=state.get("input_expansion_rounds_done", 0),
        )
    )
    return updates


def advance_round_node(state: WorkflowState) -> Dict[str, Any]:
    return {"current_round": state["current_round"] + 1}


def finalize_node(state: WorkflowState) -> Dict[str, Any]:
    config = state["config"]
    args = config.args
    layer_id = config.layer_id
    feature_id = config.feature_id
    ts = config.timestamp

    final_hypotheses_source = state.get("last_executed_hypotheses") if state.get("last_round_executed", 0) > 0 else state["current_hypotheses"]
    final_input_hypotheses = list(final_hypotheses_source.get("input_side_hypotheses", []))
    final_output_hypotheses = list(final_hypotheses_source.get("output_side_hypotheses", []))
    final_input_reasons = list(final_hypotheses_source.get("input_side_hypothesis_reasons", []))
    final_output_reasons = list(final_hypotheses_source.get("output_side_hypothesis_reasons", []))

    ts_dir = build_feature_dir(layer_id=layer_id, feature_id=feature_id) / ts
    ts_dir.mkdir(parents=True, exist_ok=True)

    input_hypothesis_cache_path = ts_dir / f"layer{layer_id}-feature{feature_id}-input-side-hypotheses-cache.json"
    input_hypothesis_cache = _build_input_hypothesis_selection_payload(
        model_id=_clean_text(final_hypotheses_source.get("model_id") or args.model_id),
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_executions=state.get("round_executions", {}),
    )
    _save_json(input_hypothesis_cache_path, input_hypothesis_cache)
    best_input_hypothesis = input_hypothesis_cache.get("best_hypothesis")
    if not isinstance(best_input_hypothesis, dict):
        fallback_hypothesis = final_input_hypotheses[0] if final_input_hypotheses else ""
        best_input_hypothesis = {
            "source": "final_input_hypotheses_fallback",
            "round_index": state.get("last_round_executed", 0),
            "round_id": _round_id_from_index(state.get("last_round_executed", 0)),
            "hypothesis_index": 1 if fallback_hypothesis else 0,
            "hypothesis": fallback_hypothesis,
            "test_type": None,
            "score_non_zero_rate": None,
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
    memory_lines.append(f"- scheduled_input_total_rounds_p_plus_q: {config.scheduled_input_total_rounds}")
    memory_lines.append(f"- effective_refinement_rounds: {config.effective_max_rounds}")
    memory_lines.append("")
    memory_lines.append("## Round Memories")
    if state.get("round_memories"):
        for round_index in sorted(state["round_memories"].keys()):
            memory_lines.append(f"### {_round_id_from_index(round_index)}")
            memory_lines.append("```json")
            memory_lines.append(json.dumps(state["round_memories"][round_index], ensure_ascii=False, indent=2))
            memory_lines.append("```")
            memory_lines.append("")
    else:
        memory_lines.append("- no iterative memory generated (effective_max_rounds=0)")
    workflow_memory_md.write_text("\n".join(memory_lines) + "\n", encoding="utf-8")

    final_dir = ts_dir / "final_result"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_result: Dict[str, Any] = {
        "model_id": _clean_text(final_hypotheses_source.get("model_id") or args.model_id),
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "max_rounds": config.effective_max_rounds,
        "input_activation_max_rounds": config.activation_max_rounds,
        "input_expansion_max_rounds": config.expansion_max_rounds,
        "scheduled_input_total_rounds": config.scheduled_input_total_rounds,
        "input_activation_rounds_done": state.get("input_activation_rounds_done", 0),
        "input_expansion_rounds_done": state.get("input_expansion_rounds_done", 0),
        "last_input_test_mode": state.get("current_input_test_mode", "activation"),
        "executed_rounds": state.get("last_round_executed", 0),
        "converged": state.get("converged", False),
        "converged_round": state.get("converged_round"),
        "input_side_final_hypotheses": final_input_hypotheses,
        "input_side_final_reasons": final_input_reasons,
        "output_side_final_hypotheses": final_output_hypotheses,
        "output_side_final_reasons": final_output_reasons,
        "run_side": args.side,
        "history_rounds": args.history_rounds,
        "enable_hypothesis_merge": config.merge_enabled,
        "hypothesis_merge_mode": "llm_semantic" if config.merge_enabled else "off",
        "merged_rounds": sorted(state.get("round_merges", {}).keys()),
        "output_intervention_method": state.get("last_output_intervention_method"),
        "output_score_name": state.get("last_output_score_name"),
        "output_logit_top_k": state.get("last_output_logit_top_k"),
        "input_side_hypothesis_cache_path": str(input_hypothesis_cache_path),
        "input_side_hypothesis_scoring_formula": input_hypothesis_cache.get("score_formula"),
        "input_side_best_hypothesis": best_input_hypothesis,
        "token_usage_total": state.get("total_tokens", _usage_zero()),
        "token_usage_this_run": state.get("run_tokens", _usage_zero()),
        "executed_steps": state.get("executed_steps", []),
        "loaded_steps": state.get("loaded_steps", []),
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
    lines.append(f"- max_rounds: {config.effective_max_rounds}")
    lines.append(f"- input_activation_max_rounds: {config.activation_max_rounds}")
    lines.append(f"- input_expansion_max_rounds: {config.expansion_max_rounds}")
    lines.append(f"- scheduled_input_total_rounds: {config.scheduled_input_total_rounds}")
    lines.append(f"- input_activation_rounds_done: {state.get('input_activation_rounds_done', 0)}")
    lines.append(f"- input_expansion_rounds_done: {state.get('input_expansion_rounds_done', 0)}")
    lines.append(f"- last_input_test_mode: {state.get('current_input_test_mode', 'activation')}")
    lines.append(f"- executed_rounds: {state.get('last_round_executed', 0)}")
    lines.append(f"- converged: {state.get('converged', False)}")
    lines.append(f"- converged_round: {state.get('converged_round')}")
    lines.append(f"- enable_hypothesis_merge: {config.merge_enabled}")
    lines.append(f"- hypothesis_merge_mode: {'llm_semantic' if config.merge_enabled else 'off'}")
    lines.append(f"- merged_rounds: {sorted(state.get('round_merges', {}).keys())}")
    lines.append(f"- output_intervention_method: {state.get('last_output_intervention_method')}")
    lines.append(f"- output_score_name: {state.get('last_output_score_name')}")
    lines.append(f"- output_logit_top_k: {state.get('last_output_logit_top_k')}")
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
    lines.append(f"- test_type: {best_input_hypothesis.get('test_type')}")
    lines.append(f"- score_non_zero_rate: {best_input_hypothesis.get('score_non_zero_rate')}")
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
    for item in final_result["executed_steps"]:
        lines.append(f"  - {item}")
    lines.append("- loaded:")
    for item in final_result["loaded_steps"]:
        lines.append(f"  - {item}")
    lines.append("")
    final_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "final_result_json_path": str(final_json_path),
                "final_result_md_path": str(final_md_path),
                "workflow_memory_md_path": str(workflow_memory_md),
                "converged": state.get("converged", False),
                "converged_round": state.get("converged_round"),
                "enable_hypothesis_merge": config.merge_enabled,
                "hypothesis_merge_mode": "llm_semantic" if config.merge_enabled else "off",
                "merged_rounds": sorted(state.get("round_merges", {}).keys()),
                "output_intervention_method": state.get("last_output_intervention_method"),
                "output_score_name": state.get("last_output_score_name"),
                "input_side_hypothesis_cache_path": final_result.get("input_side_hypothesis_cache_path"),
                "input_side_best_hypothesis": best_input_hypothesis,
                "input_final_hypothesis_count": len(final_input_hypotheses),
                "output_final_hypothesis_count": len(final_output_hypotheses),
                "token_usage_total": state.get("total_tokens", _usage_zero()),
                "token_usage_this_run": state.get("run_tokens", _usage_zero()),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return {"final_result": final_result}


def route_after_baseline(state: WorkflowState) -> str:
    return "finalize" if _should_stop_before_round(state) else "refine_hypotheses"


def route_after_memory(state: WorkflowState) -> str:
    if state.get("converged", False):
        return "finalize"
    next_round = state.get("current_round", 1) + 1
    if next_round > state["config"].effective_max_rounds:
        return "finalize"
    if state["config"].args.side in ("input", "both"):
        if (
            state.get("current_input_test_mode", "activation") == "expansion"
            and state.get("input_expansion_rounds_done", 0) >= state["config"].expansion_max_rounds
        ):
            return "finalize"
    return "advance_round"


def build_langgraph_workflow():
    if StateGraph is None:
        raise ImportError(
            "langgraph is not installed. Install it first, for example: pip install langgraph langchain-core"
        )

    graph = StateGraph(WorkflowState)
    graph.add_node("collect_observation", collect_observation_node)
    graph.add_node("generate_initial_hypotheses", generate_initial_hypotheses_node)
    graph.add_node("design_baseline_experiments", design_baseline_experiments_node)
    graph.add_node("execute_baseline_experiments", execute_baseline_experiments_node)
    graph.add_node("build_baseline_memory", build_baseline_memory_node)
    graph.add_node("refine_hypotheses", refine_hypotheses_node)
    graph.add_node("merge_hypotheses", merge_hypotheses_node)
    graph.add_node("prepare_round_hypotheses", prepare_round_hypotheses_node)
    graph.add_node("design_experiments", design_experiments_node)
    graph.add_node("execute_experiments", execute_experiments_node)
    graph.add_node("build_memory", build_memory_node)
    graph.add_node("advance_round", advance_round_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "collect_observation")
    graph.add_edge("collect_observation", "generate_initial_hypotheses")
    graph.add_edge("generate_initial_hypotheses", "design_baseline_experiments")
    graph.add_edge("design_baseline_experiments", "execute_baseline_experiments")
    graph.add_edge("execute_baseline_experiments", "build_baseline_memory")
    graph.add_conditional_edges(
        "build_baseline_memory",
        route_after_baseline,
        {"refine_hypotheses": "refine_hypotheses", "finalize": "finalize"},
    )
    graph.add_edge("refine_hypotheses", "merge_hypotheses")
    graph.add_edge("merge_hypotheses", "prepare_round_hypotheses")
    graph.add_edge("prepare_round_hypotheses", "design_experiments")
    graph.add_edge("design_experiments", "execute_experiments")
    graph.add_edge("execute_experiments", "build_memory")
    graph.add_conditional_edges(
        "build_memory",
        route_after_memory,
        {"advance_round": "advance_round", "finalize": "finalize"},
    )
    graph.add_edge("advance_round", "refine_hypotheses")
    graph.add_edge("finalize", END)
    return graph.compile()


def _validate_args(args: Any) -> WorkflowConfig:
    merge_enabled = bool(args.enable_hypothesis_merge)
    generation_model = str(args.llm_generation_model).strip() or DEFAULT_MODEL_NAME
    judge_model = str(args.llm_judge_model).strip() if args.llm_judge_model else generation_model
    activation_max_rounds = _safe_int(args.input_activation_max_rounds, 1)
    expansion_max_rounds = _safe_int(args.input_expansion_max_rounds, 1)
    if activation_max_rounds < 1:
        raise ValueError("--input-activation-max-rounds must be >= 1.")
    if expansion_max_rounds < 0:
        raise ValueError("--input-expansion-max-rounds must be >= 0.")
    scheduled_input_total_rounds = activation_max_rounds + expansion_max_rounds
    if scheduled_input_total_rounds < 1:
        raise ValueError("input-side total rounds p+q must be >= 1.")
    effective_max_rounds = max(scheduled_input_total_rounds - 1, 0)
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
    if args.start_round > effective_max_rounds and args.start_round > 0:
        raise ValueError("--start-round cannot be greater than effective max rounds.")
    if args.history_rounds is not None and args.history_rounds < 0:
        raise ValueError("--history-rounds must be >= 0 when provided.")
    if (args.start_round != 0 or args.start_step != 1) and not args.reuse_from_logs:
        raise ValueError("Resume from middle requires --reuse-from-logs.")
    if args.reuse_from_logs and args.timestamp is None:
        raise ValueError("When --reuse-from-logs is set, --timestamp is required.")

    return WorkflowConfig(
        args=args,
        layer_id=str(args.layer_id),
        feature_id=str(args.feature_id),
        timestamp=args.timestamp or time.strftime("%Y%m%d_%H%M%S"),
        merge_enabled=merge_enabled,
        generation_model=generation_model,
        judge_model=judge_model,
        activation_max_rounds=activation_max_rounds,
        expansion_max_rounds=expansion_max_rounds,
        scheduled_input_total_rounds=scheduled_input_total_rounds,
        effective_max_rounds=effective_max_rounds,
    )


def build_initial_state(config: WorkflowConfig) -> WorkflowState:
    args = config.args
    return WorkflowState(
        config=config,
        workflow_start_time=time.perf_counter(),
        total_tokens=TokenUsageAccumulator().as_dict(),
        run_tokens=TokenUsageAccumulator().as_dict(),
        executed_steps=[],
        loaded_steps=[],
        round_memories={},
        round_refinements={},
        round_merges={},
        round_executions={},
        module=None,
        converged=False,
        converged_round=None,
        last_round_executed=0,
        last_output_intervention_method=str(args.output_intervention_method),
        last_output_score_name="score_blind_accuracy",
        last_output_logit_top_k=args.output_logit_top_k,
        current_input_test_mode="activation",
        input_activation_rounds_done=0,
        input_expansion_rounds_done=0,
        current_round=1,
        previous_execution_result=None,
        previous_memory_result=None,
        previous_experiments_result=None,
        previous_round_hypotheses=None,
        refinement_result=None,
        merged_result=None,
        experiments_result=None,
        execution_result=None,
        memory_result=None,
        final_result=None,
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _validate_args(args)
    graph = build_langgraph_workflow()
    graph.invoke(build_initial_state(config))


if __name__ == "__main__":
    main()

LANGGRAPH_AVAILABLE = StateGraph is not None


def _build_langgraph_app():
    return build_langgraph_workflow()
