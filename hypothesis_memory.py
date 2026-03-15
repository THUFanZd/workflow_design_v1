from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from experiments_execution import execute_hypothesis_experiments
from experiments_execution_output import KL_DIV_VALUES_DEFAULT
from experiments_design import design_hypothesis_experiments
from initial_hypothesis_generation import generate_initial_hypotheses
from llm_api.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from llm_api.llm_api_info import base_url as DEFAULT_BASE_URL
from llm_api.llm_api_info import model_name as DEFAULT_MODEL_NAME
from model_with_sae import ModelWithSAEModule
from neuronpedia_feature_api import fetch_and_parse_feature_observation

SideType = Literal["input", "output"]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _extract_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            result.append(text)
    return result


def _normalize_reason_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(x) for x in value]
    if isinstance(value, dict):
        sortable_items: List[Tuple[int, str]] = []
        for key, item in value.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            sortable_items.append((index, _clean_text(item)))
        sortable_items.sort(key=lambda pair: pair[0])
        return [text for _, text in sortable_items]
    return []


def _extract_reasons_from_result(initial_hypotheses_result: Dict[str, Any], side: SideType) -> List[str]:
    if side == "input":
        keys = (
            "input_side_hypothesis_reasons",
            "input_side_reasons",
            "input_hypothesis_reasons",
        )
    else:
        keys = (
            "output_side_hypothesis_reasons",
            "output_side_reasons",
            "output_hypothesis_reasons",
        )

    for key in keys:
        if key in initial_hypotheses_result:
            reasons = _normalize_reason_list(initial_hypotheses_result.get(key))
            if reasons:
                return reasons
    return []


def _resolve_reason_map(
    *,
    initial_hypotheses_result: Dict[str, Any],
    hypothesis_reasons: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    input_base = _extract_reasons_from_result(initial_hypotheses_result, "input")
    output_base = _extract_reasons_from_result(initial_hypotheses_result, "output")

    if not isinstance(hypothesis_reasons, dict):
        return {"input": input_base, "output": output_base}

    input_override = _normalize_reason_list(
        hypothesis_reasons.get("input", hypothesis_reasons.get("input_side_reasons"))
    )
    output_override = _normalize_reason_list(
        hypothesis_reasons.get("output", hypothesis_reasons.get("output_side_reasons"))
    )

    return {
        "input": input_override if input_override else input_base,
        "output": output_override if output_override else output_base,
    }


def _reason_at(reasons: Sequence[str], index: int) -> str:
    if index < 1 or index > len(reasons):
        return ""
    return _clean_text(reasons[index - 1])


def _choose_hypothesis_text(
    *,
    initial_hypotheses: Sequence[str],
    experiments_item: Dict[str, Any],
    execution_item: Dict[str, Any],
    index: int,
) -> str:
    candidates: List[str] = []
    if 0 <= index - 1 < len(initial_hypotheses):
        candidates.append(_clean_text(initial_hypotheses[index - 1]))
    candidates.append(_clean_text(experiments_item.get("hypothesis")))
    candidates.append(_clean_text(execution_item.get("hypothesis")))
    for text in candidates:
        if text:
            return text
    return ""


def _match_input_sentence_result(
    sentence_results: Sequence[Dict[str, Any]],
    case_index: int,
    sentence: str,
) -> Dict[str, Any]:
    for item in sentence_results:
        if _safe_int(item.get("sentence_index"), default=-1) == case_index:
            return item
    for item in sentence_results:
        if _clean_text(item.get("sentence")) == sentence:
            return item
    return {}


def _build_input_hypothesis_memory(
    *,
    hypothesis_index: int,
    hypothesis_text: str,
    reason: str,
    experiments_item: Dict[str, Any],
    execution_item: Dict[str, Any],
) -> Dict[str, Any]:
    designed_sentences = _extract_string_list(experiments_item.get("designed_sentences"))
    sentence_results_raw = execution_item.get("sentence_results")
    sentence_results: List[Dict[str, Any]] = (
        [item for item in sentence_results_raw if isinstance(item, dict)]
        if isinstance(sentence_results_raw, list)
        else []
    )

    test_cases: List[Dict[str, Any]] = []
    if designed_sentences:
        for case_index, sentence in enumerate(designed_sentences, start=1):
            result = _match_input_sentence_result(sentence_results, case_index, sentence)
            is_non_zero = bool(result.get("is_non_zero", False))
            test_cases.append(
                {
                    "case_index": case_index,
                    "sentence": sentence,
                    "summary_activation": _safe_float(result.get("summary_activation"), 0.0),
                    "summary_activation_mean": _safe_float(result.get("summary_activation_mean"), 0.0),
                    "summary_activation_sum": _safe_float(result.get("summary_activation_sum"), 0.0),
                    "max_token_index": _safe_int(result.get("max_token_index"), 0),
                    "max_token": _clean_text(result.get("max_token")),
                    "is_non_zero": is_non_zero,
                    "failed": not is_non_zero,
                }
            )
    else:
        for item in sentence_results:
            sentence = _clean_text(item.get("sentence"))
            if not sentence:
                continue
            case_index = _safe_int(item.get("sentence_index"), default=len(test_cases) + 1)
            is_non_zero = bool(item.get("is_non_zero", False))
            test_cases.append(
                {
                    "case_index": case_index,
                    "sentence": sentence,
                    "summary_activation": _safe_float(item.get("summary_activation"), 0.0),
                    "summary_activation_mean": _safe_float(item.get("summary_activation_mean"), 0.0),
                    "summary_activation_sum": _safe_float(item.get("summary_activation_sum"), 0.0),
                    "max_token_index": _safe_int(item.get("max_token_index"), 0),
                    "max_token": _clean_text(item.get("max_token")),
                    "is_non_zero": is_non_zero,
                    "failed": not is_non_zero,
                }
            )

    failed_test_cases = [item for item in test_cases if bool(item.get("failed", False))]
    score_non_zero_rate = _safe_float(execution_item.get("score_non_zero_rate"), 0.0)

    return {
        "side": "input",
        "hypothesis_index": hypothesis_index,
        "hypothesis": hypothesis_text,
        "reason": reason,
        "score": score_non_zero_rate,
        "score_non_zero_rate": score_non_zero_rate,
        "non_zero_count": _safe_int(execution_item.get("non_zero_count"), 0),
        "total_sentences": _safe_int(execution_item.get("total_sentences"), len(test_cases)),
        "test_cases": test_cases,
        "failed_test_cases": failed_test_cases,
    }


def _build_output_hypothesis_memory(
    *,
    hypothesis_index: int,
    hypothesis_text: str,
    reason: str,
    experiments_item: Dict[str, Any],
    execution_item: Dict[str, Any],
) -> Dict[str, Any]:
    designed_prompts = _extract_string_list(experiments_item.get("designed_sentences"))
    if not designed_prompts:
        designed_prompts = _extract_string_list(execution_item.get("designed_sentences"))

    test_cases = [
        {
            "case_index": index,
            "prompt": prompt,
        }
        for index, prompt in enumerate(designed_prompts, start=1)
    ]

    trial_results_raw = execution_item.get("trial_results")
    trial_results: List[Dict[str, Any]] = (
        [item for item in trial_results_raw if isinstance(item, dict)]
        if isinstance(trial_results_raw, list)
        else []
    )
    normalized_trials: List[Dict[str, Any]] = []
    failed_trials: List[Dict[str, Any]] = []
    for trial in trial_results:
        normalized = {
            "trial_index": _safe_int(trial.get("trial_index"), 0),
            "correct_choice": _safe_int(trial.get("correct_choice"), -1),
            "chosen_choice": _safe_int(trial.get("chosen_choice"), -1),
            "success": bool(trial.get("success", False)),
            "judge_response": _clean_text(trial.get("judge_response")),
        }
        normalized_trials.append(normalized)
        if not normalized["success"]:
            failed_trials.append(normalized)

    score_blind_accuracy = _safe_float(execution_item.get("score_blind_accuracy"), 0.0)
    kl_values = execution_item.get("kl_values")
    normalized_kl_values = [_safe_float(x) for x in kl_values] if isinstance(kl_values, list) else []

    return {
        "side": "output",
        "hypothesis_index": hypothesis_index,
        "hypothesis": hypothesis_text,
        "reason": reason,
        "score": score_blind_accuracy,
        "score_blind_accuracy": score_blind_accuracy,
        "blind_judge_successes": _safe_int(execution_item.get("blind_judge_successes"), 0),
        "blind_judge_trials": _safe_int(execution_item.get("blind_judge_trials"), len(normalized_trials)),
        "kl_values": normalized_kl_values,
        "test_cases": test_cases,
        "intervention_result": _clean_text(execution_item.get("intervention_result")),
        "trial_results": normalized_trials,
        "failed_test_cases": failed_trials,
    }


def build_hypothesis_memory(
    *,
    initial_hypotheses_result: Dict[str, Any],
    experiments_result: Dict[str, Any],
    execution_result: Dict[str, Any],
    hypothesis_reasons: Optional[Dict[str, Any]] = None,
    round_index: int = 1,
    round_id: Optional[str] = None,
) -> Dict[str, Any]:
    model_id = _clean_text(
        execution_result.get("model_id")
        or experiments_result.get("model_id")
        or initial_hypotheses_result.get("model_id")
        or "unknown-model"
    )
    layer_id = _clean_text(
        execution_result.get("layer_id")
        or experiments_result.get("layer_id")
        or initial_hypotheses_result.get("layer_id")
    )
    feature_id = _clean_text(
        execution_result.get("feature_id")
        or experiments_result.get("feature_id")
        or initial_hypotheses_result.get("feature_id")
    )
    ts = _clean_text(
        execution_result.get("timestamp")
        or experiments_result.get("timestamp")
        or initial_hypotheses_result.get("timestamp")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    reasons = _resolve_reason_map(
        initial_hypotheses_result=initial_hypotheses_result,
        hypothesis_reasons=hypothesis_reasons,
    )

    input_hypotheses = _extract_string_list(initial_hypotheses_result.get("input_side_hypotheses"))
    output_hypotheses = _extract_string_list(initial_hypotheses_result.get("output_side_hypotheses"))

    input_experiments_raw = experiments_result.get("input_side_experiments")
    output_experiments_raw = experiments_result.get("output_side_experiments")
    input_experiments = (
        [item for item in input_experiments_raw if isinstance(item, dict)]
        if isinstance(input_experiments_raw, list)
        else []
    )
    output_experiments = (
        [item for item in output_experiments_raw if isinstance(item, dict)]
        if isinstance(output_experiments_raw, list)
        else []
    )

    input_execution = execution_result.get("input_side_execution", {})
    output_execution = execution_result.get("output_side_execution", {})
    input_execution_raw = input_execution.get("hypothesis_results", [])
    output_execution_raw = output_execution.get("hypothesis_results", [])
    input_execution_list = (
        [item for item in input_execution_raw if isinstance(item, dict)]
        if isinstance(input_execution_raw, list)
        else []
    )
    output_execution_list = (
        [item for item in output_execution_raw if isinstance(item, dict)]
        if isinstance(output_execution_raw, list)
        else []
    )

    input_count = max(len(input_hypotheses), len(input_experiments), len(input_execution_list), len(reasons["input"]))
    output_count = max(
        len(output_hypotheses),
        len(output_experiments),
        len(output_execution_list),
        len(reasons["output"]),
    )

    input_memories: List[Dict[str, Any]] = []
    for index in range(1, input_count + 1):
        experiments_item = input_experiments[index - 1] if index - 1 < len(input_experiments) else {}
        execution_item = input_execution_list[index - 1] if index - 1 < len(input_execution_list) else {}
        input_memories.append(
            _build_input_hypothesis_memory(
                hypothesis_index=index,
                hypothesis_text=_choose_hypothesis_text(
                    initial_hypotheses=input_hypotheses,
                    experiments_item=experiments_item,
                    execution_item=execution_item,
                    index=index,
                ),
                reason=_reason_at(reasons["input"], index),
                experiments_item=experiments_item,
                execution_item=execution_item,
            )
        )

    output_memories: List[Dict[str, Any]] = []
    for index in range(1, output_count + 1):
        experiments_item = output_experiments[index - 1] if index - 1 < len(output_experiments) else {}
        execution_item = output_execution_list[index - 1] if index - 1 < len(output_execution_list) else {}
        output_memories.append(
            _build_output_hypothesis_memory(
                hypothesis_index=index,
                hypothesis_text=_choose_hypothesis_text(
                    initial_hypotheses=output_hypotheses,
                    experiments_item=experiments_item,
                    execution_item=execution_item,
                    index=index,
                ),
                reason=_reason_at(reasons["output"], index),
                experiments_item=experiments_item,
                execution_item=execution_item,
            )
        )

    sides: Dict[str, Dict[str, Any]] = {
        "input": {
            "overall_score_non_zero_rate": _safe_float(input_execution.get("overall_score_non_zero_rate"), 0.0),
            "hypotheses": input_memories,
        },
        "output": {
            "overall_score_blind_accuracy": _safe_float(output_execution.get("overall_score_blind_accuracy"), 0.0),
            "hypotheses": output_memories,
        },
    }

    lookup_by_key: Dict[str, Dict[str, Any]] = {}
    for side_name in ("input", "output"):
        for item in sides[side_name]["hypotheses"]:
            key = f"{side_name}:{item['hypothesis_index']}"
            lookup_by_key[key] = item

    return {
        "round_index": _safe_int(round_index, 1),
        "round_id": _clean_text(round_id) or ts,
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "sides": sides,
        "lookup_by_key": lookup_by_key,
    }


def get_hypothesis_memory(memory: Dict[str, Any], *, side: SideType, hypothesis_index: int) -> Optional[Dict[str, Any]]:
    key = f"{side}:{hypothesis_index}"
    lookup = memory.get("lookup_by_key", {})
    if not isinstance(lookup, dict):
        return None
    item = lookup.get(key)
    return item if isinstance(item, dict) else None


def get_side_hypothesis_memories(memory: Dict[str, Any], *, side: SideType) -> List[Dict[str, Any]]:
    sides = memory.get("sides", {})
    if not isinstance(sides, dict):
        return []
    side_data = sides.get(side, {})
    if not isinstance(side_data, dict):
        return []
    hypotheses = side_data.get("hypotheses", [])
    return [item for item in hypotheses if isinstance(item, dict)] if isinstance(hypotheses, list) else []


def write_hypothesis_memory_markdown(path: Path, *, memory: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# SAE Hypothesis Memory")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- round_index: {memory.get('round_index', 1)}")
    lines.append(f"- round_id: {memory.get('round_id', '')}")
    lines.append(f"- model_id: {memory.get('model_id', '')}")
    lines.append(f"- layer_id: {memory.get('layer_id', '')}")
    lines.append(f"- feature_id: {memory.get('feature_id', '')}")
    lines.append(f"- timestamp: {memory.get('timestamp', '')}")
    lines.append("")

    sides = memory.get("sides", {})
    input_side = sides.get("input", {}) if isinstance(sides, dict) else {}
    output_side = sides.get("output", {}) if isinstance(sides, dict) else {}

    lines.append("## Input-side Memory")
    lines.append(f"- overall_score_non_zero_rate: {input_side.get('overall_score_non_zero_rate', 0.0)}")
    lines.append("")
    input_hypotheses = input_side.get("hypotheses", []) if isinstance(input_side, dict) else []
    for item in input_hypotheses:
        if not isinstance(item, dict):
            continue
        lines.append(f"### Input Hypothesis {item.get('hypothesis_index', '')}")
        lines.append(f"- hypothesis: {item.get('hypothesis', '')}")
        lines.append(f"- reason: {item.get('reason', '')}")
        lines.append(f"- score_non_zero_rate: {item.get('score_non_zero_rate', 0.0)}")
        failed_cases = item.get("failed_test_cases", [])
        failed_count = len(failed_cases) if isinstance(failed_cases, list) else 0
        lines.append(f"- failed_case_count: {failed_count}")
        lines.append("")
        lines.append("| case_index | sentence | summary_activation | max_token | is_non_zero | failed |")
        lines.append("| --- | --- | ---: | --- | --- | --- |")
        test_cases = item.get("test_cases", [])
        if isinstance(test_cases, list) and test_cases:
            for case in test_cases:
                if not isinstance(case, dict):
                    continue
                sentence = str(case.get("sentence", "")).replace("|", "\\|")
                lines.append(
                    f"| {case.get('case_index', '')} | {sentence} | {case.get('summary_activation', 0.0)} | "
                    f"{case.get('max_token', '')} | {case.get('is_non_zero', False)} | {case.get('failed', False)} |"
                )
        else:
            lines.append("| - | - | - | - | - | - |")
        lines.append("")

    lines.append("## Output-side Memory")
    lines.append(f"- overall_score_blind_accuracy: {output_side.get('overall_score_blind_accuracy', 0.0)}")
    lines.append("")
    output_hypotheses = output_side.get("hypotheses", []) if isinstance(output_side, dict) else []
    for item in output_hypotheses:
        if not isinstance(item, dict):
            continue
        lines.append(f"### Output Hypothesis {item.get('hypothesis_index', '')}")
        lines.append(f"- hypothesis: {item.get('hypothesis', '')}")
        lines.append(f"- reason: {item.get('reason', '')}")
        lines.append(f"- score_blind_accuracy: {item.get('score_blind_accuracy', 0.0)}")
        lines.append(f"- kl_values: {item.get('kl_values', [])}")
        failed_cases = item.get("failed_test_cases", [])
        failed_count = len(failed_cases) if isinstance(failed_cases, list) else 0
        lines.append(f"- failed_trial_count: {failed_count}")
        lines.append("")
        lines.append("#### Designed Prompts")
        test_cases = item.get("test_cases", [])
        if isinstance(test_cases, list) and test_cases:
            for case in test_cases:
                if not isinstance(case, dict):
                    continue
                lines.append(f"- [{case.get('case_index', '')}] {case.get('prompt', '')}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("#### Trial Results")
        lines.append("| trial_index | correct_choice | chosen_choice | success |")
        lines.append("| --- | ---: | ---: | --- |")
        trial_results = item.get("trial_results", [])
        if isinstance(trial_results, list) and trial_results:
            for trial in trial_results:
                if not isinstance(trial, dict):
                    continue
                lines.append(
                    f"| {trial.get('trial_index', '')} | {trial.get('correct_choice', '')} | "
                    f"{trial.get('chosen_choice', '')} | {trial.get('success', False)} |"
                )
        else:
            lines.append("| - | - | - | - |")
        lines.append("")
        lines.append("#### Intervention Result")
        lines.append("```text")
        lines.append(str(item.get("intervention_result", "")))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_average_l0_from_canonical_map(
    *,
    canonical_map_path: Path,
    layer_id: str,
    width: str,
) -> Optional[str]:
    if not canonical_map_path.exists():
        return None

    target_id = f"layer_{layer_id}/width_{width}/canonical"
    in_target_block = False
    path_pattern = re.compile(
        rf"layer_{re.escape(layer_id)}/width_{re.escape(width)}/average_l0_([0-9]+(?:\.[0-9]+)?)"
    )

    with canonical_map_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("- id:"):
                current_id = line.split(":", 1)[1].strip()
                in_target_block = current_id == target_id
                continue

            if in_target_block and line.startswith("path:"):
                match = path_pattern.search(line.split(":", 1)[1].strip())
                if match:
                    return match.group(1)
                return None
    return None


def _default_sae_path(
    *,
    layer_id: str,
    width: str,
    release: str,
    average_l0: Optional[str],
    canonical_map_path: Optional[str],
) -> str:
    resolved_average_l0 = average_l0
    if not resolved_average_l0 and canonical_map_path:
        resolved_average_l0 = _extract_average_l0_from_canonical_map(
            canonical_map_path=Path(canonical_map_path),
            layer_id=layer_id,
            width=width,
        )

    if not resolved_average_l0:
        resolved_average_l0 = "70"

    return (
        "sae-lens://"
        f"release={release};"
        f"sae_id=layer_{layer_id}/width_{width}/average_l0_{resolved_average_l0}"
    )


def _load_control_results(control_result_files: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for file_path in control_result_files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find control result file: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())
    return texts


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hypothesis_reasons(reasons_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not reasons_json_path:
        return None
    payload = _load_json(Path(reasons_json_path))
    if not isinstance(payload, dict):
        raise ValueError("Reasons JSON must be a dict.")
    return payload


def _load_from_logs(
    *,
    layer_id: str,
    feature_id: str,
    timestamp: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base_dir = Path("logs") / f"{layer_id}_{feature_id}" / timestamp
    initial_path = base_dir / f"layer{layer_id}-feature{feature_id}-initial-hypotheses.json"
    experiments_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments.json"
    execution_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments-execution.json"
    return _load_json(initial_path), _load_json(experiments_path), _load_json(execution_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 5 of SAE workflow: build memory for each hypothesis from step2/3/4 results.",
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id")
    parser.add_argument("--layer-id", required=True, help="Layer id")
    parser.add_argument("--feature-id", required=True, help="Feature id")
    parser.add_argument("--num-hypothesis", type=int, default=3, help="Hypothesis count n for each side")
    parser.add_argument(
        "--generation-mode",
        choices=["single_call", "iterative"],
        default="single_call",
        help="Generation mode used in initial hypothesis generation.",
    )
    parser.add_argument(
        "--num-input-sentences-per-hypothesis",
        type=int,
        default=5,
        help="For each input-side hypothesis, generate this many activation sentences.",
    )
    parser.add_argument("--width", default="16k", help="Neuronpedia source width")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument("--timestamp", default=None, help="Custom timestamp for logs/{layer}_{feature}/{timestamp}")
    parser.add_argument("--round-index", type=int, default=1, help="Round index used as memory anchor.")
    parser.add_argument("--round-id", default=None, help="Optional explicit round id for the memory record.")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, reuse logs/{layer}_{feature}/{timestamp} intermediate JSON files.",
    )
    parser.add_argument(
        "--initial-hypotheses-json-path",
        default=None,
        help="Optional direct path to layer{layer}-feature{feature}-initial-hypotheses.json.",
    )
    parser.add_argument(
        "--experiments-json-path",
        default=None,
        help="Optional direct path to layer{layer}-feature{feature}-experiments.json.",
    )
    parser.add_argument(
        "--execution-json-path",
        default=None,
        help="Optional direct path to layer{layer}-feature{feature}-experiments-execution.json.",
    )
    parser.add_argument(
        "--reasons-json-path",
        default=None,
        help="Optional JSON path with hypothesis reasons, e.g. {'input': [...], 'output': [...]}",
    )

    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=20000)

    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--sae-path", default=None, help="SAE path or sae-lens URI")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    parser.add_argument(
        "--sae-average-l0",
        default=None,
        help="Optional explicit average_l0 suffix. If not provided, resolve from canonical_map.txt.",
    )
    parser.add_argument(
        "--sae-canonical-map",
        default=str(Path("model_download") / "canonical_map.txt"),
        help="Path to canonical_map.txt used to resolve average_l0 when --sae-average-l0 is omitted.",
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
        "--control-result-files",
        nargs="*",
        default=[
            "explanation_quality_evaluation/output-side-evaluation/intervention_example_2.txt",
            "explanation_quality_evaluation/output-side-evaluation/intervention_example_3.txt",
        ],
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.execution_json_path:
        execution_result = _load_json(Path(args.execution_json_path))
        layer_id = _clean_text(execution_result.get("layer_id") or args.layer_id)
        feature_id = _clean_text(execution_result.get("feature_id") or args.feature_id)
        ts = _clean_text(execution_result.get("timestamp") or ts)

        default_base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts
        initial_path = (
            Path(args.initial_hypotheses_json_path)
            if args.initial_hypotheses_json_path
            else default_base_dir / f"layer{layer_id}-feature{feature_id}-initial-hypotheses.json"
        )
        experiments_path = (
            Path(args.experiments_json_path)
            if args.experiments_json_path
            else default_base_dir / f"layer{layer_id}-feature{feature_id}-experiments.json"
        )
        initial_result = _load_json(initial_path)
        experiments_result = _load_json(experiments_path)
    elif args.reuse_from_logs:
        if args.timestamp is None:
            raise ValueError("When --reuse-from-logs is set, --timestamp is required.")
        print(f"Reusing logs from {ts} for layer {args.layer_id} feature {args.feature_id}")
        initial_result, experiments_result, execution_result = _load_from_logs(
            layer_id=args.layer_id,
            feature_id=args.feature_id,
            timestamp=ts,
        )
    else:
        observation = fetch_and_parse_feature_observation(
            model_id=args.model_id,
            layer_id=args.layer_id,
            feature_id=args.feature_id,
            width=args.width,
            selection_method=args.selection_method,
            m=args.observation_m,
            n=args.observation_n,
            api_key=args.neuronpedia_api_key,
            timeout=args.neuronpedia_timeout,
            timestamp=ts,
        )
        initial_result = generate_initial_hypotheses(
            observation=observation,
            model_id=args.model_id,
            layer_id=args.layer_id,
            feature_id=args.feature_id,
            num_hypothesis=args.num_hypothesis,
            generation_mode=args.generation_mode,
            timestamp=ts,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        experiments_result = design_hypothesis_experiments(
            hypotheses_result=initial_result,
            num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        layer_id = str(experiments_result["layer_id"])
        feature_id = str(experiments_result["feature_id"])
        sae_path = args.sae_path or _default_sae_path(
            layer_id=layer_id,
            width=args.width,
            release=args.sae_release,
            average_l0=args.sae_average_l0,
            canonical_map_path=args.sae_canonical_map,
        )
        module = ModelWithSAEModule(
            llm_name=args.model_checkpoint_path,
            sae_path=sae_path,
            sae_layer=int(layer_id),
            feature_index=int(feature_id),
            device=args.device,
        )
        control_results = _load_control_results(args.control_result_files)
        execution_result = execute_hypothesis_experiments(
            experiments_result=experiments_result,
            module=module,
            control_results=control_results,
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
        )

    reasons = _load_hypothesis_reasons(args.reasons_json_path)
    memory = build_hypothesis_memory(
        initial_hypotheses_result=initial_result,
        experiments_result=experiments_result,
        execution_result=execution_result,
        hypothesis_reasons=reasons,
        round_index=args.round_index,
        round_id=args.round_id,
    )

    layer_id = _clean_text(memory.get("layer_id"))
    feature_id = _clean_text(memory.get("feature_id"))
    ts = _clean_text(memory.get("timestamp"))
    base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts
    base_dir.mkdir(parents=True, exist_ok=True)

    memory_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-memory.json"
    memory_json_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    memory_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-memory.md"
    write_hypothesis_memory_markdown(memory_md_path, memory=memory)

    print(
        json.dumps(
            {
                "memory_json_path": str(memory_json_path),
                "memory_md_path": str(memory_md_path),
                "round_id": memory.get("round_id", ""),
                "round_index": memory.get("round_index", 1),
                "hypothesis_counts": {
                    "input": len(get_side_hypothesis_memories(memory, side="input")),
                    "output": len(get_side_hypothesis_memories(memory, side="output")),
                },
            },
            ensure_ascii=True,
            indent=2,
        )
    )
