
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from openai import OpenAI

from function import (
    TokenUsageAccumulator,
    build_round_dir,
    call_llm,
    extract_json_object,
    normalize_round_id,
    read_api_key,
)
from llm_api.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from llm_api.llm_api_info import base_url as DEFAULT_BASE_URL
from llm_api.llm_api_info import model_name as DEFAULT_MODEL_NAME
from prompts.refine_prompt import HistoryScope, build_system_prompt, build_user_prompt

SideType = Literal["input", "output"]
KL_DIV_VALUES_DEFAULT = [0.25, 0.5, -0.25, -0.5]


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a dict: {path}")
    return payload


def _score_key(side: SideType) -> str:
    return "score_non_zero_rate" if side == "input" else "score_blind_accuracy"


def _find_success_example(hypothesis_memory: Dict[str, Any], side: SideType) -> Dict[str, Any]:
    if side == "input":
        test_cases = hypothesis_memory.get("test_cases", [])
        if not isinstance(test_cases, list):
            return {}
        successes = [case for case in test_cases if isinstance(case, dict) and not bool(case.get("failed", False))]
        if not successes:
            return {}
        successes.sort(key=lambda item: _safe_float(item.get("summary_activation"), 0.0), reverse=True)
        return dict(successes[0])

    trial_results = hypothesis_memory.get("trial_results", [])
    if not isinstance(trial_results, list):
        return {}
    for trial in trial_results:
        if isinstance(trial, dict) and bool(trial.get("success", False)):
            return dict(trial)
    return {}


def _find_failed_examples(hypothesis_memory: Dict[str, Any], side: SideType) -> List[Dict[str, Any]]:
    failed = hypothesis_memory.get("failed_test_cases")
    if isinstance(failed, list):
        return [dict(item) for item in failed if isinstance(item, dict)]

    if side == "input":
        test_cases = hypothesis_memory.get("test_cases", [])
        if not isinstance(test_cases, list):
            return []
        return [dict(item) for item in test_cases if isinstance(item, dict) and bool(item.get("failed", False))]

    trial_results = hypothesis_memory.get("trial_results", [])
    if not isinstance(trial_results, list):
        return []
    return [dict(item) for item in trial_results if isinstance(item, dict) and not bool(item.get("success", False))]


def extract_refinement_evidence_from_memory(
    *,
    memory: Dict[str, Any],
    side: SideType,
    hypothesis_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract compact refinement evidence from one memory dict.
    Requirement coverage:
    1) keep current hypothesis + reason
    2) keep exactly one successful example and all failed examples, with score fields
    """
    sides = memory.get("sides", {})
    if not isinstance(sides, dict):
        return {
            "round_index": _safe_int(memory.get("round_index"), 0),
            "round_id": _clean_text(memory.get("round_id")),
            "timestamp": _clean_text(memory.get("timestamp")),
            "side": side,
            "hypotheses": [],
        }

    side_data = sides.get(side, {})
    hypotheses_raw = side_data.get("hypotheses", []) if isinstance(side_data, dict) else []
    hypotheses_list = [item for item in hypotheses_raw if isinstance(item, dict)] if isinstance(hypotheses_raw, list) else []

    compact_hypotheses: List[Dict[str, Any]] = []
    for item in hypotheses_list:
        index = _safe_int(item.get("hypothesis_index"), 0)
        if hypothesis_index is not None and index != hypothesis_index:
            continue

        score_name = _score_key(side)
        score_value = _safe_float(item.get(score_name, item.get("score", 0.0)), 0.0)
        compact_hypotheses.append(
            {
                "hypothesis_index": index,
                "hypothesis": _clean_text(item.get("hypothesis")),
                "reason": _clean_text(item.get("reason")),
                "score_name": score_name,
                "score_value": score_value,
                "successful_example": _find_success_example(item, side),
                "failed_examples": _find_failed_examples(item, side),
            }
        )

    return {
        "round_index": _safe_int(memory.get("round_index"), 0),
        "round_id": _clean_text(memory.get("round_id")),
        "timestamp": _clean_text(memory.get("timestamp")),
        "side": side,
        "hypotheses": compact_hypotheses,
    }

def _extract_execution_evidence(
    *,
    execution_result: Dict[str, Any],
    side: SideType,
    hypothesis_index: int,
) -> Dict[str, Any]:
    if side == "input":
        side_data = execution_result.get("input_side_execution", {})
    else:
        side_data = execution_result.get("output_side_execution", {})
    if not isinstance(side_data, dict):
        return {}

    results_raw = side_data.get("hypothesis_results", [])
    results = [item for item in results_raw if isinstance(item, dict)] if isinstance(results_raw, list) else []

    target = next((item for item in results if _safe_int(item.get("hypothesis_index"), -1) == hypothesis_index), None)
    if target is None and 0 <= hypothesis_index - 1 < len(results):
        target = results[hypothesis_index - 1]
    if target is None:
        return {}

    if side == "input":
        sentence_results_raw = target.get("sentence_results", [])
        sentence_results = (
            [item for item in sentence_results_raw if isinstance(item, dict)]
            if isinstance(sentence_results_raw, list)
            else []
        )
        successes = [item for item in sentence_results if bool(item.get("is_non_zero", False))]
        successes.sort(key=lambda item: _safe_float(item.get("summary_activation"), 0.0), reverse=True)
        failed = [item for item in sentence_results if not bool(item.get("is_non_zero", False))]
        return {
            "score_non_zero_rate": _safe_float(target.get("score_non_zero_rate"), 0.0),
            "non_zero_count": _safe_int(target.get("non_zero_count"), 0),
            "total_sentences": _safe_int(target.get("total_sentences"), len(sentence_results)),
            "successful_example": dict(successes[0]) if successes else {},
            "failed_examples": [dict(item) for item in failed],
        }

    trials_raw = target.get("trial_results", [])
    trials = [item for item in trials_raw if isinstance(item, dict)] if isinstance(trials_raw, list) else []
    success_example = next((dict(item) for item in trials if bool(item.get("success", False))), {})
    failed_examples = [dict(item) for item in trials if not bool(item.get("success", False))]
    return {
        "score_blind_accuracy": _safe_float(target.get("score_blind_accuracy"), 0.0),
        "blind_judge_successes": _safe_int(target.get("blind_judge_successes"), 0),
        "blind_judge_trials": _safe_int(target.get("blind_judge_trials"), len(trials)),
        "successful_example": success_example,
        "failed_examples": failed_examples,
    }


def _select_top_hypotheses(
    *,
    memory: Dict[str, Any],
    side: SideType,
    top_m: int,
) -> List[Dict[str, Any]]:
    extracted = extract_refinement_evidence_from_memory(memory=memory, side=side)
    candidates = list(extracted.get("hypotheses", []))
    if top_m <= 0:
        return []
    candidates.sort(
        key=lambda item: (
            -_safe_float(item.get("score_value"), 0.0),
            _safe_int(item.get("hypothesis_index"), 10**9),
        )
    )
    return candidates[:top_m]


def _parse_refinement_output(raw_output: str) -> Tuple[str, str]:
    parsed = extract_json_object(raw_output)
    if not isinstance(parsed, dict):
        raise ValueError(f"Cannot parse JSON object from refinement output: {raw_output}")

    reason = _clean_text(parsed.get("reason"))
    if not reason:
        reason = _clean_text(parsed.get("rationale"))
    hypothesis = _clean_text(parsed.get("hypothesis"))
    if not hypothesis:
        hypothesis = _clean_text(parsed.get("revised_hypothesis"))

    if not reason or not hypothesis:
        raise ValueError(f"Refinement output must contain both reason and hypothesis: {raw_output}")
    return reason, hypothesis


def _build_history_evidence(
    *,
    historical_memories: Sequence[Dict[str, Any]],
    side: SideType,
    hypothesis_index: int,
    history_scope: HistoryScope,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for memory in historical_memories:
        if history_scope == "same_hypothesis":
            round_view = extract_refinement_evidence_from_memory(
                memory=memory,
                side=side,
                hypothesis_index=hypothesis_index,
            )
        else:
            round_view = extract_refinement_evidence_from_memory(memory=memory, side=side)
        if round_view.get("hypotheses"):
            evidence.append(round_view)
    return evidence


def refine_hypotheses_for_side(
    *,
    side: SideType,
    current_memory: Dict[str, Any],
    current_execution_result: Dict[str, Any],
    historical_memories: Sequence[Dict[str, Any]],
    top_m: int,
    history_scope: HistoryScope,
    client: OpenAI,
    llm_model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    selected_hypotheses = _select_top_hypotheses(memory=current_memory, side=side, top_m=top_m)
    refined: List[Dict[str, Any]] = []

    for item in selected_hypotheses:
        hypothesis_index = _safe_int(item.get("hypothesis_index"), 0)
        current_hypothesis = _clean_text(item.get("hypothesis"))
        current_reason = _clean_text(item.get("reason"))
        current_score_name = _clean_text(item.get("score_name"))
        current_score = _safe_float(item.get("score_value"), 0.0)
        current_success_example = item.get("successful_example", {})
        current_failed_examples = item.get("failed_examples", [])
        history_evidence = _build_history_evidence(
            historical_memories=historical_memories,
            side=side,
            hypothesis_index=hypothesis_index,
            history_scope=history_scope,
        )
        current_execution_evidence = _extract_execution_evidence(
            execution_result=current_execution_result,
            side=side,
            hypothesis_index=hypothesis_index,
        )

        system_prompt = build_system_prompt(side)
        user_prompt = build_user_prompt(
            side=side,
            hypothesis_index=hypothesis_index,
            current_hypothesis=current_hypothesis,
            current_reason=current_reason,
            current_score_name=current_score_name,
            current_score=current_score,
            current_success_example=current_success_example if isinstance(current_success_example, dict) else {},
            current_failed_examples=current_failed_examples if isinstance(current_failed_examples, list) else [],
            history_scope=history_scope,
            historical_evidence=history_evidence,
            current_execution_evidence=current_execution_evidence,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_output, usage_obj, response_debug = call_llm(
            client=client,
            model=llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            response_format_text=True,
            return_debug=True,
        )
        usage_counts = token_counter.add(usage_obj)
        refined_reason, refined_hypothesis = _parse_refinement_output(raw_output)

        refined_item: Dict[str, Any] = {
            "side": side,
            "hypothesis_index": hypothesis_index,
            "original_hypothesis": current_hypothesis,
            "original_reason": current_reason,
            "score_name": current_score_name,
            "score_value": current_score,
            "evidence": {
                "current_memory": {
                    "successful_example": current_success_example,
                    "failed_examples": current_failed_examples,
                },
                "current_execution": current_execution_evidence,
                "historical_memory": history_evidence,
                "history_scope": history_scope,
            },
            "refined_reason": refined_reason,
            "refined_hypothesis": refined_hypothesis,
        }
        refined.append(refined_item)

        llm_calls.append(
            {
                "side": side,
                "hypothesis_index": hypothesis_index,
                "messages": messages,
                "raw_output": raw_output,
                "parsed_output": {
                    "reason": refined_reason,
                    "hypothesis": refined_hypothesis,
                },
                "usage": usage_counts,
                "response_debug": response_debug,
            }
        )

    return refined


def _write_refinement_markdown(
    path: Path,
    *,
    result: Dict[str, Any],
    llm_calls: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# SAE Hypothesis Refinement")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- model_id: {result['model_id']}")
    lines.append(f"- layer_id: {result['layer_id']}")
    lines.append(f"- feature_id: {result['feature_id']}")
    lines.append(f"- timestamp: {result['timestamp']}")
    if "round_id" in result:
        lines.append(f"- round_id: {result['round_id']}")
    lines.append(f"- top_m: {result['top_m']}")
    lines.append(f"- history_rounds: {result['history_rounds']}")
    lines.append(f"- history_scope: {result['history_scope']}")
    lines.append(f"- llm_model: {result['llm_model']}")
    lines.append("")
    lines.append("## Token Usage (Hypothesis Refinement)")
    token_usage = result["token_usage"]
    lines.append(f"- prompt_tokens: {token_usage['prompt_tokens']}")
    lines.append(f"- completion_tokens: {token_usage['completion_tokens']}")
    lines.append(f"- total_tokens: {token_usage['total_tokens']}")
    lines.append("")

    lines.append("## Input-side Refined Hypotheses")
    for item in result["refined_hypotheses"]["input"]:
        lines.append(f"### Input Hypothesis {item['hypothesis_index']}")
        lines.append(f"- original_hypothesis: {item['original_hypothesis']}")
        lines.append(f"- original_reason: {item['original_reason']}")
        lines.append(f"- {item['score_name']}: {item['score_value']}")
        lines.append(f"- refined_reason: {item['refined_reason']}")
        lines.append(f"- refined_hypothesis: {item['refined_hypothesis']}")
        lines.append("")
        lines.append("#### Evidence Used")
        lines.append("```json")
        lines.append(json.dumps(item.get("evidence", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Output-side Refined Hypotheses")
    for item in result["refined_hypotheses"]["output"]:
        lines.append(f"### Output Hypothesis {item['hypothesis_index']}")
        lines.append(f"- original_hypothesis: {item['original_hypothesis']}")
        lines.append(f"- original_reason: {item['original_reason']}")
        lines.append(f"- {item['score_name']}: {item['score_value']}")
        lines.append(f"- refined_reason: {item['refined_reason']}")
        lines.append(f"- refined_hypothesis: {item['refined_hypothesis']}")
        lines.append("")
        lines.append("#### Evidence Used")
        lines.append("```json")
        lines.append(json.dumps(item.get("evidence", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## LLM API Calls")
    for call_index, call in enumerate(llm_calls, start=1):
        lines.append(f"### Call {call_index}")
        lines.append(f"- side: {call.get('side', '')}")
        lines.append(f"- hypothesis_index: {call.get('hypothesis_index', '')}")
        usage = call.get("usage", {})
        lines.append(f"- prompt_tokens: {usage.get('prompt_tokens', 0)}")
        lines.append(f"- completion_tokens: {usage.get('completion_tokens', 0)}")
        lines.append(f"- total_tokens: {usage.get('total_tokens', 0)}")
        lines.append("")
        lines.append("#### Messages")
        lines.append("```json")
        lines.append(json.dumps(call.get("messages", []), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("#### Raw Output")
        lines.append("```text")
        lines.append(_clean_text(call.get("raw_output", "")))
        lines.append("```")
        lines.append("")
        lines.append("#### Parsed Output")
        lines.append("```json")
        lines.append(json.dumps(call.get("parsed_output", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_memory_markdown(path: Path, *, memory: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# SAE Hypothesis Memory Snapshot")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- round_index: {memory.get('round_index', '')}")
    lines.append(f"- round_id: {memory.get('round_id', '')}")
    lines.append(f"- model_id: {memory.get('model_id', '')}")
    lines.append(f"- layer_id: {memory.get('layer_id', '')}")
    lines.append(f"- feature_id: {memory.get('feature_id', '')}")
    lines.append(f"- timestamp: {memory.get('timestamp', '')}")
    lines.append("")
    lines.append("## Memory JSON")
    lines.append("```json")
    lines.append(json.dumps(memory, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def refine_hypotheses(
    *,
    current_memory: Dict[str, Any],
    current_execution_result: Dict[str, Any],
    historical_memories: Sequence[Dict[str, Any]],
    model_id: str,
    layer_id: str,
    feature_id: str,
    top_m: int = 3,
    history_scope: HistoryScope = "same_hypothesis",
    timestamp: Optional[str] = None,
    round_id: Optional[str] = None,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    ts = timestamp or _clean_text(current_memory.get("timestamp")) or datetime.now().strftime("%Y%m%d_%H%M%S")
    round_index = _safe_int(current_memory.get("round_index"), 1)
    resolved_round_id = normalize_round_id(
        round_id or _clean_text(current_memory.get("round_id")) or None,
        round_index=round_index,
    )
    client = OpenAI(
        base_url=llm_base_url,
        api_key=read_api_key(llm_api_key_file),
    )

    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    refined_input = refine_hypotheses_for_side(
        side="input",
        current_memory=current_memory,
        current_execution_result=current_execution_result,
        historical_memories=historical_memories,
        top_m=top_m,
        history_scope=history_scope,
        client=client,
        llm_model=llm_model,
        token_counter=token_counter,
        llm_calls=llm_calls,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    refined_output = refine_hypotheses_for_side(
        side="output",
        current_memory=current_memory,
        current_execution_result=current_execution_result,
        historical_memories=historical_memories,
        top_m=top_m,
        history_scope=history_scope,
        client=client,
        llm_model=llm_model,
        token_counter=token_counter,
        llm_calls=llm_calls,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    result: Dict[str, Any] = {
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "round_id": resolved_round_id,
        "top_m": top_m,
        "history_rounds": len(historical_memories),
        "history_scope": history_scope,
        "llm_model": llm_model,
        "input_side_hypotheses": [item["refined_hypothesis"] for item in refined_input],
        "input_side_hypothesis_reasons": [item["refined_reason"] for item in refined_input],
        "output_side_hypotheses": [item["refined_hypothesis"] for item in refined_output],
        "output_side_hypothesis_reasons": [item["refined_reason"] for item in refined_output],
        "selected_hypothesis_indices": {
            "input": [item["hypothesis_index"] for item in refined_input],
            "output": [item["hypothesis_index"] for item in refined_output],
        },
        "refined_hypotheses": {
            "input": refined_input,
            "output": refined_output,
        },
        "token_usage": token_counter.as_dict(),
    }

    base_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=resolved_round_id,
        round_index=round_index,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-refined-hypotheses.json"
    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-refined-hypotheses.md"
    _write_refinement_markdown(result_md_path, result=result, llm_calls=llm_calls)
    return result


def _load_historical_memories(
    *,
    layer_id: str,
    feature_id: str,
    current_timestamp: str,
    current_round_id: Optional[str],
    history_rounds: int,
) -> List[Dict[str, Any]]:
    if history_rounds <= 0:
        return []

    feature_dir = Path("logs") / f"{layer_id}_{feature_id}"
    if not feature_dir.exists():
        return []

    current_round = normalize_round_id(current_round_id, round_index=1)
    memory_entries: List[Tuple[str, str, Path]] = []
    for ts_dir in feature_dir.iterdir():
        if not ts_dir.is_dir():
            continue

        # Preferred new layout: logs/{layer}_{feature}/{timestamp}/{round_id}/...
        for round_dir in ts_dir.iterdir():
            if not round_dir.is_dir():
                continue
            memory_path = round_dir / f"layer{layer_id}-feature{feature_id}-memory.json"
            if memory_path.exists():
                memory_entries.append((ts_dir.name, round_dir.name, memory_path))

        # Backward-compatible legacy layout.
        legacy_memory_path = ts_dir / f"layer{layer_id}-feature{feature_id}-memory.json"
        if legacy_memory_path.exists():
            memory_entries.append((ts_dir.name, "", legacy_memory_path))

    filtered = [
        item
        for item in memory_entries
        if not (item[0] == current_timestamp and item[1] == current_round)
    ]
    filtered.sort(key=lambda item: (item[0], item[1]))
    selected = filtered[-history_rounds:]

    histories: List[Dict[str, Any]] = []
    for _, _, memory_path in selected:
        payload = _load_json(memory_path)
        histories.append(payload)
    return histories


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6 of SAE workflow: refine hypotheses using memory and current experiment evidence.",
    )

    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id")
    parser.add_argument("--layer-id", required=True, help="Layer id")
    parser.add_argument("--feature-id", required=True, help="Feature id")
    parser.add_argument("--timestamp", default=None, help="Custom timestamp for logs/{layer}_{feature}/{timestamp}")
    parser.add_argument("--round-id", default=None, help="Round directory under timestamp, e.g. round_1")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, reuse existing logs/{layer}_{feature}/{timestamp}/{round_id} files for initial/experiments/execution/memory.",
    )
    parser.add_argument("--history-rounds", type=int, default=1, help="Use previous n rounds as historical memory.")
    parser.add_argument(
        "--history-scope",
        choices=["same_hypothesis", "all_hypotheses"],
        default="same_hypothesis",
        help="same_hypothesis: use only same index hypothesis history; all_hypotheses: use full side history.",
    )
    parser.add_argument("--top-m", type=int, default=3, help="Refine top-m hypotheses per side by score.")

    parser.add_argument("--memory-json-path", default=None, help="Optional direct path to memory JSON.")
    parser.add_argument("--execution-json-path", default=None, help="Optional direct path to experiments execution JSON.")
    parser.add_argument("--initial-hypotheses-json-path", default=None)
    parser.add_argument("--experiments-json-path", default=None)

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

    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2000)

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
    target_round_id = normalize_round_id(args.round_id, round_index=1)

    initial_result: Dict[str, Any]
    experiments_result: Dict[str, Any]
    execution_result: Dict[str, Any]
    memory: Dict[str, Any]

    if args.memory_json_path:
        memory_path = Path(args.memory_json_path)
        memory = _load_json(memory_path)
        layer_id = _clean_text(memory.get("layer_id") or args.layer_id)
        feature_id = _clean_text(memory.get("feature_id") or args.feature_id)
        ts = _clean_text(memory.get("timestamp") or ts)
        memory_round_index = _safe_int(memory.get("round_index"), 1)
        memory_round_id = normalize_round_id(
            _clean_text(memory.get("round_id")) or target_round_id,
            round_index=memory_round_index,
        )

        default_base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts / memory_round_id
        execution_path = (
            Path(args.execution_json_path)
            if args.execution_json_path
            else default_base_dir / f"layer{layer_id}-feature{feature_id}-experiments-execution.json"
        )
        execution_result = _load_json(execution_path)

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

        resolved_round_id = normalize_round_id(args.round_id, round_index=1)
        base_dir = Path("logs") / f"{args.layer_id}_{args.feature_id}" / ts / resolved_round_id
        initial_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-initial-hypotheses.json"
        experiments_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-experiments.json"
        execution_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-experiments-execution.json"
        memory_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-memory.json"

        initial_result = _load_json(initial_path)
        experiments_result = _load_json(experiments_path)
        execution_result = _load_json(execution_path)

        if memory_path.exists():
            memory = _load_json(memory_path)
        else:
            from hypothesis_memory import build_hypothesis_memory

            memory = build_hypothesis_memory(
                initial_hypotheses_result=initial_result,
                experiments_result=experiments_result,
                execution_result=execution_result,
                round_index=1,
                round_id=resolved_round_id,
            )
            memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        from experiments_execution import execute_hypothesis_experiments
        from experiments_design import design_hypothesis_experiments
        from hypothesis_memory import build_hypothesis_memory
        from initial_hypothesis_generation import generate_initial_hypotheses
        from model_with_sae import ModelWithSAEModule
        from neuronpedia_feature_api import fetch_and_parse_feature_observation

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
            round_id=target_round_id,
        )
        initial_result = generate_initial_hypotheses(
            observation=observation,
            model_id=args.model_id,
            layer_id=args.layer_id,
            feature_id=args.feature_id,
            num_hypothesis=args.num_hypothesis,
            generation_mode=args.generation_mode,
            timestamp=ts,
            round_id=target_round_id,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key_file=args.llm_api_key_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        experiments_result = design_hypothesis_experiments(
            hypotheses_result=initial_result,
            num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
            round_id=target_round_id,
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
            round_id=target_round_id,
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
        memory = build_hypothesis_memory(
            initial_hypotheses_result=initial_result,
            experiments_result=experiments_result,
            execution_result=execution_result,
            round_index=1,
            round_id=target_round_id,
        )

    layer_id = _clean_text(memory.get("layer_id") or args.layer_id)
    feature_id = _clean_text(memory.get("feature_id") or args.feature_id)
    ts = _clean_text(memory.get("timestamp") or ts)
    model_id = _clean_text(memory.get("model_id") or args.model_id)
    memory_round_index = _safe_int(memory.get("round_index"), 1)
    memory_round_id = normalize_round_id(
        _clean_text(memory.get("round_id")) or target_round_id,
        round_index=memory_round_index,
    )

    base_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=memory_round_id,
        round_index=memory_round_index,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    memory_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-memory.json"
    memory_json_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    memory_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-memory.md"
    _write_memory_markdown(memory_md_path, memory=memory)

    historical_memories = _load_historical_memories(
        layer_id=layer_id,
        feature_id=feature_id,
        current_timestamp=ts,
        current_round_id=memory_round_id,
        history_rounds=max(args.history_rounds, 0),
    )

    refinement_result = refine_hypotheses(
        current_memory=memory,
        current_execution_result=execution_result,
        historical_memories=historical_memories,
        model_id=model_id,
        layer_id=layer_id,
        feature_id=feature_id,
        top_m=args.top_m,
        history_scope=args.history_scope,
        timestamp=ts,
        round_id=memory_round_id,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key_file=args.llm_api_key_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(refinement_result, ensure_ascii=False, indent=2))
