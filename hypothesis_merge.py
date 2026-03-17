from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from openai import OpenAI

from function import (
    TokenUsageAccumulator,
    build_round_dir,
    call_llm,
    extract_json_object,
    normalize_round_id,
    read_api_key,
)
from prompts.merge_prompt import build_system_prompt, build_user_prompt
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from support_info.llm_api_info import base_url as DEFAULT_BASE_URL
from support_info.llm_api_info import model_name as DEFAULT_MODEL_NAME

RunSideType = Literal["input", "output", "both"]
SideType = Literal["input", "output"]


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value]


def _round_index_from_round_id(round_id: Optional[str]) -> Optional[int]:
    text = _clean_text(round_id)
    if not text.startswith("round_"):
        return None
    idx = _safe_int(text[6:], 0)
    return idx if idx > 0 else None


def _identity_groups(
    *,
    hypotheses: Sequence[str],
    reasons: Sequence[str],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    merged_hypotheses: List[str] = []
    merged_reasons: List[str] = []
    groups: List[Dict[str, Any]] = []
    for i, hypothesis in enumerate(hypotheses, start=1):
        reason = reasons[i - 1] if i - 1 < len(reasons) else ""
        merged_hypotheses.append(hypothesis)
        merged_reasons.append(reason)
        groups.append(
            {
                "merged_index": i,
                "merged_hypothesis": hypothesis,
                "source_indices": [i],
                "source_hypotheses": [hypothesis],
                "source_reasons": [reason],
            }
        )
    return merged_hypotheses, merged_reasons, groups


def _validate_and_build_groups(
    *,
    parsed_output: Dict[str, Any],
    hypotheses: Sequence[str],
    reasons: Sequence[str],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    merged_raw = parsed_output.get("merged")
    if not isinstance(merged_raw, list) or not merged_raw:
        raise ValueError(f"Merge output must contain non-empty list field 'merged': {parsed_output}")

    total = len(hypotheses)
    seen: Set[int] = set()
    normalized_groups: List[Tuple[List[int], str]] = []

    for item in merged_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Each merged item must be a dict: {parsed_output}")
        hypothesis_text = _clean_text(item.get("hypothesis"))
        source_indices_raw = item.get("source_indices", [])
        if not isinstance(source_indices_raw, list) or not source_indices_raw:
            raise ValueError(f"Each merged item must contain non-empty source_indices: {parsed_output}")

        source_indices: List[int] = []
        local_seen: Set[int] = set()
        for raw in source_indices_raw:
            idx = _safe_int(raw, 0)
            if idx <= 0 or idx > total:
                raise ValueError(f"source_indices contains invalid index {raw}, total={total}")
            if idx in local_seen:
                raise ValueError(f"source_indices contains duplicated index {idx} in one group")
            local_seen.add(idx)
            source_indices.append(idx)

        for idx in source_indices:
            if idx in seen:
                raise ValueError(f"source_indices overlap on index {idx}")
            seen.add(idx)

        if not hypothesis_text:
            fallback_idx = min(source_indices) - 1
            hypothesis_text = hypotheses[fallback_idx]
        normalized_groups.append((sorted(source_indices), hypothesis_text))

    expected = set(range(1, total + 1))
    if seen != expected:
        missing = sorted(expected.difference(seen))
        raise ValueError(f"Merged groups must cover all original hypotheses exactly once, missing={missing}")

    normalized_groups.sort(key=lambda item: item[0][0])
    merged_hypotheses: List[str] = []
    merged_reasons: List[str] = []
    groups: List[Dict[str, Any]] = []
    for merged_index, (source_indices, merged_hypothesis) in enumerate(normalized_groups, start=1):
        source_hypotheses = [hypotheses[i - 1] for i in source_indices]
        source_reasons = [reasons[i - 1] if i - 1 < len(reasons) else "" for i in source_indices]
        merged_reason = ""
        for reason in source_reasons:
            if _clean_text(reason):
                merged_reason = _clean_text(reason)
                break

        merged_hypotheses.append(merged_hypothesis)
        merged_reasons.append(merged_reason)
        groups.append(
            {
                "merged_index": merged_index,
                "merged_hypothesis": merged_hypothesis,
                "source_indices": source_indices,
                "source_hypotheses": source_hypotheses,
                "source_reasons": source_reasons,
            }
        )

    return merged_hypotheses, merged_reasons, groups


def _merge_one_side(
    *,
    side: SideType,
    should_run: bool,
    hypotheses: Sequence[str],
    reasons: Sequence[str],
    client: OpenAI,
    llm_model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    hypotheses_list: List[str] = []
    reasons_list: List[str] = []
    for idx, raw_hypothesis in enumerate(hypotheses):
        hypothesis_text = _clean_text(raw_hypothesis)
        if not hypothesis_text:
            continue
        reason_text = _clean_text(reasons[idx]) if idx < len(reasons) else ""
        hypotheses_list.append(hypothesis_text)
        reasons_list.append(reason_text)

    if not should_run:
        merged_hypotheses, merged_reasons, groups = _identity_groups(
            hypotheses=hypotheses_list,
            reasons=reasons_list,
        )
        return {
            "side": side,
            "status": "skipped_side_disabled",
            "before_count": len(hypotheses_list),
            "after_count": len(merged_hypotheses),
            "groups": groups,
            "hypotheses": merged_hypotheses,
            "reasons": merged_reasons,
        }

    if len(hypotheses_list) <= 1:
        merged_hypotheses, merged_reasons, groups = _identity_groups(
            hypotheses=hypotheses_list,
            reasons=reasons_list,
        )
        return {
            "side": side,
            "status": "skipped_insufficient_candidates",
            "before_count": len(hypotheses_list),
            "after_count": len(merged_hypotheses),
            "groups": groups,
            "hypotheses": merged_hypotheses,
            "reasons": merged_reasons,
        }

    system_prompt = build_system_prompt(side)
    user_prompt = build_user_prompt(
        side=side,
        hypotheses=hypotheses_list,
        reasons=reasons_list,
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
    parsed_output = extract_json_object(raw_output)
    if not isinstance(parsed_output, dict):
        raise ValueError(f"Cannot parse JSON object from merge output: {raw_output}")

    merged_hypotheses, merged_reasons, groups = _validate_and_build_groups(
        parsed_output=parsed_output,
        hypotheses=hypotheses_list,
        reasons=reasons_list,
    )
    llm_calls.append(
        {
            "side": side,
            "messages": messages,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "usage": usage_counts,
            "response_debug": response_debug,
        }
    )
    return {
        "side": side,
        "status": "merged",
        "before_count": len(hypotheses_list),
        "after_count": len(merged_hypotheses),
        "groups": groups,
        "hypotheses": merged_hypotheses,
        "reasons": merged_reasons,
    }


def _write_merge_markdown(
    path: Path,
    *,
    result: Dict[str, Any],
    llm_calls: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# SAE Hypothesis Merge")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- model_id: {result.get('model_id', '')}")
    lines.append(f"- layer_id: {result.get('layer_id', '')}")
    lines.append(f"- feature_id: {result.get('feature_id', '')}")
    lines.append(f"- timestamp: {result.get('timestamp', '')}")
    lines.append(f"- round_id: {result.get('round_id', '')}")
    lines.append(f"- run_side: {result.get('run_side', '')}")
    lines.append(f"- llm_model: {result.get('llm_model', '')}")
    lines.append("")
    lines.append("## Token Usage (Hypothesis Merge)")
    usage = result.get("token_usage", {})
    lines.append(f"- prompt_tokens: {_safe_int(usage.get('prompt_tokens'), 0)}")
    lines.append(f"- completion_tokens: {_safe_int(usage.get('completion_tokens'), 0)}")
    lines.append(f"- total_tokens: {_safe_int(usage.get('total_tokens'), 0)}")
    lines.append("")

    before = result.get("before_merge", {})
    details = result.get("merge_details", {})
    for side in ("input", "output"):
        side_detail = details.get(side, {}) if isinstance(details, dict) else {}
        lines.append(f"## {side.capitalize()} Side")
        lines.append(f"- status: {_clean_text(side_detail.get('status'))}")
        lines.append(f"- before_count: {_safe_int(side_detail.get('before_count'), 0)}")
        lines.append(f"- after_count: {_safe_int(side_detail.get('after_count'), 0)}")
        lines.append("")

        lines.append("### Before Merge")
        for idx, hypothesis in enumerate(before.get(f"{side}_side_hypotheses", []), start=1):
            lines.append(f"{idx}. {_clean_text(hypothesis)}")
        lines.append("")

        lines.append("### After Merge")
        for idx, hypothesis in enumerate(result.get(f"{side}_side_hypotheses", []), start=1):
            lines.append(f"{idx}. {_clean_text(hypothesis)}")
        lines.append("")

        lines.append("### Merge Groups")
        groups = side_detail.get("groups", [])
        if isinstance(groups, list) and groups:
            for group in groups:
                if not isinstance(group, dict):
                    continue
                lines.append(
                    f"- merged_index={_safe_int(group.get('merged_index'), 0)} "
                    f"source_indices={group.get('source_indices', [])}"
                )
                lines.append(f"  merged_hypothesis: {_clean_text(group.get('merged_hypothesis'))}")
        else:
            lines.append("- (none)")
        lines.append("")

    lines.append("## LLM API Calls")
    for call_index, call in enumerate(llm_calls, start=1):
        lines.append(f"### Call {call_index}")
        lines.append(f"- side: {_clean_text(call.get('side'))}")
        usage_obj = call.get("usage", {})
        lines.append(f"- prompt_tokens: {_safe_int(usage_obj.get('prompt_tokens'), 0)}")
        lines.append(f"- completion_tokens: {_safe_int(usage_obj.get('completion_tokens'), 0)}")
        lines.append(f"- total_tokens: {_safe_int(usage_obj.get('total_tokens'), 0)}")
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


def merge_refined_hypotheses(
    *,
    refined_hypotheses_result: Dict[str, Any],
    model_id: str,
    layer_id: str,
    feature_id: str,
    run_side: RunSideType = "both",
    timestamp: Optional[str] = None,
    round_id: Optional[str] = None,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    temperature: float = 0.0,
    max_tokens: int = 20000,
) -> Dict[str, Any]:
    ts = timestamp or _clean_text(refined_hypotheses_result.get("timestamp")) or datetime.now().strftime("%Y%m%d_%H%M%S")
    preferred_round_id = round_id or _clean_text(refined_hypotheses_result.get("round_id")) or None
    resolved_round_id = normalize_round_id(
        preferred_round_id,
        round_index=_round_index_from_round_id(preferred_round_id),
    )

    run_input = run_side in ("input", "both")
    run_output = run_side in ("output", "both")

    input_hypotheses = _normalize_str_list(refined_hypotheses_result.get("input_side_hypotheses", []))
    input_reasons = _normalize_str_list(refined_hypotheses_result.get("input_side_hypothesis_reasons", []))
    output_hypotheses = _normalize_str_list(refined_hypotheses_result.get("output_side_hypotheses", []))
    output_reasons = _normalize_str_list(refined_hypotheses_result.get("output_side_hypothesis_reasons", []))

    client = OpenAI(
        base_url=llm_base_url,
        api_key=read_api_key(llm_api_key_file),
    )
    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    input_detail = _merge_one_side(
        side="input",
        should_run=run_input,
        hypotheses=input_hypotheses,
        reasons=input_reasons,
        client=client,
        llm_model=llm_model,
        token_counter=token_counter,
        llm_calls=llm_calls,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_detail = _merge_one_side(
        side="output",
        should_run=run_output,
        hypotheses=output_hypotheses,
        reasons=output_reasons,
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
        "run_side": run_side,
        "llm_model": llm_model,
        "before_merge": {
            "input_side_hypotheses": input_hypotheses,
            "input_side_hypothesis_reasons": input_reasons,
            "output_side_hypotheses": output_hypotheses,
            "output_side_hypothesis_reasons": output_reasons,
        },
        "input_side_hypotheses": list(input_detail.get("hypotheses", [])),
        "input_side_hypothesis_reasons": list(input_detail.get("reasons", [])),
        "output_side_hypotheses": list(output_detail.get("hypotheses", [])),
        "output_side_hypothesis_reasons": list(output_detail.get("reasons", [])),
        "merge_details": {
            "input": {
                "status": input_detail.get("status"),
                "before_count": input_detail.get("before_count"),
                "after_count": input_detail.get("after_count"),
                "groups": input_detail.get("groups", []),
            },
            "output": {
                "status": output_detail.get("status"),
                "before_count": output_detail.get("before_count"),
                "after_count": output_detail.get("after_count"),
                "groups": output_detail.get("groups", []),
            },
        },
        "token_usage": token_counter.as_dict(),
    }

    base_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=resolved_round_id,
        round_index=_round_index_from_round_id(resolved_round_id),
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    json_path = base_dir / f"layer{layer_id}-feature{feature_id}-merged-hypotheses.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = base_dir / f"layer{layer_id}-feature{feature_id}-merged-hypotheses.md"
    _write_merge_markdown(md_path, result=result, llm_calls=llm_calls)
    return result
