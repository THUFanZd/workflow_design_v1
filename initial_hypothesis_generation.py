from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from openai import OpenAI

from function import (
    DEFAULT_MAX_TOKENS,
    TokenUsageAccumulator,
    build_feature_dir,
    build_round_dir,
    call_llm_stream,
    extract_json_object,
    normalize_round_id,
    read_api_key,
    resolve_existing_round_dir,
)
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from support_info.llm_api_info import base_url as DEFAULT_BASE_URL
from support_info.llm_api_info import model_name as DEFAULT_MODEL_NAME
from neuronpedia_feature_api import fetch_and_parse_feature_observation
from prompts.hypothesis_generation_prompt import (
    build_iterative_user_prompt,
    build_single_call_user_prompt,
    build_system_prompt,
)
from prompts.bos_token_semantic_cluster_prompt import (
    build_bos_token_semantic_cluster_system_prompt,
    build_bos_token_semantic_cluster_user_prompt,
)

SideType = Literal["input", "output"]
RunSideType = Literal["input", "output", "both"]
GenerationMode = Literal["single_call", "iterative"]


def _normalize_hypothesis(text: str) -> str:
    stripped = text.strip().strip('"').strip("'")
    stripped = re.sub(r"^\d+[\).\s-]+", "", stripped)
    return " ".join(stripped.split())


def _parse_hypothesis_list(raw_output: str, expected_count: int) -> List[str]:
    parsed = extract_json_object(raw_output)
    hypotheses: List[str] = []

    if isinstance(parsed, dict):
        candidate = parsed.get("hypotheses")
        if isinstance(candidate, list):
            hypotheses = [
                _normalize_hypothesis(item)
                for item in candidate
                if isinstance(item, str) and _normalize_hypothesis(item)
            ]

    if not hypotheses:
        lines = [line for line in raw_output.splitlines() if line.strip()]
        hypotheses = [_normalize_hypothesis(line) for line in lines if _normalize_hypothesis(line)]

    if not hypotheses:
        raise ValueError(f"Failed to parse hypotheses from output: {raw_output}")

    return hypotheses[:expected_count]


def _parse_single_hypothesis(raw_output: str) -> str:
    parsed = extract_json_object(raw_output)
    if isinstance(parsed, dict):
        for key in ("hypothesis", "output", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                normalized = _normalize_hypothesis(value)
                if normalized:
                    return normalized

    lines = [line for line in raw_output.splitlines() if line.strip()]
    for line in lines:
        normalized = _normalize_hypothesis(line)
        if normalized:
            return normalized
    raise ValueError(f"Failed to parse one hypothesis from output: {raw_output}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _get_side_observation(observation_dict: Dict[str, Any], side: SideType) -> Dict[str, Any]:
    if side == "input":
        keys = ("input_side_observation", "input_side_obseravtion")
    else:
        keys = ("output_side_observation", "output_side_obseravtion")

    for key in keys:
        value = observation_dict.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError(f"Cannot find {side} observation in observation dict.")


def _is_bos_token_side_observation(side_observation: Dict[str, Any]) -> bool:
    source_raw = side_observation.get("source")
    source = str(source_raw).strip().lower() if source_raw is not None else ""
    if source == "bos_token":
        return True
    return "bos_token_top_tokens" in side_observation or "bos_token_scan_meta" in side_observation


def _normalize_cluster_token(token: str) -> str:
    cleaned = str(token).strip()
    cleaned = re.sub(r"^[Ġ▁]+", "", cleaned)
    return cleaned.strip()


def _normalize_alpha_token(token: str) -> str:
    token_lower = _normalize_cluster_token(token).lower()
    return re.sub(r"[^a-z]+", "", token_lower)


def _simple_lemma(alpha_token: str) -> str:
    text = alpha_token
    if len(text) > 4 and text.endswith("ies"):
        return text[:-3] + "y"
    if len(text) > 5 and text.endswith("ing"):
        base = text[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(text) > 4 and text.endswith("ed"):
        base = text[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(text) > 4 and text.endswith("es"):
        if text.endswith(("ses", "xes", "zes", "ches", "shes")):
            return text[:-2]
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def _extract_bos_token_entries(side_observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    token_to_activation: Dict[str, float] = {}

    top_tokens_raw = side_observation.get("bos_token_top_tokens", [])
    top_tokens = [item for item in top_tokens_raw if isinstance(item, dict)] if isinstance(top_tokens_raw, list) else []
    for item in top_tokens:
        token_raw = item.get("token")
        token = _normalize_cluster_token(token_raw if token_raw is not None else "")
        if not token:
            continue
        activation = _safe_float(item.get("activation"), 0.0)
        previous = token_to_activation.get(token)
        if previous is None or activation > previous:
            token_to_activation[token] = activation

    if not token_to_activation:
        activation_examples_raw = side_observation.get("activation_examples", [])
        activation_examples = (
            [item for item in activation_examples_raw if isinstance(item, dict)]
            if isinstance(activation_examples_raw, list)
            else []
        )
        for example in activation_examples:
            activation_tokens_raw = example.get("activation_tokens", [])
            activation_tokens = (
                [item for item in activation_tokens_raw if isinstance(item, dict)]
                if isinstance(activation_tokens_raw, list)
                else []
            )
            for activation_item in activation_tokens:
                token_raw = activation_item.get("token")
                token = _normalize_cluster_token(token_raw if token_raw is not None else "")
                if not token:
                    continue
                activation = _safe_float(activation_item.get("value"), 0.0)
                previous = token_to_activation.get(token)
                if previous is None or activation > previous:
                    token_to_activation[token] = activation

    sorted_entries = sorted(
        [{"token": token, "activation": activation} for token, activation in token_to_activation.items()],
        key=lambda item: (-_safe_float(item.get("activation"), 0.0), str(item.get("token", "")).lower()),
    )
    return sorted_entries


def _select_half_max_entries(entries: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float]:
    if not entries:
        return [], 0.0, 0.0
    max_activation = max(_safe_float(item.get("activation"), 0.0) for item in entries)
    threshold = 0.5 * max_activation
    selected = [item for item in entries if _safe_float(item.get("activation"), 0.0) > threshold]
    if not selected:
        best = max(entries, key=lambda item: _safe_float(item.get("activation"), 0.0))
        selected = [best]
    return selected, max_activation, threshold


def _cluster_entries_by_morphology(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    cluster_index_by_key: Dict[str, int] = {}

    for item in entries:
        token = _normalize_cluster_token(item.get("token", ""))
        if not token:
            continue
        activation = _safe_float(item.get("activation"), 0.0)
        alpha_token = _normalize_alpha_token(token)
        if alpha_token:
            key = f"lemma:{_simple_lemma(alpha_token)}"
        else:
            key = f"raw:{token.lower()}"

        if key not in cluster_index_by_key:
            cluster_index_by_key[key] = len(clusters)
            clusters.append({"tokens": [token], "max_activation": activation})
            continue

        cluster = clusters[cluster_index_by_key[key]]
        cluster_tokens = cluster.get("tokens", [])
        if isinstance(cluster_tokens, list) and token not in cluster_tokens:
            cluster_tokens.append(token)
        cluster["max_activation"] = max(_safe_float(cluster.get("max_activation"), 0.0), activation)
    return clusters


def _validate_and_merge_cluster_ids(
    *,
    cluster_count: int,
    raw_output: str,
) -> List[List[int]]:
    parsed = extract_json_object(raw_output)
    valid_ids = set(range(1, cluster_count + 1))
    used_ids: set[int] = set()
    merged_id_groups: List[List[int]] = []

    if isinstance(parsed, dict):
        merged_raw = parsed.get("merged_clusters")
        merged_clusters = [item for item in merged_raw if isinstance(item, dict)] if isinstance(merged_raw, list) else []
        for item in merged_clusters:
            ids_raw = item.get("cluster_ids")
            if not isinstance(ids_raw, list):
                continue
            ids: List[int] = []
            for value in ids_raw:
                try:
                    cluster_id = int(value)
                except (TypeError, ValueError):
                    continue
                if cluster_id not in valid_ids or cluster_id in used_ids or cluster_id in ids:
                    continue
                ids.append(cluster_id)
            if ids:
                merged_id_groups.append(ids)
                used_ids.update(ids)

    if not merged_id_groups:
        merged_id_groups = [[idx] for idx in range(1, cluster_count + 1)]
        used_ids = set(range(1, cluster_count + 1))

    for missing_id in range(1, cluster_count + 1):
        if missing_id not in used_ids:
            merged_id_groups.append([missing_id])
    return merged_id_groups


def _semantic_merge_morph_clusters(
    *,
    morph_clusters: Sequence[Dict[str, Any]],
    max_clusters: int,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    if len(morph_clusters) <= 1:
        return list(morph_clusters)

    cluster_payload = [
        {
            "cluster_id": index + 1,
            "tokens": list(cluster.get("tokens", [])),
            "max_activation": _safe_float(cluster.get("max_activation"), 0.0),
        }
        for index, cluster in enumerate(morph_clusters)
    ]
    system_prompt = build_bos_token_semantic_cluster_system_prompt()
    user_prompt = build_bos_token_semantic_cluster_user_prompt(
        clusters=cluster_payload,
        max_clusters=max_clusters,
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
    llm_calls.append(
        {
            "side": "input",
            "mode": "bos_token_semantic_cluster",
            "round": 1,
            "messages": messages,
            "raw_output": raw_output,
            "usage": usage_counts,
        }
    )

    merged_id_groups = _validate_and_merge_cluster_ids(
        cluster_count=len(morph_clusters),
        raw_output=raw_output,
    )

    merged_clusters: List[Dict[str, Any]] = []
    for id_group in merged_id_groups:
        tokens: List[str] = []
        max_activation = 0.0
        for cluster_id in id_group:
            source = morph_clusters[cluster_id - 1]
            source_tokens = source.get("tokens", [])
            if isinstance(source_tokens, list):
                for token in source_tokens:
                    token_text = _normalize_cluster_token(token)
                    if token_text and token_text not in tokens:
                        tokens.append(token_text)
            max_activation = max(max_activation, _safe_float(source.get("max_activation"), 0.0))
        if tokens:
            merged_clusters.append({"tokens": tokens, "max_activation": max_activation})
    return merged_clusters


def _cap_clusters(
    *,
    clusters: Sequence[Dict[str, Any]],
    max_count: int,
) -> List[Dict[str, Any]]:
    if max_count <= 0:
        return []
    sorted_clusters = sorted(
        [dict(cluster) for cluster in clusters if isinstance(cluster, dict)],
        key=lambda cluster: (
            -_safe_float(cluster.get("max_activation"), 0.0),
            str((cluster.get("tokens") or [""])[0]).lower(),
        ),
    )
    return sorted_clusters[:max_count]


def _build_bos_token_clustered_hypotheses(
    *,
    side_observation: Dict[str, Any],
    num_hypothesis: int,
    enable_semantic_cluster: bool,
    client: Optional[OpenAI],
    model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> List[str]:
    entries = _extract_bos_token_entries(side_observation)
    selected_entries, _, _ = _select_half_max_entries(entries)
    if not selected_entries:
        return []

    morph_clusters = _cluster_entries_by_morphology(selected_entries)
    final_clusters = morph_clusters
    if enable_semantic_cluster and client is not None:
        final_clusters = _semantic_merge_morph_clusters(
            morph_clusters=morph_clusters,
            max_clusters=max(1, num_hypothesis),
            client=client,
            model=model,
            token_counter=token_counter,
            llm_calls=llm_calls,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if not final_clusters:
        final_clusters = morph_clusters

    bounded_clusters = _cap_clusters(
        clusters=final_clusters,
        max_count=max(1, num_hypothesis),
    )
    hypotheses: List[str] = []
    for cluster in bounded_clusters:
        tokens = [str(item).strip() for item in cluster.get("tokens", []) if str(item).strip()]
        if tokens:
            noun = "token" if len(tokens) == 1 else "tokens"
            hypotheses.append(f"Activate on {noun} {', '.join(tokens)}")
    return hypotheses


def _generate_hypotheses_for_side(
    client: OpenAI,
    model: str,
    side: SideType,
    side_observation: Dict[str, Any],
    num_hypothesis: int,
    generation_mode: GenerationMode,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> List[str]:
    if num_hypothesis <= 0:
        raise ValueError("num_hypothesis must be a positive integer.")

    system_prompt = build_system_prompt(side)
    hypotheses: List[str] = []

    if generation_mode == "single_call":
        user_prompt = build_single_call_user_prompt(
            side=side,
            observation=side_observation,
            num_hypothesis=num_hypothesis,
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
        hypotheses = _parse_hypothesis_list(raw_output, expected_count=num_hypothesis)

        llm_calls.append(
            {
                "side": side,
                "mode": generation_mode,
                "round": 1,
                "messages": messages,
                "raw_output": raw_output,
                "usage": usage_counts,
            }
        )
        return hypotheses

    if generation_mode == "iterative":
        for idx in range(1, num_hypothesis + 1):
            user_prompt = build_iterative_user_prompt(
                side=side,
                observation=side_observation,
                existing_hypotheses=hypotheses,
                current_index=idx,
                total_count=num_hypothesis,
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
            one_hypothesis = _parse_single_hypothesis(raw_output)
            hypotheses.append(one_hypothesis)
            llm_calls.append(
                {
                    "side": side,
                    "mode": generation_mode,
                    "round": idx,
                    "messages": messages,
                    "raw_output": raw_output,
                    "usage": usage_counts,
                }
            )
        return hypotheses

    raise ValueError(f"Unsupported generation mode: {generation_mode}")


def _write_markdown_log(
    path: Path,
    *,
    result: Dict[str, Any],
    llm_calls: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# SAE Initial Hypothesis Generation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- layer_id: {result['layer_id']}")
    lines.append(f"- feature_id: {result['feature_id']}")
    lines.append(f"- timestamp: {result['timestamp']}")
    if "round_id" in result:
        lines.append(f"- round_id: {result['round_id']}")
    lines.append(f"- num_hypothesis: {result['num_hypothesis']}")
    lines.append(f"- run_side: {result.get('run_side', 'both')}")
    lines.append(f"- generation_mode: {result['generation_mode']}")
    lines.append(f"- llm_model: {result['llm_model']}")
    lines.append(f"- enable_bos_token_semantic_cluster: {bool(result.get('enable_bos_token_semantic_cluster', False))}")
    lines.append("")
    lines.append("## Token Usage (Full Initial Hypothesis Generation)")
    token_usage = result["token_usage"]
    lines.append(f"- prompt_tokens: {token_usage['prompt_tokens']}")
    lines.append(f"- completion_tokens: {token_usage['completion_tokens']}")
    lines.append(f"- total_tokens: {token_usage['total_tokens']}")
    lines.append("")
    run_side = str(result.get("run_side", "both"))
    if run_side in ("input", "both"):
        lines.append("## Input-side Hypotheses")
        for idx, hyp in enumerate(result["input_side_hypotheses"], start=1):
            lines.append(f"{idx}. {hyp}")
        if not result["input_side_hypotheses"]:
            lines.append("- (none)")
        lines.append("")
    if run_side in ("output", "both"):
        lines.append("## Output-side Hypotheses")
        for idx, hyp in enumerate(result["output_side_hypotheses"], start=1):
            lines.append(f"{idx}. {hyp}")
        if not result["output_side_hypotheses"]:
            lines.append("- (none)")
        lines.append("")
    lines.append("## LLM Calls")
    for i, call in enumerate(llm_calls, start=1):
        lines.append(f"### Call {i}")
        lines.append(f"- side: {call['side']}")
        lines.append(f"- mode: {call['mode']}")
        lines.append(f"- round: {call['round']}")
        usage = call.get("usage", {})
        lines.append(f"- prompt_tokens: {usage.get('prompt_tokens', 0)}")
        lines.append(f"- completion_tokens: {usage.get('completion_tokens', 0)}")
        lines.append(f"- total_tokens: {usage.get('total_tokens', 0)}")
        lines.append("")
        lines.append("#### Messages")
        lines.append("```json")
        lines.append(json.dumps(call["messages"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("#### Raw Output")
        lines.append("```text")
        lines.append(call.get("raw_output", ""))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_initial_hypotheses(
    *,
    observation: Dict[str, Any],
    model_id: str,
    layer_id: str,
    feature_id: str,
    num_hypothesis: int,
    generation_mode: GenerationMode,
    run_side: RunSideType = "both",
    timestamp: Optional[str] = None,
    round_id: Optional[str] = "round_0",
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    enable_bos_token_semantic_cluster: bool = False,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_round_id = normalize_round_id(round_id, round_index=0)
    if run_side not in ("input", "output", "both"):
        raise ValueError(f"Unsupported run_side: {run_side}")

    client: Optional[OpenAI] = None

    def _ensure_client() -> OpenAI:
        nonlocal client
        if client is None:
            client = OpenAI(
                base_url=llm_base_url,
                api_key=read_api_key(llm_api_key_file),
            )
        return client

    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    input_hypotheses: List[str] = []
    output_hypotheses: List[str] = []
    if run_side in ("input", "both"):
        input_observation = _get_side_observation(observation, "input")
        if _is_bos_token_side_observation(input_observation):
            input_hypotheses = _build_bos_token_clustered_hypotheses(
                side_observation=input_observation,
                num_hypothesis=num_hypothesis,
                enable_semantic_cluster=enable_bos_token_semantic_cluster,
                client=_ensure_client() if enable_bos_token_semantic_cluster else None,
                model=llm_model,
                token_counter=token_counter,
                llm_calls=llm_calls,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            input_hypotheses = _generate_hypotheses_for_side(
                client=_ensure_client(),
                model=llm_model,
                side="input",
                side_observation=input_observation,
                num_hypothesis=num_hypothesis,
                generation_mode=generation_mode,
                token_counter=token_counter,
                llm_calls=llm_calls,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    if run_side in ("output", "both"):
        output_observation = _get_side_observation(observation, "output")
        if _is_bos_token_side_observation(output_observation):
            output_hypotheses = []
        else:
            output_hypotheses = _generate_hypotheses_for_side(
                client=_ensure_client(),
                model=llm_model,
                side="output",
                side_observation=output_observation,
                num_hypothesis=num_hypothesis,
                generation_mode=generation_mode,
                token_counter=token_counter,
                llm_calls=llm_calls,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    base_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=resolved_round_id,
        round_index=0,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "round_id": resolved_round_id,
        "run_side": run_side,
        "num_hypothesis": num_hypothesis,
        "generation_mode": generation_mode,
        "llm_model": llm_model,
        "enable_bos_token_semantic_cluster": bool(enable_bos_token_semantic_cluster),
        "input_side_hypotheses": input_hypotheses,
        "output_side_hypotheses": output_hypotheses,
        "token_usage": token_counter.as_dict(),
    }

    result_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-initial-hypotheses.json"
    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-initial-hypotheses.md"
    _write_markdown_log(result_md_path, result=result, llm_calls=llm_calls)

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2 of SAE workflow: generate initial input/output hypotheses from Neuronpedia observations.",
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id")
    parser.add_argument("--layer-id", required=True, help="Layer id")
    parser.add_argument("--feature-id", required=True, help="Feature id")
    parser.add_argument("--num-hypothesis", type=int, default=3, help="Hypothesis count n for each side")
    parser.add_argument(
        "--generation-mode",
        choices=["single_call", "iterative"],
        default="single_call",
        help="single_call: one call outputs n hypotheses; iterative: n calls output n hypotheses",
    )
    parser.add_argument(
        "--side",
        choices=["input", "output", "both"],
        default="both",
        help="Generate initial hypotheses for input/output side or both.",
    )
    parser.add_argument("--width", default="16k", help="Neuronpedia source width")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Custom timestamp for logs/layer-{layer_id}/feature-{feature_id}/{timestamp}",
    )
    parser.add_argument("--round-id", default="round_0", help="Round directory under timestamp, e.g. round_0")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, read observation input from logs artifacts instead of refetching from Neuronpedia.",
    )
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument(
        "--enable-bos-token-semantic-cluster",
        action="store_true",
        help="If set, run semantic clustering after morphology clustering for bos_token input hypotheses.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.reuse_from_logs:
        if args.timestamp is None:
            raise ValueError("When --reuse-from-logs is set, --timestamp is required.")
        base_dir = resolve_existing_round_dir(
            layer_id=str(args.layer_id),
            feature_id=str(args.feature_id),
            timestamp=ts,
            round_id=args.round_id,
            round_index=0,
        )
        if base_dir is None:
            raise FileNotFoundError(
                f"Cannot find round directory under logs for layer={args.layer_id}, "
                f"feature={args.feature_id}, timestamp={ts}, round_id={args.round_id}"
            )
        observation_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-observation-input.json"
        if not observation_path.exists():
            raise FileNotFoundError(f"Cannot find observation input file: {observation_path}")
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
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
            round_id=args.round_id,
        )

    result = generate_initial_hypotheses(
        observation=observation,
        model_id=args.model_id,
        layer_id=args.layer_id,
        feature_id=args.feature_id,
        num_hypothesis=args.num_hypothesis,
        generation_mode=args.generation_mode,
        run_side=args.side,
        timestamp=ts,
        round_id=args.round_id,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key_file=args.llm_api_key_file,
        enable_bos_token_semantic_cluster=bool(args.enable_bos_token_semantic_cluster),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
