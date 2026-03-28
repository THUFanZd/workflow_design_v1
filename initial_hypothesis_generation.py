from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from openai import OpenAI

from function import (
    TokenUsageAccumulator,
    build_feature_dir,
    build_round_dir,
    call_llm_stream,
    extract_json_object,
    normalize_round_id,
    read_api_key,
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

SideType = Literal["input", "output"]
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
        print('raw output:')
        print(raw_output)
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
    lines.append(f"- generation_mode: {result['generation_mode']}")
    lines.append(f"- llm_model: {result['llm_model']}")
    lines.append("")
    lines.append("## Token Usage (Full Initial Hypothesis Generation)")
    token_usage = result["token_usage"]
    lines.append(f"- prompt_tokens: {token_usage['prompt_tokens']}")
    lines.append(f"- completion_tokens: {token_usage['completion_tokens']}")
    lines.append(f"- total_tokens: {token_usage['total_tokens']}")
    lines.append("")
    lines.append("## Input-side Hypotheses")
    for idx, hyp in enumerate(result["input_side_hypotheses"], start=1):
        lines.append(f"{idx}. {hyp}")
    lines.append("")
    lines.append("## Output-side Hypotheses")
    for idx, hyp in enumerate(result["output_side_hypotheses"], start=1):
        lines.append(f"{idx}. {hyp}")
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
    timestamp: Optional[str] = None,
    round_id: Optional[str] = "round_0",
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_round_id = normalize_round_id(round_id, round_index=0)

    input_observation = _get_side_observation(observation, "input")
    output_observation = _get_side_observation(observation, "output")

    client = OpenAI(
        base_url=llm_base_url,
        api_key=read_api_key(llm_api_key_file),
    )

    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    input_hypotheses = _generate_hypotheses_for_side(
        client=client,
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
    output_hypotheses = _generate_hypotheses_for_side(
        client=client,
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
        "num_hypothesis": num_hypothesis,
        "generation_mode": generation_mode,
        "llm_model": llm_model,
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
    parser.add_argument("--width", default="16k", help="Neuronpedia source width")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Custom timestamp for logs/layer-{layer}/feature-{feature}/{timestamp}",
    )
    parser.add_argument("--round-id", default="round_0", help="Round directory under timestamp, e.g. round_0")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, read observation input from logs/layer-{layer}/feature-{feature}/{timestamp}/{round_id} instead of refetching from Neuronpedia.",
    )
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.reuse_from_logs:
        if args.timestamp is None:
            raise ValueError("When --reuse-from-logs is set, --timestamp is required.")
        observation_path = (
            build_feature_dir(layer_id=str(args.layer_id), feature_id=str(args.feature_id))
            / ts
            / args.round_id
            / f"layer{args.layer_id}-feature{args.feature_id}-observation-input.json"
        )
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
        timestamp=ts,
        round_id=args.round_id,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key_file=args.llm_api_key_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
