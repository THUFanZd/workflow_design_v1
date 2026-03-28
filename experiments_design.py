from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from openai import OpenAI

from function import (
    DEFAULT_MAX_TOKENS,
    TokenUsageAccumulator,
    build_round_dir,
    call_llm_stream,
    extract_json_object,
    normalize_round_id,
    read_api_key,
    resolve_existing_round_dir,
)
from initial_hypothesis_generation import generate_initial_hypotheses
from neuronpedia_feature_api import fetch_and_parse_feature_observation
from prompts.experiments_design_prompt import InputTestType, build_system_prompt, build_user_prompt
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from support_info.llm_api_info import base_url as DEFAULT_BASE_URL
from support_info.llm_api_info import model_name as DEFAULT_MODEL_NAME

SideType = Literal["input", "output"]
RunSideType = Literal["input", "output", "both"]
OUTPUT_SIDE_PLACEHOLDER = ["The explanation is simple:", "I think", "We"]


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

    if len(sentences) < expected_count:
        raise ValueError(
            f"Expected {expected_count} sentences, but parsed {len(sentences)} from output: {raw_output}"
        )
    return sentences[:expected_count]


def _extract_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return [normalized] if normalized else []
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            items.extend(_extract_string_list(item))
        return items
    if isinstance(value, dict):
        items = []
        for item in value.values():
            items.extend(_extract_string_list(item))
        return items
    return []


def generate_boundary_contexts(
    *,
    client: OpenAI,
    model: str,
    explanation: str,
    boundary_case_count: int = 5,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    token_counter: Optional[TokenUsageAccumulator] = None,
    llm_calls: Optional[List[Dict[str, Any]]] = None,
    call_metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if boundary_case_count <= 0:
        raise ValueError("boundary_case_count must be a positive integer.")

    system_prompt = (
        "You are an expert at designing adversarial boundary test cases for SAE feature explanations. "
        "Return JSON only."
    )
    user_prompt = (
        "Task: generate boundary contexts for the hypothesis below.\n\n"
        "Definition of boundary case (critical):\n"
        "- A boundary case is near the edge of the explained set by the feature explanation:\n"
        "  it looks lexically or semantically similar,\n"
        "  but should still fall outside the true activation set.\n"
        "- The case should be tempting and confusable, but the feature should not activate strongly on it.\n"
        "- Use multiple near-miss types when possible, such as context shift, minimal lexical edits,\n"
        "  orthographic variants, or the same surface form in a different domain.\n\n"
        f"Hypothesis / explanation:\n{explanation}\n\n"
        f"Generate exactly {boundary_case_count} boundary contexts.\n"
        "Each context should be a single natural sentence or short snippet.\n"
        "Return JSON only in this format:\n"
        "{\n"
        '  "boundary_cases": ["case 1", "case 2", "case 3"]\n'
        "}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw_output, usage_obj = call_llm_stream(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    usage_counts: Optional[Dict[str, int]] = None
    if token_counter is not None:
        usage_counts = token_counter.add(usage_obj)

    parsed = extract_json_object(raw_output)
    candidates: List[str] = []
    if isinstance(parsed, dict):
        for key in ("boundary_cases", "cases", "examples", "contexts", "items"):
            if key in parsed:
                candidates.extend(_extract_string_list(parsed[key]))
        if not candidates:
            candidates.extend(_extract_string_list(parsed))

    if not candidates:
        for line in raw_output.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                candidates.append(" ".join(cleaned.split()))

    deduped: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)

    if llm_calls is not None:
        payload: Dict[str, Any] = dict(call_metadata or {})
        payload.update(
            {
                "messages": messages,
                "raw_output": raw_output,
            }
        )
        if usage_counts is not None:
            payload["usage"] = usage_counts
        llm_calls.append(payload)

    if len(deduped) < boundary_case_count:
        raise ValueError(
            f"Boundary case generator returned {len(deduped)} cases, "
            f"but {boundary_case_count} are required. Raw output: {raw_output}"
        )
    return deduped[:boundary_case_count]


def _design_sentences(
    *,
    side: SideType,
    hypotheses: Sequence[str],
    num_sentences: int,
    client: OpenAI,
    model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    input_test_mode: InputTestType = "activation",
    previous_input_hypotheses: Optional[Sequence[str]] = None,
) -> List[List[str]]:
    if side == "output":
        return [list(OUTPUT_SIDE_PLACEHOLDER) for _ in hypotheses]

    all_sentences: List[List[str]] = []
    system_prompt = build_system_prompt(side, input_test_type=input_test_mode)

    for index, hypothesis in enumerate(hypotheses, start=1):
        previous_hypothesis = ""
        if side == "input" and input_test_mode == "expansion" and previous_input_hypotheses is not None:
            if 0 <= index - 1 < len(previous_input_hypotheses):
                previous_hypothesis = str(previous_input_hypotheses[index - 1]).strip()

        user_prompt = build_user_prompt(
            side=side,
            hypothesis=hypothesis,
            num_sentences=num_sentences,
            input_test_type=input_test_mode,
            previous_hypothesis=previous_hypothesis,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_output, usage_obj = call_llm_stream(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage_counts = token_counter.add(usage_obj)
        designed_sentences = _parse_sentence_list(raw_output, expected_count=num_sentences)
        all_sentences.append(designed_sentences)

        llm_calls.append(
            {
                "side": side,
                "input_test_mode": input_test_mode if side == "input" else None,
                "hypothesis_index": index,
                "hypothesis_text": hypothesis,
                "previous_hypothesis": previous_hypothesis,
                "messages": messages,
                "raw_output": raw_output,
                "usage": usage_counts,
            }
        )

    return all_sentences


def _write_markdown_log(
    path: Path,
    *,
    result: Dict[str, Any],
    llm_calls: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# SAE Hypothesis Experiments Generation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- model_id: {result['model_id']}")
    lines.append(f"- layer_id: {result['layer_id']}")
    lines.append(f"- feature_id: {result['feature_id']}")
    lines.append(f"- timestamp: {result['timestamp']}")
    if "round_id" in result:
        lines.append(f"- round_id: {result['round_id']}")
    lines.append(f"- num_hypothesis: {result['num_hypothesis']}")
    lines.append(f"- num_input_sentences_per_hypothesis: {result['num_input_sentences_per_hypothesis']}")
    lines.append(f"- input_test_mode: {result.get('input_test_mode')}")
    lines.append(f"- llm_model: {result['llm_model']}")
    lines.append("")
    lines.append("## Token Usage (Experiments Generation)")
    token_usage = result["token_usage"]
    lines.append(f"- prompt_tokens: {token_usage['prompt_tokens']}")
    lines.append(f"- completion_tokens: {token_usage['completion_tokens']}")
    lines.append(f"- total_tokens: {token_usage['total_tokens']}")
    lines.append("")
    lines.append("## Input-side Hypotheses And Designed Sentences")
    for idx, pair in enumerate(result["input_side_experiments"], start=1):
        lines.append(f"### Input Hypothesis {idx}")
        lines.append(f"- hypothesis: {pair['hypothesis']}")
        lines.append(f"- test_type: {pair.get('test_type', 'activation')}")
        lines.append(f"- reference_hypothesis: {pair.get('reference_hypothesis', '')}")
        lines.append("- designed_sentences:")
        for sentence in pair.get("designed_sentences", []):
            lines.append(f"  - {sentence}")
        lines.append("")
    lines.append("## Output-side Hypotheses And Designed Sentences")
    for idx, pair in enumerate(result["output_side_experiments"], start=1):
        lines.append(f"### Output Hypothesis {idx}")
        lines.append(f"- hypothesis: {pair['hypothesis']}")
        lines.append("- designed_sentences:")
        for sentence in pair.get("designed_sentences", []):
            lines.append(f"  - {sentence}")
        lines.append("")
    lines.append("## LLM Calls")
    for i, call in enumerate(llm_calls, start=1):
        lines.append(f"### Call {i}")
        lines.append(f"- side: {call.get('side')}")
        lines.append(f"- input_test_mode: {call.get('input_test_mode')}")
        lines.append(f"- hypothesis_index: {call.get('hypothesis_index')}")
        lines.append(f"- hypothesis_text: {call.get('hypothesis_text')}")
        if call.get("previous_hypothesis"):
            lines.append(f"- previous_hypothesis: {call.get('previous_hypothesis')}")
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
        lines.append(call.get("raw_output", ""))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def design_hypothesis_experiments(
    *,
    hypotheses_result: Dict[str, Any],
    num_input_sentences_per_hypothesis: int,
    round_id: Optional[str] = None,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    run_side: RunSideType = "both",
    input_test_mode: InputTestType = "activation",
    previous_input_hypotheses: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    model_id = hypotheses_result.get("model_id", "unknown-model")
    layer_id = str(hypotheses_result["layer_id"])
    feature_id = str(hypotheses_result["feature_id"])
    ts = str(hypotheses_result["timestamp"])
    resolved_round_id = normalize_round_id(
        round_id or str(hypotheses_result.get("round_id", "")).strip() or None,
        round_index=1,
    )
    input_hypotheses = [str(item).strip() for item in hypotheses_result.get("input_side_hypotheses", [])]
    output_hypotheses = [str(item).strip() for item in hypotheses_result.get("output_side_hypotheses", [])]
    num_hypothesis = max(len(input_hypotheses), len(output_hypotheses))
    run_input = run_side in ("input", "both")
    run_output = run_side in ("output", "both")

    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []
    client = OpenAI(
        base_url=llm_base_url,
        api_key=read_api_key(llm_api_key_file),
    )

    if run_input:
        input_sentences = _design_sentences(
            side="input",
            hypotheses=input_hypotheses,
            num_sentences=num_input_sentences_per_hypothesis,
            client=client,
            model=llm_model,
            token_counter=token_counter,
            llm_calls=llm_calls,
            temperature=temperature,
            max_tokens=max_tokens,
            input_test_mode=input_test_mode,
            previous_input_hypotheses=previous_input_hypotheses,
        )
    else:
        input_sentences = []

    if run_output:
        output_sentences = _design_sentences(
            side="output",
            hypotheses=output_hypotheses,
            num_sentences=num_input_sentences_per_hypothesis,
            client=client,
            model=llm_model,
            token_counter=token_counter,
            llm_calls=llm_calls,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        output_sentences = []

    input_experiments: List[Dict[str, Any]] = []
    for idx, (hypothesis, sentences) in enumerate(zip(input_hypotheses, input_sentences), start=1):
        reference_hypothesis = ""
        if input_test_mode == "expansion" and previous_input_hypotheses is not None:
            if 0 <= idx - 1 < len(previous_input_hypotheses):
                reference_hypothesis = str(previous_input_hypotheses[idx - 1]).strip()
        input_experiments.append(
            {
                "hypothesis": hypothesis,
                "test_type": input_test_mode,
                "reference_hypothesis": reference_hypothesis,
                "designed_sentences": sentences,
            }
        )

    result: Dict[str, Any] = {
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "round_id": resolved_round_id,
        "num_hypothesis": num_hypothesis,
        "generation_mode": hypotheses_result.get("generation_mode"),
        "run_side": run_side,
        "num_input_sentences_per_hypothesis": num_input_sentences_per_hypothesis,
        "input_test_mode": input_test_mode,
        "llm_model": llm_model,
        "input_side_experiments": input_experiments,
        "output_side_experiments": [
            {"hypothesis": hyp, "designed_sentences": sentences}
            for hyp, sentences in zip(output_hypotheses, output_sentences)
        ],
        "token_usage": token_counter.as_dict(),
        "llm_calls": llm_calls,
    }

    base_dir = build_round_dir(
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=resolved_round_id,
        round_index=1,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments.json"
    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments.md"
    _write_markdown_log(result_md_path, result=result, llm_calls=llm_calls)
    return result


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a dict: {path}")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3 of SAE workflow: design hypothesis validation experiments.",
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
        help="For each input-side hypothesis, generate this many activation or expansion sentences.",
    )
    parser.add_argument(
        "--input-test-mode",
        choices=["activation", "expansion"],
        default="activation",
        help="Input-side experiment type for this design run.",
    )
    parser.add_argument(
        "--previous-input-hypotheses-json-path",
        default=None,
        help="Optional JSON path with {'input_side_hypotheses': [...]} or a raw list, used in expansion mode.",
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
    parser.add_argument("--round-id", default=None, help="Round directory under timestamp, e.g. round_1")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, reuse logs artifacts from the target round directory.",
    )
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser


def _load_previous_input_hypotheses(path_text: Optional[str]) -> Optional[List[str]]:
    if not path_text:
        return None
    raw = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(item).strip() for item in raw]
    if not isinstance(raw, dict):
        raise ValueError("previous-input-hypotheses JSON must be either a list or a dict.")
    if isinstance(raw.get("input_side_hypotheses"), list):
        return [str(item).strip() for item in raw.get("input_side_hypotheses", [])]
    if isinstance(raw.get("hypotheses"), list):
        return [str(item).strip() for item in raw.get("hypotheses", [])]
    return None


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    previous_input_hypotheses = _load_previous_input_hypotheses(args.previous_input_hypotheses_json_path)

    if args.reuse_from_logs:
        if args.timestamp is None:
            raise ValueError("When --reuse-from-logs is set, --timestamp is required.")
        resolved_round_id = normalize_round_id(args.round_id, round_index=1)
        base_dir = resolve_existing_round_dir(
            layer_id=str(args.layer_id),
            feature_id=str(args.feature_id),
            timestamp=ts,
            round_id=resolved_round_id,
            round_index=1,
        )
        if base_dir is None:
            raise FileNotFoundError(
                f"Cannot find round directory under logs for layer={args.layer_id}, "
                f"feature={args.feature_id}, timestamp={ts}, round_id={resolved_round_id}"
            )
        observation_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-observation-input.json"
        initial_hypotheses_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-initial-hypotheses.json"
        if not observation_path.exists():
            raise FileNotFoundError(f"Cannot find observation input file: {observation_path}")
        if not initial_hypotheses_path.exists():
            raise FileNotFoundError(f"Cannot find initial hypotheses file: {initial_hypotheses_path}")
        _ = _load_json(observation_path)
        initial_result = _load_json(initial_hypotheses_path)
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
        initial_result = generate_initial_hypotheses(
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

    result = design_hypothesis_experiments(
        hypotheses_result=initial_result,
        num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
        round_id=args.round_id,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key_file=args.llm_api_key_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        input_test_mode=str(args.input_test_mode),
        previous_input_hypotheses=previous_input_hypotheses,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
