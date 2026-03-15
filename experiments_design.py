from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from openai import OpenAI

from function import TokenUsageAccumulator, call_llm_stream, extract_json_object, read_api_key
from initial_hypothesis_generation import GenerationMode, generate_initial_hypotheses
from llm_api.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from llm_api.llm_api_info import base_url as DEFAULT_BASE_URL
from llm_api.llm_api_info import model_name as DEFAULT_MODEL_NAME
from neuronpedia_feature_api import fetch_and_parse_feature_observation
from prompts.experiments_generation_prompt import build_system_prompt, build_user_prompt

SideType = Literal["input", "output"]

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

    if not sentences:
        raise ValueError(f"Failed to parse sentences from output: {raw_output}")

    if len(sentences) < expected_count:
        raise ValueError(
            f"Expected {expected_count} sentences, but only parsed {len(sentences)}: {raw_output}"
        )
    return sentences[:expected_count]


def _design_experiments_for_side(
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
) -> List[List[str]]:
    if side == "output":
        return [list(OUTPUT_SIDE_PLACEHOLDER) for _ in hypotheses]

    system_prompt = build_system_prompt(side)
    all_sentences: List[List[str]] = []

    for index, hypothesis in enumerate(hypotheses, start=1):
        user_prompt = build_user_prompt(side=side, hypothesis=hypothesis, num_sentences=num_sentences)
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
        designed_sentences = _parse_sentence_list(raw_output, expected_count=num_sentences)
        all_sentences.append(designed_sentences)

        llm_calls.append(
            {
                "side": side,
                "hypothesis_index": index,
                "hypothesis_text": hypothesis,
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
    lines.append(f"- num_hypothesis: {result['num_hypothesis']}")
    lines.append(f"- num_input_sentences_per_hypothesis: {result['num_input_sentences_per_hypothesis']}")
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
        lines.append("- designed_sentences:")
        for sentence in pair["designed_sentences"]:
            lines.append(f"  - {sentence}")
        lines.append("")
    lines.append("## Output-side Hypotheses And Designed Sentences")
    for idx, pair in enumerate(result["output_side_experiments"], start=1):
        lines.append(f"### Output Hypothesis {idx}")
        lines.append(f"- hypothesis: {pair['hypothesis']}")
        lines.append("- designed_sentences:")
        for sentence in pair["designed_sentences"]:
            lines.append(f"  - {sentence}")
        lines.append("")
    lines.append("## LLM Calls")
    for i, call in enumerate(llm_calls, start=1):
        lines.append(f"### Call {i}")
        lines.append(f"- side: {call['side']}")
        lines.append(f"- hypothesis_index: {call['hypothesis_index']}")
        lines.append(f"- hypothesis_text: {call['hypothesis_text']}")
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


def design_hypothesis_experiments(
    *,
    hypotheses_result: Dict[str, Any],
    num_input_sentences_per_hypothesis: int,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    model_id = hypotheses_result.get("model_id", "unknown-model")
    layer_id = str(hypotheses_result["layer_id"])
    feature_id = str(hypotheses_result["feature_id"])
    ts = str(hypotheses_result["timestamp"])
    input_hypotheses = list(hypotheses_result["input_side_hypotheses"])
    output_hypotheses = list(hypotheses_result["output_side_hypotheses"])
    num_hypothesis = len(input_hypotheses)

    client = OpenAI(
        base_url=llm_base_url,
        api_key=read_api_key(llm_api_key_file),
    )
    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    input_sentences = _design_experiments_for_side(
        side="input",
        hypotheses=input_hypotheses,
        num_sentences=num_input_sentences_per_hypothesis,
        client=client,
        model=llm_model,
        token_counter=token_counter,
        llm_calls=llm_calls,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_sentences = _design_experiments_for_side(
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

    base_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts
    base_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "num_hypothesis": num_hypothesis,
        "generation_mode": hypotheses_result.get("generation_mode"),
        "num_input_sentences_per_hypothesis": num_input_sentences_per_hypothesis,
        "llm_model": llm_model,
        "input_side_experiments": [
            {"hypothesis": hyp, "designed_sentences": sentences}
            for hyp, sentences in zip(input_hypotheses, input_sentences)
        ],
        "output_side_experiments": [
            {"hypothesis": hyp, "designed_sentences": sentences}
            for hyp, sentences in zip(output_hypotheses, output_sentences)
        ],
        "token_usage": token_counter.as_dict(),
    }

    result_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments.json"
    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments.md"
    _write_markdown_log(result_md_path, result=result, llm_calls=llm_calls)

    return result


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
        help="For each input-side hypothesis, generate this many activation sentences.",
    )
    parser.add_argument("--width", default="16k", help="Neuronpedia source width")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument("--timestamp", default=None, help="Custom timestamp for logs/{layer}_{feature}/{timestamp}")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, reuse logs/{layer}_{feature}/{timestamp} intermediate JSON files instead of refetching.",
    )
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=50000)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.reuse_from_logs:
        if args.timestamp is None:
            raise ValueError("When --reuse-from-logs is set, --timestamp is required.")
        base_dir = Path("logs") / f"{args.layer_id}_{args.feature_id}" / ts
        observation_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-observation-input.json"
        initial_hypotheses_path = base_dir / f"layer{args.layer_id}-feature{args.feature_id}-initial-hypotheses.json"

        if not observation_path.exists():
            raise FileNotFoundError(f"Cannot find observation input file: {observation_path}")
        if not initial_hypotheses_path.exists():
            raise FileNotFoundError(f"Cannot find initial hypotheses file: {initial_hypotheses_path}")

        # Kept for workflow completeness: this is the step-1 parsed output structure.
        _ = json.loads(observation_path.read_text(encoding="utf-8"))
        initial_result = json.loads(initial_hypotheses_path.read_text(encoding="utf-8"))
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

    result = design_hypothesis_experiments(
        hypotheses_result=initial_result,
        num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key_file=args.llm_api_key_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
