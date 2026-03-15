from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from experiments_execution_input import execute_input_side_experiments
from experiments_execution_output import KL_DIV_VALUES_DEFAULT, execute_output_side_experiments
from experiments_design import design_hypothesis_experiments
from function import TokenUsageAccumulator, build_round_dir, normalize_round_id, read_api_key
from initial_hypothesis_generation import generate_initial_hypotheses
from llm_api.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from llm_api.llm_api_info import base_url as DEFAULT_BASE_URL
from llm_api.llm_api_info import model_name as DEFAULT_MODEL_NAME
from model_with_sae import ModelWithSAEModule
from neuronpedia_feature_api import fetch_and_parse_feature_observation


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
        # Keep backward-compatible fallback.
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


def _write_markdown_log(
    path: Path,
    *,
    result: Dict[str, Any],
    llm_calls: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# SAE Hypothesis Experiments Execution")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- model_id: {result['model_id']}")
    lines.append(f"- layer_id: {result['layer_id']}")
    lines.append(f"- feature_id: {result['feature_id']}")
    lines.append(f"- timestamp: {result['timestamp']}")
    if "round_id" in result:
        lines.append(f"- round_id: {result['round_id']}")
    lines.append(f"- output_judge_llm_model: {result['output_judge_llm_model']}")
    lines.append("")
    lines.append("## Token Usage (Output-side Blind Judge)")
    token_usage = result["token_usage"]
    lines.append(f"- prompt_tokens: {token_usage['prompt_tokens']}")
    lines.append(f"- completion_tokens: {token_usage['completion_tokens']}")
    lines.append(f"- total_tokens: {token_usage['total_tokens']}")
    lines.append("")

    input_side = result["input_side_execution"]
    lines.append("## Input-side Execution")
    lines.append(f"- non_zero_threshold: {input_side['non_zero_threshold']}")
    lines.append(f"- overall_score_non_zero_rate: {input_side['overall_score_non_zero_rate']}")
    lines.append("")
    for hypothesis_result in input_side["hypothesis_results"]:
        lines.append(f"### Input Hypothesis {hypothesis_result['hypothesis_index']}")
        lines.append(f"- explanation_original: {hypothesis_result['hypothesis']}")
        lines.append(f"- score_non_zero_rate: {hypothesis_result['score_non_zero_rate']}")
        lines.append(f"- mean_summary_activation: {hypothesis_result['mean_summary_activation']}")
        lines.append(f"- max_summary_activation: {hypothesis_result['max_summary_activation']}")
        lines.append("")
        lines.append("#### Input Activation Context")
        lines.append("```text")
        lines.append(hypothesis_result["input_activation_context"])
        lines.append("```")
        lines.append("| sentence_index | sentence | summary_activation | max_token | non_zero |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for sentence_result in hypothesis_result["sentence_results"]:
            sentence = str(sentence_result["sentence"]).replace("|", "\\|")
            lines.append(
                f"| {sentence_result['sentence_index']} | {sentence} | "
                f"{sentence_result['summary_activation']} | {sentence_result['max_token']} | "
                f"{sentence_result['is_non_zero']} |"
            )
        lines.append("")

    output_side = result["output_side_execution"]
    lines.append("## Output-side Execution")
    lines.append(f"- blind_judge_num_choices: {output_side['blind_judge_num_choices']}")
    lines.append(f"- blind_judge_trials: {output_side['blind_judge_trials']}")
    lines.append(f"- overall_score_blind_accuracy: {output_side['overall_score_blind_accuracy']}")
    lines.append(f"- kl_values: {output_side['kl_values']}")
    lines.append("")
    for hypothesis_result in output_side["hypothesis_results"]:
        lines.append(f"### Output Hypothesis {hypothesis_result['hypothesis_index']}")
        lines.append(f"- explanation_original: {hypothesis_result['hypothesis']}")
        lines.append(f"- score_blind_accuracy: {hypothesis_result['score_blind_accuracy']}")
        lines.append(f"- blind_judge_successes: {hypothesis_result['blind_judge_successes']}")
        lines.append(f"- blind_judge_trials: {hypothesis_result['blind_judge_trials']}")
        lines.append("")
        lines.append("#### Designed Sentences")
        for sentence in hypothesis_result["designed_sentences"]:
            lines.append(f"- {sentence}")
        lines.append("")
        lines.append("#### Intervention Result")
        lines.append("```text")
        lines.append(hypothesis_result["intervention_result"])
        lines.append("```")
        lines.append("")
        lines.append("#### Trial Results")
        for trial_result in hypothesis_result["trial_results"]:
            lines.append(
                f"- trial {trial_result['trial_index']}: "
                f"correct={trial_result['correct_choice']}, "
                f"chosen={trial_result['chosen_choice']}, "
                f"success={trial_result['success']}"
            )
        lines.append("")

    lines.append("## LLM API Calls")
    for call_index, call in enumerate(llm_calls, start=1):
        lines.append(f"### Call {call_index}")
        lines.append(f"- call_type: {call.get('call_type', '')}")
        lines.append(f"- hypothesis_index: {call.get('hypothesis_index', '')}")
        lines.append(f"- trial_index: {call.get('trial_index', '')}")
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
        lines.append(str(call.get("raw_output", "")))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_hypothesis_experiments(
    *,
    experiments_result: Dict[str, Any],
    module: ModelWithSAEModule,
    control_results: Sequence[str],
    round_id: Optional[str] = None,
    llm_base_url: str = DEFAULT_BASE_URL,
    llm_model: str = DEFAULT_MODEL_NAME,
    llm_api_key_file: str = DEFAULT_API_KEY_FILE,
    input_non_zero_threshold: float = 0.0,
    output_judge_num_choices: int = 3,
    output_judge_trials: int = 1,
    output_judge_seed: int = 42,
    output_max_new_tokens: int = 25,
    output_generation_temperature: float = 0.75,
    output_judge_temperature: float = 0.0,
    output_judge_max_tokens: int = 1024,
    output_kl_values: Sequence[float] = KL_DIV_VALUES_DEFAULT,
) -> Dict[str, Any]:
    model_id = str(experiments_result.get("model_id", "unknown-model"))
    layer_id = str(experiments_result["layer_id"])
    feature_id = str(experiments_result["feature_id"])
    ts = str(experiments_result["timestamp"])
    resolved_round_id = normalize_round_id(
        round_id or str(experiments_result.get("round_id", "")).strip() or None,
        round_index=1,
    )
    input_side_experiments = list(experiments_result.get("input_side_experiments", []))
    output_side_experiments = list(experiments_result.get("output_side_experiments", []))

    input_side_execution = execute_input_side_experiments(
        input_side_experiments=input_side_experiments,
        module=module,
        non_zero_threshold=input_non_zero_threshold,
    )

    llm_calls: List[Dict[str, Any]] = []
    token_counter = TokenUsageAccumulator()

    if output_side_experiments and (
        not isinstance(getattr(module, "sae", None), dict) or "__sae_lens_obj__" not in module.sae
    ):
        raise RuntimeError(
            "Output-side execution requires a loaded SAE-Lens object for KL-guided steering. "
            "Current SAE is unavailable or not SAE-Lens format. "
            "Please provide a valid --sae-path (sae-lens URI or compatible checkpoint)."
        )

    client = OpenAI(
        base_url=llm_base_url,
        api_key=read_api_key(llm_api_key_file),
    )
    output_side_execution = execute_output_side_experiments(
        output_side_experiments=output_side_experiments,
        module=module,
        client=client,
        llm_model=llm_model,
        token_counter=token_counter,
        llm_calls=llm_calls,
        control_results=control_results,
        num_choices=output_judge_num_choices,
        trials=output_judge_trials,
        seed=output_judge_seed,
        max_new_tokens=output_max_new_tokens,
        generation_temperature=output_generation_temperature,
        judge_temperature=output_judge_temperature,
        judge_max_tokens=output_judge_max_tokens,
        kl_values=output_kl_values,
    )

    result: Dict[str, Any] = {
        "model_id": model_id,
        "layer_id": layer_id,
        "feature_id": feature_id,
        "timestamp": ts,
        "round_id": resolved_round_id,
        "output_judge_llm_model": llm_model,
        "input_side_execution": input_side_execution,
        "output_side_execution": output_side_execution,
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
    result_json_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments-execution.json"
    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_md_path = base_dir / f"layer{layer_id}-feature{feature_id}-experiments-execution.md"
    _write_markdown_log(result_md_path, result=result, llm_calls=llm_calls)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 4 of SAE workflow: execute hypothesis experiments for input-side activation and output-side intervention.",
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
    parser.add_argument("--round-id", default=None, help="Round directory under timestamp, e.g. round_1")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="If set, reuse logs/{layer}_{feature}/{timestamp}/{round_id} intermediate JSON files instead of refetching.",
    )
    parser.add_argument(
        "--experiments-json-path",
        default=None,
        help="Optional direct path to a layer{layer}-feature{feature}-experiments.json file.",
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

    if args.experiments_json_path:
        experiments_path = Path(args.experiments_json_path)
        if not experiments_path.exists():
            raise FileNotFoundError(f"Cannot find experiments file: {experiments_path}")
        experiments_result = json.loads(experiments_path.read_text(encoding="utf-8"))
    elif args.reuse_from_logs:
        if args.timestamp is None:
            raise ValueError("When --reuse-from-logs is set, --timestamp is required.")
        resolved_round_id = normalize_round_id(args.round_id, round_index=1)
        experiments_path = (
            Path("logs")
            / f"{args.layer_id}_{args.feature_id}"
            / ts
            / resolved_round_id
            / f"layer{args.layer_id}-feature{args.feature_id}-experiments.json"
        )
        if not experiments_path.exists():
            raise FileNotFoundError(f"Cannot find experiments file: {experiments_path}")
        experiments_result = json.loads(experiments_path.read_text(encoding="utf-8"))
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
        experiments_result = design_hypothesis_experiments(
            hypotheses_result=initial_result,
            num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
            round_id=args.round_id,
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
        round_id=args.round_id,
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
    print(json.dumps(execution_result, ensure_ascii=False, indent=2))
