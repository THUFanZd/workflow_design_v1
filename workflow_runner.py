from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments_design import design_hypothesis_experiments
from experiments_execution import _default_sae_path, execute_hypothesis_experiments
from experiments_execution_output import KL_DIV_VALUES_DEFAULT
from function import TokenUsageAccumulator, build_round_dir, normalize_round_id
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


def _load_control_results(control_result_files: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for file_path in control_result_files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find control result file: {path}")
        texts.append(path.read_text(encoding="utf-8").strip())
    return texts


def _result_usage(result: Dict[str, Any]) -> Dict[str, int]:
    usage = result.get("token_usage")
    if isinstance(usage, dict):
        return usage
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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
        description="Run SAE explanation workflow (step 1-7) with round/step resume support.",
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id")
    parser.add_argument("--layer-id", required=True, help="Layer id")
    parser.add_argument("--feature-id", required=True, help="Feature id")
    parser.add_argument("--timestamp", default=None, help="Timestamp directory under logs/{layer}_{feature}/")
    parser.add_argument("--max-rounds", type=int, default=1, help="Maximum refinement calls (step 6 iterations).")
    parser.add_argument("--start-round", type=int, default=0, help="Round index to start real execution from.")
    parser.add_argument("--start-step", type=int, default=1, help="Step index to start real execution from.")
    parser.add_argument(
        "--reuse-from-logs",
        action="store_true",
        help="Reuse artifacts before start point from logs/{layer}_{feature}/{timestamp}/{round_id}.",
    )

    parser.add_argument("--num-hypothesis", type=int, default=3, help="Hypothesis count n for each side")
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
        help="Input-side designed sentences per hypothesis.",
    )
    parser.add_argument("--top-m", type=int, default=None, help="Refine top-m hypotheses per side (default=all).")
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
    parser.add_argument("--max-tokens", type=int, default=20000)

    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--sae-path", default=None, help="SAE path or sae-lens URI")
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument(
        "--sae-canonical-map",
        default=str(Path("model_download") / "canonical_map.txt"),
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

    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be >= 0.")
    if args.start_round < 0:
        raise ValueError("--start-round must be >= 0.")
    if args.start_round == 0 and args.start_step not in (1, 2):
        raise ValueError("When --start-round=0, --start-step must be 1 or 2.")
    if args.start_round >= 1 and args.start_step not in (3, 4, 5, 6):
        raise ValueError("When --start-round>=1, --start-step must be one of 3/4/5/6.")
    if args.start_round > args.max_rounds and args.start_round > 0:
        raise ValueError("--start-round cannot be greater than --max-rounds.")
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
    print('collect observation...')
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
    print('generate initial hypotheses...')
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
    round_memories: Dict[int, Dict[str, Any]] = {}
    round_refinements: Dict[int, Dict[str, Any]] = {}
    module: Optional[ModelWithSAEModule] = None
    control_results: Optional[List[str]] = None
    converged = False
    converged_round: Optional[int] = None
    last_round_executed = 0

    for round_index in range(1, args.max_rounds + 1):
        round_id = _round_id_from_index(round_index)
        print(f'round {round_index}...')
        if round_index > 1:
            prev_refinement = round_refinements.get(round_index - 1)
            if prev_refinement is None:
                prev_refine_path = _artifact_json_path(
                    layer_id=layer_id,
                    feature_id=feature_id,
                    timestamp=ts,
                    round_index=round_index - 1,
                    kind="refined_hypotheses",
                )
                prev_refinement = _load_json_or_raise(prev_refine_path)
                round_refinements[round_index - 1] = prev_refinement
                track_usage(prev_refinement, executed=False)
            current_hypotheses = _to_next_round_hypotheses(
                refinement_result=prev_refinement,
                round_index=round_index,
            )

        experiments_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="experiments",
        )
        print('design experiments...')
        if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=3):
            experiments_result = design_hypothesis_experiments(
                hypotheses_result=current_hypotheses,
                num_input_sentences_per_hypothesis=args.num_input_sentences_per_hypothesis,
                round_id=round_id,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_api_key_file=args.llm_api_key_file,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            executed_steps.append(f"{round_id}_step_3_experiments_design")
            track_usage(experiments_result, executed=True)
        else:
            experiments_result = _load_json_or_raise(experiments_path)
            loaded_steps.append(f"{round_id}_step_3_experiments_design")
            track_usage(experiments_result, executed=False)

        execution_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="experiments_execution",
        )
        print('execute experiments...')
        if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=4):
            if module is None:
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
            if control_results is None:
                control_results = _load_control_results(args.control_result_files)
            execution_result = execute_hypothesis_experiments(
                experiments_result=experiments_result,
                module=module,
                control_results=control_results,
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
            )
            executed_steps.append(f"{round_id}_step_4_experiments_execution")
            track_usage(execution_result, executed=True)
        else:
            execution_result = _load_json_or_raise(execution_path)
            loaded_steps.append(f"{round_id}_step_4_experiments_execution")
            track_usage(execution_result, executed=False)

        memory_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="memory",
        )
        memory_md_path = memory_path.with_suffix(".md")
        print('build memory...')
        if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=5):
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
            executed_steps.append(f"{round_id}_step_5_memory")
        else:
            memory_result = _load_json_or_raise(memory_path)
            loaded_steps.append(f"{round_id}_step_5_memory")
        round_memories[round_index] = memory_result

        refine_path = _artifact_json_path(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=ts,
            round_index=round_index,
            kind="refined_hypotheses",
        )
        print('refine hypotheses...')
        if _should_run(start_round=args.start_round, start_step=args.start_step, round_index=round_index, step_index=6):
            historical_memories: List[Dict[str, Any]] = []
            for hist_round in range(1, round_index):
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
                current_memory=memory_result,
                current_execution_result=execution_result,
                historical_memories=historical_memories,
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
            executed_steps.append(f"{round_id}_step_6_refinement")
            track_usage(refinement_result, executed=True)
        else:
            refinement_result = _load_json_or_raise(refine_path)
            loaded_steps.append(f"{round_id}_step_6_refinement")
            track_usage(refinement_result, executed=False)

        round_refinements[round_index] = refinement_result
        last_round_executed = round_index

        if _same_hypotheses(current_hypotheses, refinement_result):
            converged = True
            converged_round = round_index
            break

    if last_round_executed > 0:
        final_hypotheses_source = round_refinements[last_round_executed]
        final_input_hypotheses = list(final_hypotheses_source.get("input_side_hypotheses", []))
        final_output_hypotheses = list(final_hypotheses_source.get("output_side_hypotheses", []))
        final_input_reasons = list(final_hypotheses_source.get("input_side_hypothesis_reasons", []))
        final_output_reasons = list(final_hypotheses_source.get("output_side_hypothesis_reasons", []))
    else:
        final_hypotheses_source = initial_result
        final_input_hypotheses = list(initial_result.get("input_side_hypotheses", []))
        final_output_hypotheses = list(initial_result.get("output_side_hypotheses", []))
        final_input_reasons = list(initial_result.get("input_side_hypothesis_reasons", []))
        final_output_reasons = list(initial_result.get("output_side_hypothesis_reasons", []))

    ts_dir = Path("logs") / f"{layer_id}_{feature_id}" / ts
    ts_dir.mkdir(parents=True, exist_ok=True)

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
                "input_final_hypothesis_count": len(final_input_hypotheses),
                "output_final_hypothesis_count": len(final_output_hypotheses),
                "token_usage_total": total_tokens.as_dict(),
                "token_usage_this_run": run_tokens.as_dict(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
