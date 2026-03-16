from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from openai import OpenAI

from function import TokenUsageAccumulator, call_llm
from prompts.experiments_execution_prompt import build_output_judge_system_prompt, build_output_judge_user_prompt

if TYPE_CHECKING:
    from model_with_sae import ModelWithSAEModule

PROJECT_ROOT = Path(__file__).resolve().parent
INTERVENTION_BLIND_SCORE_PATH = (
    PROJECT_ROOT
    / "explanation_quality_evaluation"
    / "output-side-evaluation"
    / "intervention_blind_score.py"
)


def _load_intervention_blind_score_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "intervention_blind_score_shared",
        str(INTERVENTION_BLIND_SCORE_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {INTERVENTION_BLIND_SCORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_shared_blind_score = _load_intervention_blind_score_module()
KL_DIV_VALUES_DEFAULT = list(getattr(_shared_blind_score, "KL_DIV_VALUES_DEFAULT"))
prepare_control_results = getattr(_shared_blind_score, "prepare_control_results")
format_intervention_result = getattr(_shared_blind_score, "format_intervention_result")
_extract_choice = getattr(_shared_blind_score, "_extract_choice")


def _extract_designed_prompts(experiment_item: Dict[str, Any]) -> List[str]:
    raw = experiment_item.get("designed_sentences")
    if not isinstance(raw, list):
        raise ValueError("Each output-side experiment item must contain a list field 'designed_sentences'.")
    prompts: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            prompts.append(item.strip())
    if not prompts:
        raise ValueError("Output-side designed_sentences must contain at least one non-empty prompt.")
    return prompts


def _run_single_blind_trial(
    *,
    hypothesis_text: str,
    intervention_result: str,
    control_results: Sequence[str],
    num_choices: int,
    client: OpenAI,
    llm_model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    hypothesis_index: int,
    trial_index: int,
    rng: random.Random,
    judge_temperature: float,
    judge_max_tokens: int,
) -> Dict[str, Any]:
    option_records: List[Dict[str, Any]] = [
        {"origin": "control", "text": control_results[idx], "control_index": idx + 1}
        for idx in range(num_choices - 1)
    ]
    option_records.append({"origin": "target_intervention", "text": intervention_result, "control_index": None})
    rng.shuffle(option_records)

    option_sets = [record["text"] for record in option_records]
    correct_choice = next(index + 1 for index, record in enumerate(option_records) if record["origin"] == "target_intervention")

    system_prompt = build_output_judge_system_prompt(num_sets=num_choices)
    user_prompt = build_output_judge_user_prompt(explanation=hypothesis_text, option_sets=option_sets)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output, usage_obj, response_debug = call_llm(
        client=client,
        model=llm_model,
        messages=messages,
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
        stream=False,
        response_format_text=True,
        return_debug=True,
    )
    usage_counts = token_counter.add(usage_obj)

    try:
        chosen_choice = _extract_choice(raw_output, num_choices=num_choices)
    except ValueError:
        chosen_choice = -1

    success = chosen_choice == correct_choice
    trial_result: Dict[str, Any] = {
        "trial_index": trial_index,
        "correct_choice": correct_choice,
        "chosen_choice": chosen_choice,
        "success": success,
        "option_origins": [record["origin"] for record in option_records],
        "blind_prompt_user": user_prompt,
        "judge_response": raw_output,
        "judge_response_debug": response_debug,
    }

    llm_calls.append(
        {
            "call_type": "output_blind_judge",
            "hypothesis_index": hypothesis_index,
            "trial_index": trial_index,
            "messages": messages,
            "raw_output": raw_output,
            "usage": usage_counts,
            "correct_choice": correct_choice,
            "chosen_choice": chosen_choice,
            "success": success,
            "response_debug": response_debug,
        }
    )

    return trial_result


def execute_output_side_experiments(
    *,
    output_side_experiments: Sequence[Dict[str, Any]],
    module: ModelWithSAEModule,
    client: OpenAI,
    llm_model: str,
    token_counter: TokenUsageAccumulator,
    llm_calls: List[Dict[str, Any]],
    num_choices: int = 3,
    trials: int = 1,
    seed: int = 42,
    max_new_tokens: int = 25,
    generation_temperature: float = 0.75,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 1024,
    kl_values: Sequence[float] = KL_DIV_VALUES_DEFAULT,
) -> Dict[str, Any]:
    if num_choices < 2:
        raise ValueError("num_choices must be at least 2.")
    if trials <= 0:
        raise ValueError("trials must be a positive integer.")
    if not kl_values:
        raise ValueError("kl_values must not be empty.")

    staged_results: List[Dict[str, Any]] = []
    for hypothesis_index, item in enumerate(output_side_experiments, start=1):
        hypothesis_text = str(item.get("hypothesis", "")).strip()
        prompts = _extract_designed_prompts(item)

        completions_by_kl: Dict[float, Sequence[str]] = {}
        steering_runs: Dict[str, Any] = {}
        for kl in kl_values:
            kl_float = float(kl)
            output = module.generate_steered_completions(
                prompts=list(prompts),
                feature_index=int(module.feature_index),
                max_new_tokens=max_new_tokens,
                temperature=generation_temperature,
                target_kl=kl_float,
            )
            completions = output.get("steered_completion")
            if not isinstance(completions, list):
                raise RuntimeError(f"Unexpected steered output format for KL={kl_float}: {output}")

            completions_by_kl[kl_float] = list(completions)
            steering_runs[f"{kl_float:+g}"] = {
                "target_kl": kl_float,
                "clamp_values": output.get("clamp_values", []),
                "actual_kl_values": output.get("kl_values", []),
                "steered_completion": output.get("steered_completion", []),
                "steered_full": output.get("steered_full", []),
            }

        intervention_result = format_intervention_result(
            completions_by_kl=completions_by_kl,
            prompts=prompts,
            kl_values=kl_values,
        )
        staged_results.append(
            {
                "hypothesis_index": hypothesis_index,
                "hypothesis_text": hypothesis_text,
                "prompts": prompts,
                "steering_runs": steering_runs,
                "intervention_result": intervention_result,
            }
        )

    hypothesis_results: List[Dict[str, Any]] = []
    for current in staged_results:
        hypothesis_index = int(current["hypothesis_index"])
        hypothesis_text = str(current["hypothesis_text"])
        prompts = list(current["prompts"])
        steering_runs = dict(current["steering_runs"])
        intervention_result = str(current["intervention_result"])

        control_pool = [
            str(candidate["intervention_result"])
            for candidate in staged_results
            if int(candidate["hypothesis_index"]) != hypothesis_index
        ]
        selected_controls = prepare_control_results(control_pool, num_required=num_choices - 1)

        rng = random.Random(seed + hypothesis_index * 10007)
        trial_results: List[Dict[str, Any]] = []
        for trial_index in range(1, trials + 1):
            trial_result = _run_single_blind_trial(
                hypothesis_text=hypothesis_text,
                intervention_result=intervention_result,
                control_results=selected_controls,
                num_choices=num_choices,
                client=client,
                llm_model=llm_model,
                token_counter=token_counter,
                llm_calls=llm_calls,
                hypothesis_index=hypothesis_index,
                trial_index=trial_index,
                rng=rng,
                judge_temperature=judge_temperature,
                judge_max_tokens=judge_max_tokens,
            )
            trial_results.append(trial_result)

        success_count = sum(1 for trial in trial_results if bool(trial["success"]))
        score = success_count / trials

        hypothesis_results.append(
            {
                "hypothesis_index": hypothesis_index,
                "hypothesis": hypothesis_text,
                "designed_sentences": prompts,
                "kl_values": [float(x) for x in kl_values],
                "steering_runs": steering_runs,
                "intervention_result": intervention_result,
                "control_results_used": selected_controls,
                "blind_judge_num_choices": num_choices,
                "blind_judge_trials": trials,
                "blind_judge_successes": success_count,
                "score_blind_accuracy": score,
                "trial_results": trial_results,
            }
        )

    if hypothesis_results:
        overall_score = sum(item["score_blind_accuracy"] for item in hypothesis_results) / len(hypothesis_results)
    else:
        overall_score = 0.0

    return {
        "side": "output",
        "kl_values": [float(x) for x in kl_values],
        "blind_judge_num_choices": num_choices,
        "blind_judge_trials": trials,
        "hypothesis_results": hypothesis_results,
        "overall_score_blind_accuracy": overall_score,
    }
