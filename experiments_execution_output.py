from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence

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
INTERVENTION_LOGIT_TOPK_SCORE_PATH = (
    PROJECT_ROOT
    / "explanation_quality_evaluation"
    / "output-side-evaluation"
    / "intervention_logit_topk_score.py"
)
OutputInterventionMethod = Literal["blind", "logit"]


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


def _load_intervention_logit_topk_score_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "intervention_logit_topk_score_shared",
        str(INTERVENTION_LOGIT_TOPK_SCORE_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {INTERVENTION_LOGIT_TOPK_SCORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_shared_blind_score = _load_intervention_blind_score_module()
_shared_logit_topk_score = _load_intervention_logit_topk_score_module()
KL_DIV_VALUES_DEFAULT = list(getattr(_shared_blind_score, "KL_DIV_VALUES_DEFAULT"))
prepare_control_results = getattr(_shared_blind_score, "prepare_control_results")
format_intervention_result = getattr(_shared_blind_score, "format_intervention_result")
_extract_choice = getattr(_shared_blind_score, "_extract_choice")
score_logit_topk_for_hypothesis = getattr(_shared_logit_topk_score, "score_logit_topk_for_hypothesis")


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
    intervention_method: OutputInterventionMethod = "blind",
    logit_top_k: int = 5,
    logit_kl_tolerance: float = 0.1,
    logit_kl_max_steps: int = 12,
    logit_force_refresh_kl_cache: bool = False,
    logit_include_special_tokens: bool = False,
) -> Dict[str, Any]:
    method = str(intervention_method).strip().lower()
    if method not in ("blind", "logit"):
        raise ValueError("intervention_method must be one of: blind, logit.")

    if method == "blind":
        if num_choices < 2:
            raise ValueError("num_choices must be at least 2.")
        if trials <= 0:
            raise ValueError("trials must be a positive integer.")
    if method == "logit" and int(logit_top_k) <= 0:
        raise ValueError("logit_top_k must be a positive integer.")
    if not kl_values:
        raise ValueError("kl_values must not be empty.")

    staged_results: List[Dict[str, Any]] = []
    for hypothesis_index, item in enumerate(output_side_experiments, start=1):
        hypothesis_text = str(item.get("hypothesis", "")).strip()
        prompts = _extract_designed_prompts(item)

        steering_runs: Dict[str, Any] = {}
        intervention_result = ""
        if method == "blind":
            completions_by_kl: Dict[float, Sequence[str]] = {}
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

        selected_controls: List[str] = []
        trial_results: List[Dict[str, Any]] = []
        blind_result: Dict[str, Any] = {}
        logit_topk_result: Dict[str, Any] = {}
        score_blind_accuracy: Any = None
        score_logit_topk: Any = None
        score_name = "score_blind_accuracy" if method == "blind" else "score_logit_topk"

        if method == "blind":
            control_pool = [
                str(candidate["intervention_result"])
                for candidate in staged_results
                if int(candidate["hypothesis_index"]) != hypothesis_index
            ]
            selected_controls = prepare_control_results(control_pool, num_required=num_choices - 1)

            rng = random.Random(seed + hypothesis_index * 10007)
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
            score_blind_accuracy = success_count / trials
            blind_result = {
                "control_results_used": selected_controls,
                "blind_judge_num_choices": num_choices,
                "blind_judge_trials": trials,
                "blind_judge_successes": success_count,
                "trial_results": trial_results,
            }
        else:
            logit_topk_result = score_logit_topk_for_hypothesis(
                module=module,
                feature_id=int(module.feature_index),
                hypothesis_index=hypothesis_index,
                explanation=hypothesis_text,
                prompts=prompts,
                target_kls=[float(x) for x in kl_values],
                top_k=int(logit_top_k),
                judge_client=client,
                judge_model=llm_model,
                judge_max_tokens=int(judge_max_tokens),
                include_special_tokens=bool(logit_include_special_tokens),
                kl_tolerance=float(logit_kl_tolerance),
                kl_max_steps=int(logit_kl_max_steps),
                force_refresh_kl_cache=bool(logit_force_refresh_kl_cache),
                clamp_cache_path=None,
                llm_calls=llm_calls,
            )
            for run in logit_topk_result.get("runs", []):
                if not isinstance(run, dict):
                    continue
                usage = run.get("llm_judge", {}).get("usage", {})
                token_counter.add(usage)
            score_logit_topk = logit_topk_result.get("summary", {}).get("mean_signed_topk_accuracy")

        hypothesis_results.append(
            {
                "hypothesis_index": hypothesis_index,
                "hypothesis": hypothesis_text,
                "designed_sentences": prompts,
                "kl_values": [float(x) for x in kl_values],
                "steering_runs": steering_runs,
                "intervention_result": intervention_result,
                "control_results_used": selected_controls,
                "blind_judge_num_choices": num_choices if method == "blind" else None,
                "blind_judge_trials": trials if method == "blind" else None,
                "blind_judge_successes": blind_result.get("blind_judge_successes") if method == "blind" else None,
                "score_blind_accuracy": score_blind_accuracy,
                "score_logit_topk": score_logit_topk,
                "score_name": score_name,
                "score_primary": score_blind_accuracy if method == "blind" else score_logit_topk,
                "blind_result": blind_result,
                "logit_topk_result": logit_topk_result,
                "trial_results": trial_results,
            }
        )

    blind_scores = [
        float(item["score_blind_accuracy"])
        for item in hypothesis_results
        if isinstance(item.get("score_blind_accuracy"), (int, float))
    ]
    logit_scores = [
        float(item["score_logit_topk"])
        for item in hypothesis_results
        if isinstance(item.get("score_logit_topk"), (int, float))
    ]
    overall_blind = (sum(blind_scores) / len(blind_scores)) if blind_scores else None
    overall_logit = (sum(logit_scores) / len(logit_scores)) if logit_scores else None

    return {
        "side": "output",
        "output_intervention_method": method,
        "output_score_name": "score_blind_accuracy" if method == "blind" else "score_logit_topk",
        "kl_values": [float(x) for x in kl_values],
        "blind_judge_num_choices": num_choices if method == "blind" else None,
        "blind_judge_trials": trials if method == "blind" else None,
        "logit_top_k": int(logit_top_k) if method == "logit" else None,
        "hypothesis_results": hypothesis_results,
        "overall_score_blind_accuracy": overall_blind,
        "overall_score_logit_topk": overall_logit,
        "overall_score_primary": overall_blind if method == "blind" else overall_logit,
        "blind_evaluation": (
            {"num_choices": num_choices, "trials": trials, "overall_score_blind_accuracy": overall_blind}
            if method == "blind"
            else {}
        ),
        "logit_evaluation": (
            {"top_k": int(logit_top_k), "overall_score_logit_topk": overall_logit}
            if method == "logit"
            else {}
        ),
    }
