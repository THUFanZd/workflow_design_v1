from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

# Allow importing project modules when this script is launched from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_design import OUTPUT_SIDE_PLACEHOLDER
from experiments_execution_output import (
    KL_DIV_VALUES_DEFAULT,
    _extract_choice,
    format_intervention_result,
    prepare_control_results,
)
from function import (
    DEFAULT_CANONICAL_MAP_PATH,
    TokenUsageAccumulator,
    build_default_sae_path,
    call_llm,
)
from prompts.experiments_execution_prompt import build_output_judge_system_prompt, build_output_judge_user_prompt
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE

if TYPE_CHECKING:
    from model_with_sae import ModelWithSAEModule

KL_DIV_VALUES = [float(x) for x in KL_DIV_VALUES_DEFAULT]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"
DEFAULT_SAE_RELEASE_BY_NAME: Dict[str, str] = {
    "gemmascope-res": "gemma-scope-2b-pt-res",
}


@dataclass
class BlindTrialResult:
    trial_index: int
    correct_choice: int
    chosen_choice: int
    success: bool
    option_origins: List[str]
    blind_prompt_user: str
    judge_response: str


@dataclass
class InterventionBlindScore:
    explanation: str
    score: float
    successes: int
    trials: int
    trial_results: List[BlindTrialResult]


def _normalize_result_text(text: str) -> str:
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def generate_explanation_from_result(intervention_result: str) -> str:
    payloads: List[str] = []
    for line in intervention_result.splitlines():
        if "': '" in line:
            payloads.append(line.split("': '", 1)[1].strip().strip("'"))
    text = " ".join(payloads).lower()
    if any(x in text for x in ["marine", "ocean", "coastal", "wetland", "fishing"]):
        return "Related to marine ecological protection, climate change impacts, and mitigation actions."
    if any(x in text for x in ["court", "legal", "plaintiff", "defendant", "motion", "appeal"]):
        return "Related to legal procedure, courtroom discourse, and judicial decision language."
    if any(x in text for x in ["gene", "protein", "cell", "assay", "biomarker", "mutation"]):
        return "Related to biomedical and molecular biology research language."
    return "This SAE feature appears to steer completions toward a coherent semantic theme."


def _resolve_api_key(api_key: Optional[str], api_key_file: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key.strip()
    if api_key_file:
        key_path = Path(api_key_file)
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                return key
    return None


def _resolve_target_dir(*, output_root: Path, sae_name: str, layer_id: str, feature_id: int) -> Path:
    target_dir = output_root / sae_name / f"layer-{layer_id}" / f"feature-{feature_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _parse_feature_dir_id(path: Path) -> Optional[int]:
    name = path.name
    match = re.match(r"^feature-(\d+)$", name)
    if not match:
        return None
    return int(match.group(1))


def _discover_control_result_paths(
    *,
    output_root: Path,
    sae_name: str,
    layer_id: str,
    feature_id: int,
    num_required: int,
) -> List[Path]:
    layer_dir = output_root / sae_name / f"layer-{layer_id}"
    if not layer_dir.exists():
        return []

    candidates: List[Tuple[int, int, Path]] = []
    for item in layer_dir.iterdir():
        if not item.is_dir():
            continue
        candidate_feature_id = _parse_feature_dir_id(item)
        if candidate_feature_id is None or candidate_feature_id == feature_id:
            continue
        intervention_path = item / "intervention_output.txt"
        if not intervention_path.exists():
            continue
        text = intervention_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        distance = abs(candidate_feature_id - feature_id)
        candidates.append((distance, candidate_feature_id, intervention_path))

    candidates.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in candidates[:num_required]]


def _resolve_sae_uri(
    *,
    layer_id: str,
    width: str,
    sae_name: str,
    sae_release: Optional[str],
    sae_average_l0: Optional[str],
    sae_canonical_map: Path,
) -> str:
    release = (sae_release or DEFAULT_SAE_RELEASE_BY_NAME.get(sae_name) or sae_name).strip()
    sae_uri, _ = build_default_sae_path(
        layer_id=layer_id,
        width=width,
        release=release,
        average_l0=sae_average_l0,
        canonical_map_path=sae_canonical_map,
    )
    return sae_uri


def build_intervention_result_from_checkpoint(
    *,
    model_checkpoint_path: str,
    layer_id: str,
    width: str,
    feature_id: int,
    sae_name: str,
    sae_release: Optional[str],
    sae_average_l0: Optional[str],
    sae_canonical_map: Path,
    device: str = "cpu",
    prompts: Sequence[str] = tuple(OUTPUT_SIDE_PLACEHOLDER),
    max_new_tokens: int = 25,
    generation_temperature: float = 0.75,
    kl_values: Sequence[float] = tuple(KL_DIV_VALUES),
) -> str:
    from model_with_sae import ModelWithSAEModule

    sae_uri = _resolve_sae_uri(
        layer_id=layer_id,
        width=width,
        sae_name=sae_name,
        sae_release=sae_release,
        sae_average_l0=sae_average_l0,
        sae_canonical_map=sae_canonical_map,
    )
    module = ModelWithSAEModule(
        llm_name=model_checkpoint_path,
        sae_path=sae_uri,
        sae_layer=int(layer_id),
        feature_index=int(feature_id),
        device=device,
    )

    completions_by_kl: Dict[float, Sequence[str]] = {}
    for kl in kl_values:
        kl_float = float(kl)
        out = module.generate_steered_completions(
            prompts=list(prompts),
            feature_index=int(feature_id),
            max_new_tokens=max_new_tokens,
            temperature=generation_temperature,
            target_kl=kl_float,
        )
        completions = out.get("steered_completion")
        if not isinstance(completions, list):
            raise RuntimeError(f"Unexpected steered output format for KL={kl_float}: {out}")
        completions_by_kl[kl_float] = [str(x) for x in completions]

    return format_intervention_result(
        completions_by_kl=completions_by_kl,
        prompts=list(prompts),
        kl_values=list(kl_values),
    )


def score_intervention_explanation(
    *,
    explanation: str,
    intervention_result: str,
    control_results: Sequence[str],
    client: OpenAI,
    llm_model: str,
    trials: int,
    seed: int,
    num_choices: int,
    judge_temperature: float,
    judge_max_tokens: int,
) -> Tuple[InterventionBlindScore, Dict[str, int], List[Dict[str, Any]]]:
    if trials <= 0:
        raise ValueError("trials must be a positive integer.")
    if num_choices < 2:
        raise ValueError("num_choices must be at least 2.")

    intervention = _normalize_result_text(intervention_result)
    controls = prepare_control_results(control_results, num_required=num_choices - 1)
    rng = random.Random(seed)
    token_counter = TokenUsageAccumulator()
    llm_calls: List[Dict[str, Any]] = []

    trial_results: List[BlindTrialResult] = []
    for trial_index in range(1, trials + 1):
        option_records: List[Dict[str, Any]] = [
            {"origin": "control", "text": controls[idx], "control_index": idx + 1}
            for idx in range(num_choices - 1)
        ]
        option_records.append({"origin": "target_intervention", "text": intervention, "control_index": None})
        rng.shuffle(option_records)

        option_sets = [record["text"] for record in option_records]
        correct_choice = next(
            index + 1 for index, record in enumerate(option_records) if record["origin"] == "target_intervention"
        )

        system_prompt = build_output_judge_system_prompt(num_sets=num_choices)
        user_prompt = build_output_judge_user_prompt(explanation=explanation, option_sets=option_sets)
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
        trial_results.append(
            BlindTrialResult(
                trial_index=trial_index,
                correct_choice=correct_choice,
                chosen_choice=chosen_choice,
                success=success,
                option_origins=[record["origin"] for record in option_records],
                blind_prompt_user=user_prompt,
                judge_response=raw_output,
            )
        )
        llm_calls.append(
            {
                "trial_index": trial_index,
                "messages": messages,
                "raw_output": raw_output,
                "correct_choice": correct_choice,
                "chosen_choice": chosen_choice,
                "success": success,
                "usage": usage_counts,
                "response_debug": response_debug,
            }
        )

    successes = sum(1 for item in trial_results if item.success)
    score_result = InterventionBlindScore(
        explanation=explanation,
        score=successes / trials,
        successes=successes,
        trials=trials,
        trial_results=trial_results,
    )
    return score_result, token_counter.as_dict(), llm_calls


def _write_markdown(path: Path, *, payload: Dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    score = payload.get("score", {})
    trials = score.get("trial_results", [])

    lines: List[str] = []
    lines.append("# Output-side Blind Evaluation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- generated_at: {metadata.get('generated_at')}")
    lines.append(f"- sae_name: {metadata.get('sae_name')}")
    lines.append(f"- layer_id: {metadata.get('layer_id')}")
    lines.append(f"- feature_id: {metadata.get('feature_id')}")
    lines.append(f"- intervention_source: {metadata.get('intervention_source')}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("```text")
    lines.append(str(payload.get("explanation", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Scores")
    lines.append(f"- score_blind_accuracy: {score.get('score')}")
    lines.append(f"- blind_judge_successes: {score.get('successes')}")
    lines.append(f"- blind_judge_trials: {score.get('trials')}")
    lines.append("")
    lines.append("## Token Usage")
    usage = payload.get("token_usage", {})
    lines.append(f"- prompt_tokens: {usage.get('prompt_tokens', 0)}")
    lines.append(f"- completion_tokens: {usage.get('completion_tokens', 0)}")
    lines.append(f"- total_tokens: {usage.get('total_tokens', 0)}")
    lines.append("")
    lines.append("## Intervention And Controls")
    lines.append(f"- intervention_result_path: {payload.get('intervention_result_path')}")
    lines.append("- control_result_paths:")
    for control_path in payload.get("control_result_paths", []):
        lines.append(f"  - {control_path}")
    lines.append("")
    lines.append("## Trial Results")
    lines.append("| trial | correct_choice | chosen_choice | success |")
    lines.append("| ---: | ---: | ---: | --- |")
    for trial in trials:
        lines.append(
            f"| {trial.get('trial_index')} | {trial.get('correct_choice')} | {trial.get('chosen_choice')} | {trial.get('success')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_intervention_blind(
    *,
    layer_id: str,
    feature_id: int,
    sae_name: str,
    output_root: Path,
    explanation: Optional[str],
    intervention_result_text: Optional[str],
    intervention_result_path: Optional[Path],
    control_result_paths: Sequence[Path],
    control_result_texts: Sequence[str],
    prefer_existing: bool,
    use_checkpoint: bool,
    model_checkpoint_path: str,
    width: str,
    sae_release: Optional[str],
    sae_average_l0: Optional[str],
    sae_canonical_map: Path,
    device: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    generation_temperature: float,
    kl_values: Sequence[float],
    api_key: Optional[str],
    api_key_file: Optional[str],
    openai_model: str,
    openai_base_url: str,
    trials: int,
    seed: int,
    num_choices: int,
    judge_temperature: float,
    judge_max_tokens: int,
    json_filename: str,
    md_filename: str,
) -> Dict[str, Any]:
    target_dir = _resolve_target_dir(
        output_root=output_root,
        sae_name=sae_name,
        layer_id=layer_id,
        feature_id=feature_id,
    )
    resolved_intervention_path: Optional[Path] = intervention_result_path
    resolved_intervention_text = intervention_result_text.strip() if intervention_result_text else ""
    intervention_source = "direct_text"

    if not resolved_intervention_text and resolved_intervention_path is not None:
        if not resolved_intervention_path.exists():
            raise FileNotFoundError(f"Cannot find intervention result file: {resolved_intervention_path}")
        resolved_intervention_text = resolved_intervention_path.read_text(encoding="utf-8").strip()
        intervention_source = "file"

    if not resolved_intervention_text and prefer_existing:
        cached_path = target_dir / "intervention_output.txt"
        if cached_path.exists():
            resolved_intervention_text = cached_path.read_text(encoding="utf-8").strip()
            resolved_intervention_path = cached_path
            intervention_source = "cached_output_file"

    if not resolved_intervention_text and use_checkpoint:
        resolved_intervention_text = build_intervention_result_from_checkpoint(
            model_checkpoint_path=model_checkpoint_path,
            layer_id=layer_id,
            width=width,
            feature_id=feature_id,
            sae_name=sae_name,
            sae_release=sae_release,
            sae_average_l0=sae_average_l0,
            sae_canonical_map=sae_canonical_map,
            device=device,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            generation_temperature=generation_temperature,
            kl_values=kl_values,
        )
        intervention_source = "checkpoint_generation"

    if not resolved_intervention_text:
        raise ValueError(
            "Missing intervention result. Provide --intervention-file/--intervention-text, "
            "or use --prefer-existing with cached outputs, or set --use-checkpoint."
        )

    resolved_control_paths = [Path(p) for p in control_result_paths]
    resolved_control_texts = [_normalize_result_text(text) for text in control_result_texts if text.strip()]

    for control_path in resolved_control_paths:
        if not control_path.exists():
            raise FileNotFoundError(f"Cannot find control result file: {control_path}")
        resolved_control_texts.append(control_path.read_text(encoding="utf-8").strip())

    if not resolved_control_texts and prefer_existing:
        discovered_paths = _discover_control_result_paths(
            output_root=output_root,
            sae_name=sae_name,
            layer_id=layer_id,
            feature_id=feature_id,
            num_required=max(1, num_choices - 1),
        )
        resolved_control_paths.extend(discovered_paths)
        for discovered in discovered_paths:
            resolved_control_texts.append(discovered.read_text(encoding="utf-8").strip())

    if not resolved_control_texts:
        raise ValueError(
            "No control results available. Provide --control-files or ensure cached outputs exist under outputs/{sae}/layer-{layer}."
        )

    resolved_explanation = (explanation or "").strip() or generate_explanation_from_result(resolved_intervention_text)

    key = _resolve_api_key(api_key=api_key, api_key_file=api_key_file)
    if not key:
        raise ValueError("Missing API key: set --api-key or --api-key-file.")
    client = OpenAI(base_url=openai_base_url, api_key=key)

    score_result, token_usage, llm_calls = score_intervention_explanation(
        explanation=resolved_explanation,
        intervention_result=resolved_intervention_text,
        control_results=resolved_control_texts,
        client=client,
        llm_model=openai_model,
        trials=trials,
        seed=seed,
        num_choices=num_choices,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": now,
            "sae_name": sae_name,
            "layer_id": layer_id,
            "feature_id": feature_id,
            "intervention_source": intervention_source,
            "num_choices": num_choices,
            "seed": seed,
            "llm_model": openai_model,
        },
        "explanation": resolved_explanation,
        "intervention_result_path": str(resolved_intervention_path) if resolved_intervention_path is not None else None,
        "control_result_paths": [str(path) for path in resolved_control_paths],
        "score": asdict(score_result),
        "token_usage": token_usage,
        "llm_calls": llm_calls,
    }

    json_path = target_dir / json_filename
    md_path = target_dir / md_filename
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload=payload)
    payload["output_paths"] = {
        "output_dir": str(target_dir),
        "json_file": str(json_path),
        "md_file": str(md_path),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind-test scoring for output-side SAE explanations.")
    parser.add_argument("--layer-id", required=True, help="SAE layer id")
    parser.add_argument("--feature-id", required=True, type=int, help="Feature id")
    parser.add_argument("--width", default="16k", help="SAE width")
    parser.add_argument("--sae-name", default="gemmascope-res")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--json-filename", default="intervention_blind_score.json")
    parser.add_argument("--md-filename", default="intervention_blind_score.md")

    parser.add_argument("--explanation", default=None, help="Hypothesis text")
    parser.add_argument("--intervention-text", default=None, help="Intervention result text")
    parser.add_argument("--intervention-file", default=None, help="Path to intervention_output.txt")
    parser.add_argument("--control-files", nargs="*", default=None, help="Optional control intervention result files")

    parser.add_argument(
        "--prefer-existing",
        dest="prefer_existing",
        action="store_true",
        help="Prefer cached outputs under outputs/{sae}/layer-{layer}/feature-{id}.",
    )
    parser.add_argument(
        "--no-prefer-existing",
        dest="prefer_existing",
        action="store_false",
        help="Do not read cached outputs before other intervention sources.",
    )
    parser.add_argument("--use-checkpoint", action="store_true", help="Generate intervention result from checkpoint when cache is missing.")
    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument(
        "--sae-canonical-map",
        default=str(PROJECT_ROOT / DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt for average_l0 resolution.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--generation-temperature", type=float, default=0.75)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(OUTPUT_SIDE_PLACEHOLDER),
        help="Prompts used for checkpoint intervention generation.",
    )
    parser.add_argument(
        "--kl-values",
        type=float,
        nargs="*",
        default=[float(x) for x in KL_DIV_VALUES],
        help="KL values used in checkpoint intervention generation.",
    )

    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-choices", type=int, default=3)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=10000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--openai-model", default="zai-org/glm-4.7")
    parser.add_argument("--openai-base-url", default="https://api.ppio.com/openai")
    parser.set_defaults(prefer_existing=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_intervention_blind(
        layer_id=str(args.layer_id),
        feature_id=int(args.feature_id),
        sae_name=str(args.sae_name),
        output_root=Path(args.output_root),
        explanation=args.explanation,
        intervention_result_text=args.intervention_text,
        intervention_result_path=Path(args.intervention_file) if args.intervention_file else None,
        control_result_paths=[Path(x) for x in (args.control_files or [])],
        control_result_texts=[],
        prefer_existing=bool(args.prefer_existing),
        use_checkpoint=bool(args.use_checkpoint),
        model_checkpoint_path=str(args.model_checkpoint_path),
        width=str(args.width),
        sae_release=args.sae_release,
        sae_average_l0=args.sae_average_l0,
        sae_canonical_map=Path(args.sae_canonical_map),
        device=str(args.device),
        prompts=[str(x) for x in args.prompts],
        max_new_tokens=int(args.max_new_tokens),
        generation_temperature=float(args.generation_temperature),
        kl_values=[float(x) for x in args.kl_values],
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        openai_model=str(args.openai_model),
        openai_base_url=str(args.openai_base_url),
        trials=int(args.trials),
        seed=int(args.seed),
        num_choices=int(args.num_choices),
        judge_temperature=float(args.judge_temperature),
        judge_max_tokens=int(args.judge_max_tokens),
        json_filename=str(args.json_filename),
        md_filename=str(args.md_filename),
    )
    print(
        json.dumps(
            {
                "output_paths": payload.get("output_paths", {}),
                "score": {
                    "score_blind_accuracy": payload.get("score", {}).get("score"),
                    "blind_judge_successes": payload.get("score", {}).get("successes"),
                    "blind_judge_trials": payload.get("score", {}).get("trials"),
                },
                "intervention_result_path": payload.get("intervention_result_path"),
                "control_result_paths": payload.get("control_result_paths"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
