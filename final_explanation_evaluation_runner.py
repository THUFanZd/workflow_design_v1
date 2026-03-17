from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _resolve_workflow_final_result_path(workflow_path: Path) -> Path:
    if workflow_path.is_file():
        return workflow_path
    if not workflow_path.is_dir():
        raise FileNotFoundError(f"Workflow path does not exist: {workflow_path}")

    final_dir = workflow_path / "final_result"
    candidates = sorted(final_dir.glob("*-final-result.json")) if final_dir.exists() else []
    if not candidates:
        candidates = sorted(workflow_path.glob("**/*-final-result.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find '*-final-result.json' under workflow path: {workflow_path}"
        )
    return candidates[-1]


def _resolve_workflow_path_from_args(args: argparse.Namespace) -> Path:
    if args.workflow_path:
        return Path(str(args.workflow_path))

    missing: List[str] = []
    if args.layer_id is None:
        missing.append("--layer-id")
    if args.feature_id is None:
        missing.append("--feature-id")
    if args.timestamp is None:
        missing.append("--timestamp")
    if missing:
        raise ValueError(
            "Missing workflow locator arguments. Provide --workflow-path, or provide all of: "
            + ", ".join(missing)
        )

    return (
        Path(str(args.logs_root))
        / f"{str(args.layer_id).strip()}_{str(args.feature_id).strip()}"
        / str(args.timestamp).strip()
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_candidate_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def _normalize_cached_input_hypothesis(
    cached: Dict[str, Any],
    *,
    source: str,
) -> Optional[Dict[str, Any]]:
    hypothesis = str(cached.get("hypothesis", "")).strip()
    if not hypothesis:
        return None
    score_non_zero_raw = cached.get("score_non_zero_rate")
    score_boundary_raw = cached.get("score_boundary_non_activation_rate")
    score_non_zero = _safe_float(score_non_zero_raw, 0.0) if score_non_zero_raw is not None else None
    score_boundary = _safe_float(score_boundary_raw, 0.0) if score_boundary_raw is not None else None
    combined_raw = cached.get("combined_score")
    combined_score = (
        _safe_float(combined_raw, 0.0)
        if combined_raw is not None
        else _safe_float(score_non_zero, 0.0) + _safe_float(score_boundary, 0.0)
    )
    return {
        "source": source,
        "hypothesis_index": _safe_int(cached.get("hypothesis_index"), 0),
        "hypothesis": hypothesis,
        "round_index": _safe_int(cached.get("round_index"), 0),
        "round_id": str(cached.get("round_id", "")).strip() or None,
        "score_name": "combined_input_score",
        "score_value": combined_score,
        "score_non_zero_rate": score_non_zero,
        "score_boundary_non_activation_rate": score_boundary,
        "combined_score": combined_score,
    }


def _load_input_hypothesis_cache(
    *,
    final_result_payload: Dict[str, Any],
    workflow_timestamp_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    candidate_paths: List[Path] = []
    configured_path = str(final_result_payload.get("input_side_hypothesis_cache_path", "")).strip()
    if configured_path:
        candidate_paths.append(_resolve_candidate_path(configured_path, base_dir=PROJECT_ROOT))

    layer_id = str(final_result_payload.get("layer_id", "")).strip()
    feature_id = str(final_result_payload.get("feature_id", "")).strip()
    if layer_id and feature_id:
        candidate_paths.append(
            workflow_timestamp_dir / f"layer{layer_id}-feature{feature_id}-input-side-hypotheses-cache.json"
        )

    seen: set[str] = set()
    deduped_paths: List[Path] = []
    for candidate in candidate_paths:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_paths.append(candidate)

    for path in deduped_paths:
        if not path.exists():
            continue
        payload = _load_json(path)
        if isinstance(payload, dict):
            return payload, path
    return None, None


def _pick_best_input_hypothesis_from_workflow(
    *,
    final_result_payload: Dict[str, Any],
    workflow_timestamp_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Path]]:
    best_raw = final_result_payload.get("input_side_best_hypothesis")
    if isinstance(best_raw, dict):
        normalized = _normalize_cached_input_hypothesis(
            best_raw,
            source="workflow_final_result.input_side_best_hypothesis",
        )
        if normalized is not None:
            return normalized, best_raw, None

    cache_payload, cache_path = _load_input_hypothesis_cache(
        final_result_payload=final_result_payload,
        workflow_timestamp_dir=workflow_timestamp_dir,
    )
    if isinstance(cache_payload, dict):
        cache_best = cache_payload.get("best_hypothesis")
        if isinstance(cache_best, dict):
            normalized = _normalize_cached_input_hypothesis(
                cache_best,
                source="workflow_input_side_cache.best_hypothesis",
            )
            if normalized is not None:
                return normalized, cache_best, cache_path
    return None, None, cache_path


def _pick_best_hypothesis_from_refined(
    refined_payload: Dict[str, Any],
    *,
    side: str,
) -> Optional[Dict[str, Any]]:
    refined_section = refined_payload.get("refined_hypotheses", {})
    if not isinstance(refined_section, dict):
        return None
    items_raw = refined_section.get(side, [])
    items = [item for item in items_raw if isinstance(item, dict)]
    if not items:
        return None
    items.sort(
        key=lambda item: (
            -_safe_float(item.get("score_value"), 0.0),
            _safe_int(item.get("hypothesis_index"), 10**9),
        )
    )
    best = items[0]
    return {
        "source": "refined_hypotheses",
        "hypothesis_index": _safe_int(best.get("hypothesis_index"), 0),
        "hypothesis": str(best.get("refined_hypothesis", "")).strip()
        or str(best.get("original_hypothesis", "")).strip(),
        "score_name": str(best.get("score_name", "")).strip(),
        "score_value": _safe_float(best.get("score_value"), 0.0),
    }


def _pick_best_hypothesis_from_execution(
    execution_payload: Dict[str, Any],
    *,
    side: str,
) -> Optional[Dict[str, Any]]:
    if side == "input":
        section = execution_payload.get("input_side_execution", {})
    else:
        section = execution_payload.get("output_side_execution", {})
    if not isinstance(section, dict):
        return None
    if side == "input":
        score_name = "score_non_zero_rate"
    else:
        score_name = str(section.get("output_score_name", "score_blind_accuracy")).strip() or "score_blind_accuracy"
    results_raw = section.get("hypothesis_results", [])
    results = [item for item in results_raw if isinstance(item, dict)]
    if not results:
        return None
    def _item_score(item: Dict[str, Any]) -> float:
        return _safe_float(item.get(score_name, item.get("score_primary", item.get("score", 0.0))), 0.0)

    results.sort(
        key=lambda item: (
            -_item_score(item),
            _safe_int(item.get("hypothesis_index"), 10**9),
        )
    )
    best = results[0]
    return {
        "source": "experiments_execution",
        "hypothesis_index": _safe_int(best.get("hypothesis_index"), 0),
        "hypothesis": str(best.get("hypothesis", "")).strip(),
        "score_name": score_name,
        "score_value": _item_score(best),
    }


def _pick_best_hypothesis_from_final_result(
    final_result_payload: Dict[str, Any],
    *,
    side: str,
) -> Optional[Dict[str, Any]]:
    key = "input_side_final_hypotheses" if side == "input" else "output_side_final_hypotheses"
    hypotheses_raw = final_result_payload.get(key, [])
    hypotheses = [str(item).strip() for item in hypotheses_raw if str(item).strip()]
    if not hypotheses:
        return None
    return {
        "source": "final_result_fallback",
        "hypothesis_index": 1,
        "hypothesis": hypotheses[0],
        "score_name": "unknown",
        "score_value": None,
    }


def _pick_best_hypothesis(
    *,
    final_result_payload: Dict[str, Any],
    refined_payload: Optional[Dict[str, Any]],
    execution_payload: Optional[Dict[str, Any]],
    side: str,
) -> Dict[str, Any]:
    if refined_payload is not None:
        result = _pick_best_hypothesis_from_refined(refined_payload, side=side)
        if result is not None:
            return result
    if execution_payload is not None:
        result = _pick_best_hypothesis_from_execution(execution_payload, side=side)
        if result is not None:
            return result
    result = _pick_best_hypothesis_from_final_result(final_result_payload, side=side)
    if result is not None:
        return result
    raise ValueError(f"No {side}-side hypothesis found in workflow artifacts.")


def _run_command(cmd: Sequence[str], *, cwd: Path) -> None:
    process = subprocess.run(list(cmd), cwd=str(cwd), text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit={process.returncode}): {' '.join(cmd)}")


def _log_progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [final-eval] {message}", flush=True)


def _run_command_with_progress(
    cmd: Sequence[str],
    *,
    cwd: Path,
    step_name: str,
    heartbeat_seconds: int = 60,
) -> None:
    _log_progress(f"Start step: {step_name}")
    _log_progress(f"Command: {' '.join(cmd)}")

    started_at = time.monotonic()
    process = subprocess.Popen(list(cmd), cwd=str(cwd), text=True)
    last_heartbeat = started_at

    while True:
        return_code = process.poll()
        if return_code is not None:
            elapsed = int(time.monotonic() - started_at)
            if return_code != 0:
                raise RuntimeError(
                    f"Step failed: {step_name} (exit={return_code}, elapsed={elapsed}s). "
                    f"Command: {' '.join(cmd)}"
                )
            _log_progress(f"Finished step: {step_name} (elapsed={elapsed}s)")
            return

        now = time.monotonic()
        if heartbeat_seconds > 0 and (now - last_heartbeat) >= heartbeat_seconds:
            elapsed = int(now - started_at)
            _log_progress(f"Still running: {step_name} (elapsed={elapsed}s)")
            last_heartbeat = now
        time.sleep(1.0)


def _build_source(layer_id: str, sae_name: str, width: str) -> str:
    return f"{layer_id}-{sae_name}-{width}"


def _extract_logit_summary(logit_payload: Dict[str, Any]) -> Dict[str, Any]:
    runs = [item for item in logit_payload.get("runs", []) if isinstance(item, dict)]
    positive = [
        _safe_float(item.get("scores", {}).get("positive_topk_increase_ratio"), 0.0)
        for item in runs
    ]
    negative = [
        _safe_float(item.get("scores", {}).get("negative_topk_decrease_ratio"), 0.0)
        for item in runs
    ]
    return {
        "run_count": len(runs),
        "mean_positive_topk_increase_ratio": (sum(positive) / len(positive)) if positive else None,
        "mean_negative_topk_decrease_ratio": (sum(negative) / len(negative)) if negative else None,
    }


def _write_summary_markdown(path: Path, *, payload: Dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    selected = payload.get("selected_hypotheses", {})
    input_eval = payload.get("input_evaluation", {})
    output_eval = payload.get("output_evaluation", {})

    lines: List[str] = []
    lines.append("# Final Explanation Evaluation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- generated_at: {metadata.get('generated_at')}")
    lines.append(f"- model_id: {metadata.get('model_id')}")
    lines.append(f"- layer_id: {metadata.get('layer_id')}")
    lines.append(f"- feature_id: {metadata.get('feature_id')}")
    lines.append(f"- evaluation_timestamp: {metadata.get('evaluation_timestamp')}")
    lines.append(f"- run_mode: {metadata.get('run_mode')}")
    lines.append(f"- workflow_final_result: {metadata.get('workflow_final_result_path')}")
    lines.append("")
    lines.append("## Selected Input Hypothesis")
    lines.append(f"- source: {selected.get('input', {}).get('source')}")
    lines.append(f"- score_name: {selected.get('input', {}).get('score_name')}")
    lines.append(f"- score_value: {selected.get('input', {}).get('score_value')}")
    lines.append("```text")
    lines.append(str(selected.get("input", {}).get("hypothesis", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Selected Output Hypothesis")
    lines.append(f"- source: {selected.get('output', {}).get('source')}")
    lines.append(f"- score_name: {selected.get('output', {}).get('score_name')}")
    lines.append(f"- score_value: {selected.get('output', {}).get('score_value')}")
    lines.append("```text")
    lines.append(str(selected.get("output", {}).get("hypothesis", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Input-side Metrics")
    lines.append(f"- status: {input_eval.get('status')}")
    lines.append(f"- used_workflow_cached_scores: {input_eval.get('used_workflow_cached_scores')}")
    lines.append(f"- relative_quality_score: {input_eval.get('relative_quality_score')}")
    lines.append(f"- adherence: {input_eval.get('adherence')}")
    lines.append(f"- non_activation_relative_quality_score: {input_eval.get('non_activation_relative_quality_score')}")
    lines.append(f"- non_activation_adherence: {input_eval.get('non_activation_adherence')}")
    lines.append(f"- boundary_relative_quality_score: {input_eval.get('boundary_relative_quality_score')}")
    lines.append(f"- score_non_zero_rate: {input_eval.get('score_non_zero_rate')}")
    lines.append(f"- score_boundary_non_activation_rate: {input_eval.get('score_boundary_non_activation_rate')}")
    lines.append(f"- combined_input_score: {input_eval.get('combined_input_score')}")
    lines.append("")
    lines.append("## Output-side Metrics")
    lines.append(f"- mode: {output_eval.get('mode')}")
    lines.append(f"- score_summary: {json.dumps(output_eval.get('summary', {}), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Paths")
    lines.append(f"- input_result_json: {payload.get('paths', {}).get('input_result_json')}")
    lines.append(f"- output_result_json: {payload.get('paths', {}).get('output_result_json')}")
    lines.append(f"- final_summary_json: {payload.get('paths', {}).get('final_summary_json')}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read workflow_runner outputs from logs, select top-scoring final input/output hypotheses, "
            "run final evaluation scripts, and save summary metrics."
        )
    )
    parser.add_argument(
        "--workflow-path",
        default=None,
        help=(
            "Path to workflow final-result json file, or workflow timestamp directory. "
            "If omitted, path is composed as logs/{layer-id}_{feature-id}/{timestamp}."
        ),
    )
    parser.add_argument("--layer-id", default=None, help="Layer id used to compose workflow path.")
    parser.add_argument("--feature-id", type=int, default=None, help="Feature id used to compose workflow path.")
    parser.add_argument("--timestamp", default=None, help="Workflow timestamp directory name used to compose workflow path.")
    parser.add_argument("--logs-root", default=str(PROJECT_ROOT / "logs"), help="Logs root directory for composed workflow path.")
    parser.add_argument("--width", default="16k", help="SAE width used for source string.")
    parser.add_argument("--sae-name", default="gemmascope-res", help="SAE name used by evaluation outputs.")
    parser.add_argument(
        "--input-output-root",
        default=str(
            PROJECT_ROOT / "explanation_quality_evaluation" / "input-side-evaluation" / "outputs"
        ),
    )
    parser.add_argument(
        "--output-output-root",
        default=str(
            PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"
        ),
    )

    parser.add_argument("--input-max-explanations", type=int, default=3)
    parser.add_argument("--input-selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--input-m", type=int, default=5)
    parser.add_argument("--input-n", type=int, default=5)
    parser.add_argument("--input-non-activation-context-count", type=int, default=5)
    parser.add_argument("--input-llm-model", default="zai-org/glm-4.7")
    parser.add_argument("--input-ppio-base-url", default="https://api.ppio.com/openai")
    parser.add_argument("--input-ppio-api-key-file", default=None)
    parser.add_argument("--input-disable-boundary-score", action="store_true")
    parser.add_argument(
        "--force-run-input-eval",
        action="store_true",
        help="Deprecated no-op. Input-side Neuronpedia comparison now always runs when run-mode includes input.",
    )
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--sae-average-l0", default=None)
    parser.add_argument("--sae-canonical-map", default=str(PROJECT_ROOT / "support_info" / "canonical_map.txt"))
    parser.add_argument("--sae-device", default="auto")

    parser.add_argument("--output-eval-mode", choices=["blind", "logit"], default="blind")
    parser.add_argument("--output-api-key-file", default=None)
    parser.add_argument("--output-openai-model", default="zai-org/glm-4.7")
    parser.add_argument("--output-openai-base-url", default="https://api.ppio.com/openai")
    parser.add_argument(
        "--run-mode",
        choices=["both", "input", "output", "none"],
        default="both",
        help="Control which side(s) to run. Default is both.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Progress heartbeat interval in seconds while a child process is running.",
    )

    parser.add_argument("--blind-trials", type=int, default=1)
    parser.add_argument("--blind-seed", type=int, default=42)
    parser.add_argument("--blind-num-choices", type=int, default=3)
    parser.add_argument(
        "--blind-use-checkpoint-fallback",
        dest="blind_use_checkpoint_fallback",
        action="store_true",
        help="If cached intervention_output is missing, allow checkpoint generation fallback for blind evaluation.",
    )
    parser.add_argument(
        "--no-blind-checkpoint-fallback",
        dest="blind_use_checkpoint_fallback",
        action="store_false",
        help="Do not generate intervention_output from checkpoint when cache is missing.",
    )
    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--logit-top-k", type=int, default=5)
    parser.add_argument("--logit-target-kl", type=float, nargs="*", default=[0.25, 0.5, -0.25, -0.5])
    parser.add_argument("--logit-judge-max-tokens", type=int, default=10000)
    parser.set_defaults(blind_use_checkpoint_fallback=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log_progress("Resolving workflow artifacts")
    workflow_path = _resolve_workflow_path_from_args(args)
    final_result_path = _resolve_workflow_final_result_path(workflow_path)
    final_result = _load_json(final_result_path)

    layer_id = str(final_result.get("layer_id", "")).strip()
    feature_id_str = str(final_result.get("feature_id", "")).strip()
    model_id = str(final_result.get("model_id", "gemma-2-2b")).strip()
    executed_rounds = _safe_int(final_result.get("executed_rounds"), 0)
    if not layer_id or not feature_id_str:
        raise ValueError(f"Invalid workflow final result metadata: {final_result_path}")
    feature_id = int(feature_id_str)
    if args.layer_id is not None and str(args.layer_id).strip() != layer_id:
        raise ValueError(
            f"Layer id mismatch: args.layer_id={args.layer_id} vs final_result.layer_id={layer_id}"
        )
    if args.feature_id is not None and int(args.feature_id) != feature_id:
        raise ValueError(
            f"Feature id mismatch: args.feature_id={args.feature_id} vs final_result.feature_id={feature_id}"
        )

    workflow_timestamp_dir = final_result_path.parents[1]
    evaluation_timestamp = str(args.timestamp).strip() if args.timestamp else workflow_timestamp_dir.name
    if not str(args.timestamp or "").strip():
        _log_progress(
            "No --timestamp provided; using workflow directory name as evaluation timestamp: "
            f"{evaluation_timestamp}"
        )

    refined_payload: Optional[Dict[str, Any]] = None
    execution_payload: Optional[Dict[str, Any]] = None
    target_round = executed_rounds if executed_rounds > 0 else 0
    round_dir = workflow_timestamp_dir / f"round_{target_round}"
    if executed_rounds > 0:
        refined_path = round_dir / f"layer{layer_id}-feature{feature_id}-refined-hypotheses.json"
        if refined_path.exists():
            refined_payload = _load_json(refined_path)
    execution_path = round_dir / f"layer{layer_id}-feature{feature_id}-experiments-execution.json"
    if execution_path.exists():
        execution_payload = _load_json(execution_path)

    selected_input_from_workflow, selected_input_cache_raw, input_cache_path = _pick_best_input_hypothesis_from_workflow(
        final_result_payload=final_result,
        workflow_timestamp_dir=workflow_timestamp_dir,
    )
    if selected_input_from_workflow is not None:
        selected_input = selected_input_from_workflow
    else:
        selected_input = _pick_best_hypothesis(
            final_result_payload=final_result,
            refined_payload=refined_payload,
            execution_payload=execution_payload,
            side="input",
        )
    selected_output = _pick_best_hypothesis(
        final_result_payload=final_result,
        refined_payload=refined_payload,
        execution_payload=execution_payload,
        side="output",
    )
    _log_progress(
        f"Selected hypotheses (input idx={selected_input.get('hypothesis_index')}, "
        f"output idx={selected_output.get('hypothesis_index')})"
    )

    source = _build_source(layer_id=layer_id, sae_name=str(args.sae_name), width=str(args.width))
    input_script = (
        PROJECT_ROOT
        / "explanation_quality_evaluation"
        / "input-side-evaluation"
        / "compare_explanations_with_llm.py"
    )
    output_blind_script = (
        PROJECT_ROOT
        / "explanation_quality_evaluation"
        / "output-side-evaluation"
        / "intervention_blind_score.py"
    )
    output_logit_script = (
        PROJECT_ROOT
        / "explanation_quality_evaluation"
        / "output-side-evaluation"
        / "intervention_logit_topk_score.py"
    )

    run_input = args.run_mode in ("both", "input")
    run_output = args.run_mode in ("both", "output")
    if args.run_mode == "none":
        _log_progress("Run mode is 'none': skip input-side and output-side evaluation execution.")

    input_result_path: Optional[Path] = None
    input_result: Dict[str, Any] = {}
    input_eval_status = "skipped"
    used_cached_input_scores = False
    if run_input:
        if selected_input_cache_raw is not None:
            _log_progress(
                "Workflow cached input-side scores detected, but Neuronpedia comparison will still run."
            )
        input_cmd: List[str] = [
            sys.executable,
            str(input_script),
            "--model-id",
            model_id,
            "--layer-id",
            layer_id,
            "--width",
            str(args.width),
            "--source",
            source,
            "--feature-id",
            str(feature_id),
            "--my-explanation",
            str(selected_input["hypothesis"]),
            "--max-explanations",
            str(args.input_max_explanations),
            "--selection-method",
            str(args.input_selection_method),
            "--m",
            str(args.input_m),
            "--n",
            str(args.input_n),
            "--non-activation-context-count",
            str(args.input_non_activation_context_count),
            "--llm-model",
            str(args.input_llm_model),
            "--ppio-base-url",
            str(args.input_ppio_base_url),
            "--sae-name",
            str(args.sae_name),
            "--sae-device",
            str(args.sae_device),
            "--output-root",
            str(args.input_output_root),
            "--timestamp",
            evaluation_timestamp,
        ]
        if args.input_ppio_api_key_file:
            input_cmd.extend(["--ppio-api-key-file", str(args.input_ppio_api_key_file)])
        if args.sae_release:
            input_cmd.extend(["--sae-release", str(args.sae_release)])
        if args.sae_average_l0:
            input_cmd.extend(["--sae-average-l0", str(args.sae_average_l0)])
        if args.sae_canonical_map:
            input_cmd.extend(["--sae-canonical-map", str(args.sae_canonical_map)])
        if args.input_disable_boundary_score:
            input_cmd.append("--disable-boundary-score")

        _run_command_with_progress(
            input_cmd,
            cwd=PROJECT_ROOT,
            step_name="input-side evaluation",
            heartbeat_seconds=int(args.heartbeat_seconds),
        )
        input_result_path = (
            Path(args.input_output_root)
            / str(args.sae_name)
            / f"layer-{layer_id}"
            / f"feature-{feature_id}"
            / evaluation_timestamp
            / "result.json"
        )
        input_result = _load_json(input_result_path)
        input_eval_status = "completed"
    else:
        _log_progress("Skip input-side evaluation.")

    output_result_path: Optional[Path] = None
    output_payload: Dict[str, Any] = {}
    output_summary: Dict[str, Any] = {}
    if run_output:
        if args.output_eval_mode == "blind":
            output_cmd: List[str] = [
                sys.executable,
                str(output_blind_script),
                "--layer-id",
                layer_id,
                "--feature-id",
                str(feature_id),
                "--width",
                str(args.width),
                "--sae-name",
                str(args.sae_name),
                "--output-root",
                str(args.output_output_root),
                "--timestamp",
                evaluation_timestamp,
                "--explanation",
                str(selected_output["hypothesis"]),
                "--prefer-existing",
                "--trials",
                str(args.blind_trials),
                "--seed",
                str(args.blind_seed),
                "--num-choices",
                str(args.blind_num_choices),
                "--openai-model",
                str(args.output_openai_model),
                "--openai-base-url",
                str(args.output_openai_base_url),
                "--device",
                str(args.device),
                "--model-checkpoint-path",
                str(args.model_checkpoint_path),
            ]
            if args.output_api_key_file:
                output_cmd.extend(["--api-key-file", str(args.output_api_key_file)])
            if args.sae_release:
                output_cmd.extend(["--sae-release", str(args.sae_release)])
            if args.sae_average_l0:
                output_cmd.extend(["--sae-average-l0", str(args.sae_average_l0)])
            if args.sae_canonical_map:
                output_cmd.extend(["--sae-canonical-map", str(args.sae_canonical_map)])
            if args.blind_use_checkpoint_fallback:
                output_cmd.append("--use-checkpoint")
            _run_command_with_progress(
                output_cmd,
                cwd=PROJECT_ROOT,
                step_name="output-side evaluation (blind)",
                heartbeat_seconds=int(args.heartbeat_seconds),
            )
            output_result_path = (
                Path(args.output_output_root)
                / str(args.sae_name)
                / f"layer-{layer_id}"
                / f"feature-{feature_id}"
                / evaluation_timestamp
                / "intervention_blind_score.json"
            )
            output_payload = _load_json(output_result_path)
            output_summary = {
                "score_blind_accuracy": output_payload.get("score", {}).get("score"),
                "blind_judge_successes": output_payload.get("score", {}).get("successes"),
                "blind_judge_trials": output_payload.get("score", {}).get("trials"),
            }
        else:
            output_cmd = [
                sys.executable,
                str(output_logit_script),
                "--layer-id",
                layer_id,
                "--feature-id",
                str(feature_id),
                "--width",
                str(args.width),
                "--sae-name",
                str(args.sae_name),
                "--output-root",
                str(args.output_output_root),
                "--timestamp",
                evaluation_timestamp,
                "--explanation",
                str(selected_output["hypothesis"]),
                "--top-k",
                str(args.logit_top_k),
                "--judge-max-tokens",
                str(args.logit_judge_max_tokens),
                "--openai-model",
                str(args.output_openai_model),
                "--openai-base-url",
                str(args.output_openai_base_url),
                "--prefer-existing",
            ]
            output_cmd.extend(["--target-kl", *[str(float(kl)) for kl in args.logit_target_kl]])
            if args.output_api_key_file:
                output_cmd.extend(["--api-key-file", str(args.output_api_key_file)])
            if args.sae_release:
                output_cmd.extend(["--sae-release", str(args.sae_release)])
            if args.sae_average_l0:
                output_cmd.extend(["--sae-average-l0", str(args.sae_average_l0)])
            if args.sae_canonical_map:
                output_cmd.extend(["--sae-canonical-map", str(args.sae_canonical_map)])
            _run_command_with_progress(
                output_cmd,
                cwd=PROJECT_ROOT,
                step_name="output-side evaluation (logit)",
                heartbeat_seconds=int(args.heartbeat_seconds),
            )
            output_result_path = (
                Path(args.output_output_root)
                / str(args.sae_name)
                / f"layer-{layer_id}"
                / f"feature-{feature_id}"
                / evaluation_timestamp
                / "intervention_logit_topk_score.json"
            )
            output_payload = _load_json(output_result_path)
            output_summary = _extract_logit_summary(output_payload)
    else:
        _log_progress("Skip output-side evaluation.")

    final_summary_path = (
        workflow_timestamp_dir
        / "final_result"
        / f"layer{layer_id}-feature{feature_id}-final-evaluation.json"
    )
    final_summary_md_path = final_summary_path.with_suffix(".md")
    _log_progress("Building final merged summary payload")

    summary_payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_id": model_id,
            "layer_id": layer_id,
            "feature_id": feature_id,
            "evaluation_timestamp": evaluation_timestamp,
            "run_mode": str(args.run_mode),
            "workflow_final_result_path": str(final_result_path),
        },
        "selected_hypotheses": {
            "input": selected_input,
            "output": selected_output,
        },
        "input_evaluation": {
            "status": input_eval_status,
            "used_workflow_cached_scores": used_cached_input_scores,
            "relative_quality_score": input_result.get("relative_quality_score") if run_input else None,
            "adherence": input_result.get("adherence") if run_input else None,
            "non_activation_relative_quality_score": (
                input_result.get("non_activation_relative_quality_score") if run_input else None
            ),
            "non_activation_adherence": (
                input_result.get("non_activation_adherence") if run_input else None
            ),
            "boundary_relative_quality_score": input_result.get("boundary_relative_quality_score") if run_input else None,
            "score_non_zero_rate": input_result.get("score_non_zero_rate") if run_input else None,
            "score_boundary_non_activation_rate": (
                input_result.get("score_boundary_non_activation_rate") if run_input else None
            ),
            "combined_input_score": input_result.get("combined_input_score") if run_input else None,
            "raw_result": (
                {
                    "input_result": input_result,
                    "selected_input_cache_entry": selected_input_cache_raw,
                }
                if run_input and used_cached_input_scores
                else (input_result if run_input else None)
            ),
        },
        "output_evaluation": {
            "mode": str(args.output_eval_mode),
            "status": "completed" if run_output else "skipped",
            "summary": output_summary if run_output else {},
            "raw_result": output_payload if run_output else None,
        },
        "paths": {
            "input_result_json": str(input_result_path) if input_result_path is not None else None,
            "output_result_json": str(output_result_path) if output_result_path is not None else None,
            "final_summary_json": str(final_summary_path),
            "final_summary_md": str(final_summary_md_path),
        },
    }
    _log_progress("Writing final summary json and markdown")
    final_summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(final_summary_md_path, payload=summary_payload)

    print(
        json.dumps(
            {
                "selected_input_hypothesis": selected_input,
                "selected_output_hypothesis": selected_output,
                "input_metrics": summary_payload["input_evaluation"],
                "output_metrics": summary_payload["output_evaluation"]["summary"],
                "summary_json": str(final_summary_path),
                "summary_md": str(final_summary_md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
