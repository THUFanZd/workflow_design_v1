from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
COMPARE_SCRIPT = SCRIPT_PATH.parent / "compare_explanations_with_llm.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_PATH.parent / "outputs"
DEFAULT_TARGET_ROOT = DEFAULT_OUTPUT_ROOT / "gemmascope-res"
DEFAULT_TEMP_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "_boundary_patch_tmp"

BOUNDARY_RESULT_FIELDS = [
    "num_boundary_contexts",
    "boundary_non_activation_rate",
    "boundary_activation_threshold",
    "boundary_details",
    "boundary_reference_mean_non_activation_rate",
    "boundary_relative_quality_score",
    "boundary_reference_details",
    "boundary_warning",
]

BOUNDARY_SCORE_FIELDS = [
    "boundary_non_activation_rate",
    "boundary_activation_threshold",
    "boundary_reference_mean_non_activation_rate",
    "boundary_relative_quality_score",
]


@dataclass
class RunTask:
    kind: str  # "full" or "boundary"
    layer_id: str
    feature_id: str
    source: str
    model_id: str
    run_dir: Path
    explanation: str


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _extract_layer_feature(run_dir: Path) -> Tuple[str, str]:
    feature_match = re.match(r"feature-(\d+)$", run_dir.parent.name)
    layer_match = re.match(r"layer-(\d+)$", run_dir.parent.parent.name)
    if not feature_match or not layer_match:
        raise ValueError(f"Unexpected run directory layout: {run_dir}")
    return layer_match.group(1), feature_match.group(1)


def _extract_explanation_from_llm_inout(llm_path: Path) -> str:
    text = llm_path.read_text(encoding="utf-8")
    match = re.search(r"Explanation:\r?\n(.*?)\r?\n\r?\nContext:\r?\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Cannot parse explanation from: {llm_path}")
    explanation = match.group(1).strip()
    if not explanation:
        raise ValueError(f"Parsed empty explanation from: {llm_path}")
    return explanation


def _boundary_fields_missing(result_payload: Dict[str, Any]) -> bool:
    detail = result_payload.get("boundary_details")
    ref_detail = result_payload.get("boundary_reference_details")
    num_ctx = result_payload.get("num_boundary_contexts")
    rate = result_payload.get("boundary_non_activation_rate")

    has_detail = isinstance(detail, list) and len(detail) > 0
    has_ref_detail = isinstance(ref_detail, list) and len(ref_detail) > 0
    has_num_ctx = isinstance(num_ctx, int) and num_ctx > 0
    has_rate = isinstance(rate, (int, float))

    return not (has_detail and has_ref_detail and has_num_ctx and has_rate)


def _resolve_model_and_source(feature_dir: Path, run_dir: Path, layer_id: str) -> Tuple[str, str]:
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = _load_json(result_path)
        feature = result.get("feature") or {}
        model_id = str(feature.get("model_id") or "").strip()
        source = str(feature.get("source") or "").strip()
        if model_id and source:
            return model_id, source

    eval_path = run_dir / "evaluation_record.json"
    if eval_path.exists():
        record = _load_json(eval_path)
        feature = record.get("feature") or {}
        model_id = str(feature.get("model_id") or "").strip()
        source = str(feature.get("source") or "").strip()
        if model_id and source:
            return model_id, source

    for cache_path in sorted(feature_dir.glob("neuronpedia_reference_cache*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            cache = _load_json(cache_path)
        except Exception:
            continue
        signature = cache.get("signature") or {}
        feature = cache.get("feature") or {}
        model_id = str(signature.get("model_id") or feature.get("model_id") or "").strip()
        source = str(signature.get("source") or feature.get("source") or "").strip()
        if model_id and source:
            return model_id, source

    return "gemma-2-2b", f"{layer_id}-gemmascope-res-16k"


def _resolve_explanation(run_dir: Path) -> str:
    eval_path = run_dir / "evaluation_record.json"
    if eval_path.exists():
        try:
            hypothesis = str((_load_json(eval_path).get("hypothesis") or "")).strip()
            if hypothesis:
                return hypothesis
        except Exception:
            pass

    llm_path = run_dir / "llm_inout.md"
    if llm_path.exists():
        return _extract_explanation_from_llm_inout(llm_path)

    raise ValueError(f"Cannot resolve explanation for run: {run_dir}")


def _run_compare(
    *,
    model_id: str,
    source: str,
    layer_id: str,
    feature_id: str,
    explanation: str,
    output_root: Path,
    timestamp: str,
    boundary_only: bool,
    max_retries: int,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as tf:
        tf.write(explanation)
        explanation_file = Path(tf.name)

    cmd: List[str] = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--model-id",
        model_id,
        "--layer-id",
        layer_id,
        "--source",
        source,
        "--feature-id",
        feature_id,
        "--my-explanation-file",
        str(explanation_file),
        "--sae-name",
        "gemmascope-res",
        "--output-root",
        str(output_root),
        "--timestamp",
        timestamp,
    ]
    if boundary_only:
        cmd.extend(["--disable-activation-score", "--disable-non-activation-score"])

    last_error: Optional[RuntimeError] = None
    try:
        for attempt in range(max_retries + 1):
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0:
                return

            tail_out = "\n".join((proc.stdout or "").splitlines()[-20:])
            tail_err = "\n".join((proc.stderr or "").splitlines()[-20:])
            last_error = RuntimeError(
                "compare_explanations_with_llm.py failed\n"
                f"attempt: {attempt + 1}/{max_retries + 1}\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout_tail:\n{tail_out}\n"
                f"stderr_tail:\n{tail_err}"
            )
        if last_error is not None:
            raise last_error
    finally:
        try:
            explanation_file.unlink(missing_ok=True)
        except Exception:
            pass


def _merge_boundary_outputs(*, original_run_dir: Path, temp_run_dir: Path) -> None:
    original_result_path = original_run_dir / "result.json"
    original_record_path = original_run_dir / "evaluation_record.json"
    temp_result_path = temp_run_dir / "result.json"
    temp_record_path = temp_run_dir / "evaluation_record.json"

    if not original_result_path.exists():
        raise FileNotFoundError(f"Missing original result.json: {original_result_path}")
    if not original_record_path.exists():
        raise FileNotFoundError(f"Missing original evaluation_record.json: {original_record_path}")
    if not temp_result_path.exists() or not temp_record_path.exists():
        raise FileNotFoundError(f"Missing temp outputs in: {temp_run_dir}")

    original_result = _load_json(original_result_path)
    temp_result = _load_json(temp_result_path)

    for key in BOUNDARY_RESULT_FIELDS:
        original_result[key] = temp_result.get(key)

    evaluations_enabled = original_result.get("evaluations_enabled")
    if not isinstance(evaluations_enabled, dict):
        evaluations_enabled = {}
        original_result["evaluations_enabled"] = evaluations_enabled
    temp_enabled = temp_result.get("evaluations_enabled")
    if isinstance(temp_enabled, dict):
        evaluations_enabled["boundary"] = bool(temp_enabled.get("boundary"))

    _write_json(original_result_path, original_result)

    original_record = _load_json(original_record_path)
    temp_record = _load_json(temp_record_path)

    original_eval_enabled = original_record.get("evaluations_enabled")
    if not isinstance(original_eval_enabled, dict):
        original_eval_enabled = {}
        original_record["evaluations_enabled"] = original_eval_enabled

    temp_eval_enabled = temp_record.get("evaluations_enabled")
    if isinstance(temp_eval_enabled, dict):
        original_eval_enabled["boundary"] = bool(temp_eval_enabled.get("boundary"))

    original_scores = original_record.get("scores")
    if not isinstance(original_scores, dict):
        original_scores = {}
        original_record["scores"] = original_scores

    temp_scores = temp_record.get("scores")
    if isinstance(temp_scores, dict):
        for key in BOUNDARY_SCORE_FIELDS:
            original_scores[key] = temp_scores.get(key)

    original_test_cases = original_record.get("test_cases")
    if not isinstance(original_test_cases, dict):
        original_test_cases = {}
        original_record["test_cases"] = original_test_cases

    temp_test_cases = temp_record.get("test_cases")
    if isinstance(temp_test_cases, dict):
        original_test_cases["boundary_cases_my"] = temp_test_cases.get("boundary_cases_my", [])
        original_test_cases["boundary_cases_reference"] = temp_test_cases.get("boundary_cases_reference", [])

    original_summary = original_record.get("summary")
    if not isinstance(original_summary, dict):
        original_summary = {}
        original_record["summary"] = original_summary

    temp_summary = temp_record.get("summary")
    if isinstance(temp_summary, dict):
        original_summary["num_boundary_cases_my"] = temp_summary.get("num_boundary_cases_my")
        original_summary["num_boundary_cases_reference"] = temp_summary.get("num_boundary_cases_reference")

    original_record["boundary_warning"] = temp_record.get("boundary_warning")
    _write_json(original_record_path, original_record)


def _collect_tasks(target_root: Path) -> List[RunTask]:
    tasks: List[RunTask] = []
    for layer_dir in sorted(target_root.glob("layer-*")):
        if not layer_dir.is_dir():
            continue
        for feature_dir in sorted(layer_dir.glob("feature-*")):
            if not feature_dir.is_dir():
                continue
            for run_dir in sorted(feature_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                files = sorted([p for p in run_dir.iterdir() if p.is_file()])
                file_names = [p.name for p in files]
                layer_id, feature_id = _extract_layer_feature(run_dir)
                model_id, source = _resolve_model_and_source(feature_dir, run_dir, layer_id)

                if file_names == ["llm_inout.md"]:
                    explanation = _resolve_explanation(run_dir)
                    tasks.append(
                        RunTask(
                            kind="full",
                            layer_id=layer_id,
                            feature_id=feature_id,
                            source=source,
                            model_id=model_id,
                            run_dir=run_dir,
                            explanation=explanation,
                        )
                    )
                    continue

                if len(file_names) == 4 and "result.json" in file_names:
                    result_payload = _load_json(run_dir / "result.json")
                    if _boundary_fields_missing(result_payload):
                        explanation = _resolve_explanation(run_dir)
                        tasks.append(
                            RunTask(
                                kind="boundary",
                                layer_id=layer_id,
                                feature_id=feature_id,
                                source=source,
                                model_id=model_id,
                                run_dir=run_dir,
                                explanation=explanation,
                            )
                        )

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair input-side evaluation outputs in bulk.")
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--temp-output-root", default=str(DEFAULT_TEMP_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_root = Path(args.target_root)
    temp_output_root = Path(args.temp_output_root)

    if not target_root.exists():
        raise FileNotFoundError(f"Target root does not exist: {target_root}")

    tasks = _collect_tasks(target_root)
    full_tasks = [t for t in tasks if t.kind == "full"]
    boundary_tasks = [t for t in tasks if t.kind == "boundary"]

    print(f"[scan] full_tasks={len(full_tasks)} boundary_tasks={len(boundary_tasks)}")
    for task in tasks:
        print(f"[scan] {task.kind:8s} layer={task.layer_id} feature={task.feature_id} run={task.run_dir.name}")

    if args.dry_run:
        return

    failures: List[Tuple[RunTask, str]] = []

    for task in full_tasks:
        print(f"[run] full      layer={task.layer_id} feature={task.feature_id} run={task.run_dir.name}")
        try:
            _run_compare(
                model_id=task.model_id,
                source=task.source,
                layer_id=task.layer_id,
                feature_id=task.feature_id,
                explanation=task.explanation,
                output_root=DEFAULT_OUTPUT_ROOT,
                timestamp=task.run_dir.name,
                boundary_only=False,
                max_retries=max(0, int(args.max_retries)),
            )
        except Exception as exc:
            failures.append((task, str(exc)))
            print(f"[fail] full      layer={task.layer_id} feature={task.feature_id} run={task.run_dir.name}")

    for task in boundary_tasks:
        print(f"[run] boundary  layer={task.layer_id} feature={task.feature_id} run={task.run_dir.name}")
        try:
            _run_compare(
                model_id=task.model_id,
                source=task.source,
                layer_id=task.layer_id,
                feature_id=task.feature_id,
                explanation=task.explanation,
                output_root=temp_output_root,
                timestamp=task.run_dir.name,
                boundary_only=True,
                max_retries=max(0, int(args.max_retries)),
            )

            temp_run_dir = (
                temp_output_root
                / "gemmascope-res"
                / f"layer-{task.layer_id}"
                / f"feature-{task.feature_id}"
                / task.run_dir.name
            )
            _merge_boundary_outputs(original_run_dir=task.run_dir, temp_run_dir=temp_run_dir)
        except Exception as exc:
            failures.append((task, str(exc)))
            print(f"[fail] boundary  layer={task.layer_id} feature={task.feature_id} run={task.run_dir.name}")

    if failures:
        print(f"[done] repair completed with failures: {len(failures)}")
        for task, message in failures:
            print(
                f"[failure] kind={task.kind} layer={task.layer_id} "
                f"feature={task.feature_id} run={task.run_dir.name}"
            )
            print(message)
        sys.exit(1)

    print("[done] repair completed")


if __name__ == "__main__":
    main()
