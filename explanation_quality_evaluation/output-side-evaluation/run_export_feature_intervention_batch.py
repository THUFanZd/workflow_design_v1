from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_SCRIPT = Path(__file__).resolve().parent / "export_feature_intervention_outputs.py"

TARGET_LAYERS = [0, 6, 12, 18, 24]
FEATURE_ID_MIN = 0
FEATURE_ID_MAX = 16383
FEATURES_PER_LAYER = 10
LAYER0_FIXED_FEATURES = [1, 2, 3, 4]


def _build_default_jobs(*, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    jobs: List[Dict[str, Any]] = []

    for layer in TARGET_LAYERS:
        if layer == 0:
            additional = rng.sample(
                list(range(5, FEATURE_ID_MAX + 1)),
                FEATURES_PER_LAYER - len(LAYER0_FIXED_FEATURES),
            )
            feature_ids = LAYER0_FIXED_FEATURES + additional
        else:
            feature_ids = rng.sample(
                list(range(FEATURE_ID_MIN, FEATURE_ID_MAX + 1)),
                FEATURES_PER_LAYER,
            )

        for feature_id in feature_ids:
            jobs.append({"layer_id": layer, "feature_id": int(feature_id), "width": "16k"})
    return jobs


def _load_jobs_from_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--jobs-json must be a JSON array.")
    jobs: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each job in --jobs-json must be a JSON object.")
        if "layer_id" not in item or "feature_id" not in item:
            raise ValueError("Each job requires layer_id and feature_id.")
        job = {
            "layer_id": int(item["layer_id"]),
            "feature_id": int(item["feature_id"]),
            "width": str(item.get("width", "16k")),
        }
        jobs.append(job)
    return jobs


def _build_command(
    *,
    python_executable: str,
    export_script: Path,
    job: Dict[str, Any],
    sae_name: str,
    model_checkpoint_path: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    output_root: str,
) -> List[str]:
    return [
        python_executable,
        str(export_script),
        "--layer-id",
        str(job["layer_id"]),
        "--feature-id",
        str(job["feature_id"]),
        "--width",
        str(job["width"]),
        "--sae-name",
        sae_name,
        "--model-checkpoint-path",
        model_checkpoint_path,
        "--device",
        device,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--output-root",
        output_root,
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch run export_feature_intervention_outputs.py. "
            "By default, this script generates 50 jobs: layers [0, 6, 12, 18, 24], "
            "10 features each, following the requested sampling rules."
        )
    )
    parser.add_argument(
        "--jobs-json",
        default=None,
        help="Optional JSON file path containing job list. If omitted, use built-in 50 jobs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for built-in job generation.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--export-script", default=str(DEFAULT_EXPORT_SCRIPT))

    parser.add_argument("--sae-name", default="gemmascope-res")
    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation"),
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one job fails.")
    parser.add_argument(
        "--jobs-txt",
        default=str(Path(__file__).resolve().parent / "batch_jobs.txt"),
        help="Path to save resolved jobs as runnable command lines.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to save batch run summary JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    export_script = Path(args.export_script).resolve()
    if not export_script.exists():
        raise FileNotFoundError(f"Cannot find export script: {export_script}")

    if args.jobs_json:
        jobs = _load_jobs_from_json(Path(args.jobs_json))
    else:
        jobs = _build_default_jobs(seed=int(args.seed))

    prepared_commands: List[List[str]] = []
    for job in jobs:
        prepared_commands.append(
            _build_command(
                python_executable=str(args.python_executable),
                export_script=export_script,
                job=job,
                sae_name=str(args.sae_name),
                model_checkpoint_path=str(args.model_checkpoint_path),
                device=str(args.device),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                output_root=str(args.output_root),
            )
        )

    jobs_txt_path = Path(args.jobs_txt)
    jobs_txt_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_txt_lines: List[str] = []
    jobs_txt_lines.append(f"# generated_at={datetime.now().isoformat()}")
    jobs_txt_lines.append(f"# total_jobs={len(prepared_commands)}")
    jobs_txt_lines.append("")
    for idx, cmd in enumerate(prepared_commands, start=1):
        jobs_txt_lines.append(f"# job_{idx}")
        jobs_txt_lines.append(subprocess.list2cmdline(cmd))
    jobs_txt_path.write_text("\n".join(jobs_txt_lines) + "\n", encoding="utf-8")
    print(f"Jobs saved to: {jobs_txt_path}")

    start = datetime.now()
    successes = 0
    failures = 0
    run_logs: List[Dict[str, Any]] = []

    from tqdm import tqdm
    for idx, (job, cmd) in enumerate(
        tqdm(zip(jobs, prepared_commands), total=len(jobs), desc="Running Jobs"),
        start=1,
    ):
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        success = result.returncode == 0
        if success:
            successes += 1
        else:
            failures += 1

        run_logs.append(
            {
                "index": idx,
                "layer_id": int(job["layer_id"]),
                "feature_id": int(job["feature_id"]),
                "width": str(job["width"]),
                "success": success,
                "returncode": int(result.returncode),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": cmd,
            }
        )
        status = "OK" if success else "FAIL"
        print(
            f"[{idx}/{len(jobs)}] {status} layer={job['layer_id']} feature={job['feature_id']} width={job['width']}"
        )
        if (not success) and args.fail_fast:
            break

    end = datetime.now()
    summary = {
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "duration_seconds": (end - start).total_seconds(),
        "total_jobs": len(run_logs),
        "successes": successes,
        "failures": failures,
        "used_default_jobs": args.jobs_json is None,
        "seed": int(args.seed),
        "jobs": run_logs,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Summary saved to: {summary_path}")
    else:
        print(json.dumps({k: summary[k] for k in ["total_jobs", "successes", "failures"]}, ensure_ascii=False))

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
