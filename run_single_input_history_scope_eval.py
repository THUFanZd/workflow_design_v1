from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class RunSummary:
    layer_id: int
    feature_id: int
    side: str
    final_run_mode: str
    history_scope: str
    merge_enabled: bool
    timestamp: str
    workflow_returncode: int
    evaluation_returncode: int | None
    workflow_seconds: float
    evaluation_seconds: float
    workflow_command: list[str]
    evaluation_command: list[str] | None


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> Tuple[int, float]:
    started = time.perf_counter()
    print("$", " ".join(cmd))
    if dry_run:
        return 0, 0.0
    completed = subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), check=False)
    return completed.returncode, time.perf_counter() - started


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_timestamp(args: argparse.Namespace) -> str:
    if args.timestamp:
        return str(args.timestamp).strip()
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return now_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one workflow_runner.py task, then run final_explanation_evaluation_runner.py "
            "with the same layer/feature/timestamp."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--workflow-script",
        default=str(PROJECT_ROOT / "workflow_runner.py"),
        help="Workflow entry script path. Default: workflow_runner.py",
    )
    parser.add_argument(
        "--final-eval-script",
        default=str(PROJECT_ROOT / "final_explanation_evaluation_runner.py"),
        help="Final evaluation entry script path. Default: final_explanation_evaluation_runner.py",
    )
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--timestamp", default=None, help="If omitted, use current time YYYYMMDD_HHMMSS.")

    parser.add_argument("--model-id", default="gemma-2-2b")
    parser.add_argument("--max-rounds", "--max_round", dest="max_rounds", type=int, default=1)
    parser.add_argument("--num-hypothesis", type=int, default=3)
    parser.add_argument(
        "--generation-mode",
        choices=["single_call", "iterative"],
        default="single_call",
        help="Passed to workflow_runner.py --generation-mode.",
    )
    parser.add_argument("--side", choices=["both", "input", "output"], default="input")
    parser.add_argument(
        "--history-scope",
        choices=["same_hypothesis", "all_hypotheses"],
        default="same_hypothesis",
    )
    parser.add_argument("--enable-hypothesis-merge", action="store_true")
    parser.add_argument("--num-input-sentences-per-hypothesis", type=int, default=5)
    parser.add_argument(
        "--top-m",
        type=int,
        default=None,
        help="Passed to workflow_runner.py --top-m. If omitted, workflow default applies.",
    )
    parser.add_argument(
        "--history-rounds",
        type=int,
        default=1,
        help="Passed to workflow_runner.py --history-rounds.",
    )
    parser.add_argument("--width", default="16k", help="Passed to workflow/final-eval width args.")
    parser.add_argument(
        "--selection-method",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Passed to workflow_runner.py --selection-method.",
    )
    parser.add_argument(
        "--observation-m",
        type=int,
        default=2,
        help="Passed only to workflow_runner.py --observation-m (workflow observation sampling).",
    )
    parser.add_argument(
        "--observation-n",
        type=int,
        default=2,
        help="Passed only to workflow_runner.py --observation-n (workflow observation sampling).",
    )

    parser.add_argument("--sae-name", default="gemmascope-res")
    parser.add_argument(
        "--input-max-explanations",
        type=int,
        default=3,
        help="Passed to final_explanation_evaluation_runner.py --input-max-explanations.",
    )
    parser.add_argument(
        "--input-non-activation-context-count",
        type=int,
        default=5,
        help="Passed to final_explanation_evaluation_runner.py --input-non-activation-context-count.",
    )
    parser.add_argument(
        "--force-run-input-eval",
        action="store_true",
        help="Passed through for compatibility. (Currently a no-op in final_explanation_evaluation_runner.py)",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Optional summary json path. Default: "
            "logs/{layer_id}/{feature_id}/{timestamp}/run_single_input_history_scope_eval_summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = _resolve_timestamp(args)
    layer_id = int(args.layer_id)
    feature_id = int(args.feature_id)
    final_run_mode = str(args.side)

    workflow_cmd = [
        str(args.python_exe),
        str(args.workflow_script),
        "--layer-id",
        str(layer_id),
        "--feature-id",
        str(feature_id),
        "--timestamp",
        timestamp,
        "--side",
        str(args.side),
        "--history-scope",
        str(args.history_scope),
        "--max-rounds",
        str(args.max_rounds),
        "--num-hypothesis",
        str(args.num_hypothesis),
        "--generation-mode",
        str(args.generation_mode),
        "--num-input-sentences-per-hypothesis",
        str(args.num_input_sentences_per_hypothesis),
        "--history-rounds",
        str(args.history_rounds),
        "--width",
        str(args.width),
        "--selection-method",
        str(args.selection_method),
        "--observation-m",
        str(args.observation_m),
        "--observation-n",
        str(args.observation_n),
        "--model-id",
        str(args.model_id),
    ]
    if bool(args.enable_hypothesis_merge):
        workflow_cmd.append("--enable-hypothesis-merge")
    if args.top_m is not None:
        workflow_cmd.extend(["--top-m", str(args.top_m)])

    workflow_code, workflow_seconds = _run_command(workflow_cmd, dry_run=bool(args.dry_run))

    evaluation_cmd: list[str] | None = None
    evaluation_code: int | None = None
    evaluation_seconds = 0.0
    if workflow_code == 0:
        evaluation_cmd = [
            str(args.python_exe),
            str(args.final_eval_script),
            "--layer-id",
            str(layer_id),
            "--feature-id",
            str(feature_id),
            "--timestamp",
            timestamp,
            "--sae-name",
            str(args.sae_name),
            "--width",
            str(args.width),
            "--run-mode",
            final_run_mode,
            "--input-max-explanations",
            str(args.input_max_explanations),
            "--input-non-activation-context-count",
            str(args.input_non_activation_context_count),
        ]
        if bool(args.force_run_input_eval):
            evaluation_cmd.append("--force-run-input-eval")
        evaluation_code, evaluation_seconds = _run_command(
            evaluation_cmd,
            dry_run=bool(args.dry_run),
        )

    summary = RunSummary(
        layer_id=layer_id,
        feature_id=feature_id,
        side=str(args.side),
        final_run_mode=final_run_mode,
        history_scope=str(args.history_scope),
        merge_enabled=bool(args.enable_hypothesis_merge),
        timestamp=timestamp,
        workflow_returncode=workflow_code,
        evaluation_returncode=evaluation_code,
        workflow_seconds=workflow_seconds,
        evaluation_seconds=evaluation_seconds,
        workflow_command=workflow_cmd,
        evaluation_command=evaluation_cmd,
    )
    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "summary": asdict(summary),
    }
    if args.summary_output:
        summary_path = Path(str(args.summary_output))
    else:
        summary_path = (
            PROJECT_ROOT
            / "logs"
            / str(layer_id)
            / str(feature_id)
            / timestamp
            / "run_single_input_history_scope_eval_summary.json"
        )
    _save_json(summary_path, summary_payload)
    print(f"Saved summary: {summary_path}")

    if workflow_code != 0:
        raise SystemExit(workflow_code)
    if evaluation_code is not None and evaluation_code != 0:
        raise SystemExit(evaluation_code)


if __name__ == "__main__":
    main()
