from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"
)
HISTORY_SCOPES = ("same_hypothesis", "all_hypotheses")
TARGET_LAYERS = (6, 12, 18, 24)
FIXED_FEATURE = (0, 12154)


@dataclass
class RunRecord:
    layer_id: int
    feature_id: int
    history_scope: str
    timestamp: str
    workflow_returncode: int
    evaluation_returncode: int | None
    workflow_seconds: float
    evaluation_seconds: float


def _discover_features_for_layer(
    *,
    output_root: Path,
    sae_name: str,
    layer_id: int,
) -> List[int]:
    layer_dir = output_root / sae_name / f"layer-{layer_id}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory does not exist: {layer_dir}")
    feature_ids: List[int] = []
    for child in sorted(layer_dir.iterdir()):
        if not child.is_dir():
            continue
        match = re.fullmatch(r"feature-(\d+)", child.name)
        if match is None:
            continue
        feature_ids.append(int(match.group(1)))
    return feature_ids


def _build_selection(
    *,
    output_root: Path,
    sae_name: str,
    seed: int,
    sample_per_layer: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    selected: List[Tuple[int, int]] = [FIXED_FEATURE]
    for layer_id in TARGET_LAYERS:
        feature_ids = _discover_features_for_layer(
            output_root=output_root,
            sae_name=sae_name,
            layer_id=layer_id,
        )
        if len(feature_ids) < sample_per_layer:
            raise ValueError(
                f"layer-{layer_id} has {len(feature_ids)} features, cannot sample {sample_per_layer}."
            )
        sampled = sorted(rng.sample(feature_ids, sample_per_layer))
        selected.extend((layer_id, fid) for fid in sampled)
    return selected


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> Tuple[int, float]:
    started = time.perf_counter()
    print("$", " ".join(cmd))
    if dry_run:
        return 0, 0.0
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return completed.returncode, time.perf_counter() - started


def _save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch run input-side hypothesis iteration for two history scopes, "
            "then run final explanation evaluation."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--sae-name", default="gemmascope-res")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-layer", type=int, default=2)
    parser.add_argument("--max-rounds", "--max_round", dest="max_rounds", type=int, default=1)
    parser.add_argument(
        "--selection-method",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Passed to workflow_runner.py --selection-method.",
    )
    parser.add_argument(
        "--input-selection-method",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Passed to final_explanation_evaluation_runner.py --input-selection-method.",
    )
    parser.add_argument(
        "--final-run-mode",
        choices=["both", "input", "output", "none"],
        default="both",
        help="Passed to final_explanation_evaluation_runner.py --run-mode.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next task if one workflow/evaluation run fails.",
    )
    parser.add_argument(
        "--force-run-input-eval",
        action="store_true",
        help="Passed to final_explanation_evaluation_runner.py to force compare_explanations_with_llm.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plan-output",
        default=str(PROJECT_ROOT / "logs" / "batch_input_history_scope_plan.json"),
        help="Where to save selected features and run records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(str(args.output_root))
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    selection = _build_selection(
        output_root=output_root,
        sae_name=str(args.sae_name),
        seed=int(args.seed),
        sample_per_layer=int(args.sample_per_layer),
    )

    print("Selected features:")
    for layer_id, feature_id in selection:
        print(f"- layer={layer_id}, feature={feature_id}")
    
    records: List[RunRecord] = []
    for layer_id, feature_id in selection:
        for history_scope in HISTORY_SCOPES:
            timestamp = f"{now_tag}_l{layer_id}_f{feature_id}_{history_scope}"
            workflow_cmd = [
                str(args.python_exe),
                str(PROJECT_ROOT / "workflow_runner.py"),
                "--layer-id",
                str(layer_id),
                "--feature-id",
                str(feature_id),
                "--side",
                "input",
                "--history-scope",
                history_scope,
                "--max-rounds",
                str(args.max_rounds),
                "--selection-method",
                str(args.selection_method),
                "--timestamp",
                timestamp,
            ]
            workflow_code, workflow_seconds = _run_command(workflow_cmd, dry_run=bool(args.dry_run))

            evaluation_code: int | None = None
            evaluation_seconds = 0.0
            if workflow_code == 0:
                evaluation_cmd = [
                    str(args.python_exe),
                    str(PROJECT_ROOT / "final_explanation_evaluation_runner.py"),
                    "--layer-id",
                    str(layer_id),
                    "--feature-id",
                    str(feature_id),
                    "--timestamp",
                    timestamp,
                    "--run-mode",
                    str(args.final_run_mode),
                    "--input-selection-method",
                    str(args.input_selection_method),
                ]
                if bool(args.force_run_input_eval):
                    evaluation_cmd.append("--force-run-input-eval")
                evaluation_code, evaluation_seconds = _run_command(
                    evaluation_cmd,
                    dry_run=bool(args.dry_run),
                )
            record = RunRecord(
                layer_id=layer_id,
                feature_id=feature_id,
                history_scope=history_scope,
                timestamp=timestamp,
                workflow_returncode=workflow_code,
                evaluation_returncode=evaluation_code,
                workflow_seconds=workflow_seconds,
                evaluation_seconds=evaluation_seconds,
            )
            records.append(record)

            failed = workflow_code != 0 or (evaluation_code is not None and evaluation_code != 0)
            if failed and not args.continue_on_error:
                raise SystemExit(
                    "Batch stopped due to failure. Use --continue-on-error to keep going."
                )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "sample_per_layer": int(args.sample_per_layer),
        "max_rounds": int(args.max_rounds),
        "selection_method": int(args.selection_method),
        "input_selection_method": int(args.input_selection_method),
        "final_run_mode": str(args.final_run_mode),
        "force_run_input_eval": bool(args.force_run_input_eval),
        "selection": [{"layer_id": l, "feature_id": f} for l, f in selection],
        "records": [r.__dict__ for r in records],
    }
    _save_manifest(Path(str(args.plan_output)), summary)
    print(f"Saved manifest: {args.plan_output}")


if __name__ == "__main__":
    main()
