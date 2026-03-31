#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_SINGLE_SCRIPT = PROJECT_ROOT / "run_single_input_history_scope_eval.py"
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "logs" / "single_run_batch"

DEFAULT_PAIRS: List[Tuple[int, int]] = [
    (0, 414), (0, 819), (0, 2848), (0, 3358), (0, 3648), (0, 4572), (0, 7314), (0, 8024), (0, 9012), (0, 12154),
    (6, 869), (6, 976), (6, 1041), (6, 3070), (6, 6515), (6, 7164), (6, 7223), (6, 7623), (6, 13746), (6, 13825),
    (12, 212), (12, 5094), (12, 5231), (12, 7055), (12, 9105), (12, 9115), (12, 11029), (12, 11149), (12, 13848), (12, 14719),
    (18, 1423), (18, 3039), (18, 3169), (18, 3349), (18, 4090), (18, 8667), (18, 11270), (18, 11763), (18, 12449), (18, 15054),
    (24, 1501), (24, 2279), (24, 2582), (24, 6300), (24, 7467), (24, 8667), (24, 9606), (24, 11850), (24, 12403), (24, 12493),
]


def _iter_with_progress(items: List[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
    if tqdm is None:
        total = len(items)
        for i, item in enumerate(items, start=1):
            print(f"[{i}/{total}] layer={item[0]} feature={item[1]}")
            yield item
        return
    bar = tqdm(items, desc="single_run batch", unit="feature")
    for layer_id, feature_id in bar:
        bar.set_postfix(layer=layer_id, feature=feature_id)
        yield layer_id, feature_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run run_single_input_history_scope_eval.py for fixed feature pairs."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-single-script", default=str(DEFAULT_RUN_SINGLE_SCRIPT))
    parser.add_argument("--model-checkpoint-path", default="/data/MODEL/Gemma-2-2b")
    parser.add_argument("--sae-root", default="/data/MODEL/gemma-scope-2b-pt-res")
    parser.add_argument("--llm-api-key-file", default="support_info/ali_api_key.txt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history-scope", default="same_hypothesis", choices=["same_hypothesis", "all_hypotheses"])
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--num-hypothesis", type=int, default=3)
    parser.add_argument("--side", default="input", choices=["input", "output", "both"])
    parser.add_argument("--observation-source", default="bos_token", choices=["neuronpedia", "bos_token"])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--commands-output",
        default=None,
        help=(
            "Optional path to save all planned commands as a shell script (.sh). "
            "If omitted and --dry-run is set, defaults to <batch_dir>/dry_run_commands.sh."
        ),
    )
    parser.add_argument(
        "--batch-root",
        default=str(DEFAULT_BATCH_ROOT),
        help="Directory to store per-pair summaries and final batch manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_single_script = Path(str(args.run_single_script))
    if not run_single_script.exists():
        raise FileNotFoundError(f"run_single script not found: {run_single_script}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(str(args.batch_root)) / run_tag
    batch_dir.mkdir(parents=True, exist_ok=True)
    commands_output_path = (
        Path(str(args.commands_output))
        if args.commands_output
        else (batch_dir / "dry_run_commands.sh" if args.dry_run else None)
    )
    planned_command_lines: List[str] = []

    records = []
    for layer_id, feature_id in _iter_with_progress(DEFAULT_PAIRS):
        single_summary_path = batch_dir / f"layer-{layer_id}_feature-{feature_id}.json"
        cmd = [
            str(args.python_exe),
            str(run_single_script),
            "--layer-id",
            str(layer_id),
            "--feature-id",
            str(feature_id),
            "--model-checkpoint-path",
            str(args.model_checkpoint_path),
            "--sae-root",
            str(args.sae_root),
            "--llm-api-key-file",
            str(args.llm_api_key_file),
            "--device",
            str(args.device),
            "--history-scope",
            str(args.history_scope),
            "--max-rounds",
            str(args.max_rounds),
            "--num-hypothesis",
            str(args.num_hypothesis),
            "--side",
            str(args.side),
            "--observation-source",
            str(args.observation_source),
            "--summary-output",
            str(single_summary_path),
        ]
        planned_command_lines.append(shlex.join(cmd))

        started = time.perf_counter()
        print("$", " ".join(cmd))
        if args.dry_run:
            returncode = 0
        else:
            completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
            returncode = int(completed.returncode)
        elapsed = time.perf_counter() - started

        record = {
            "layer_id": layer_id,
            "feature_id": feature_id,
            "returncode": returncode,
            "seconds": elapsed,
            "summary_output": str(single_summary_path),
            "command": cmd,
        }
        records.append(record)

        if returncode != 0 and not args.continue_on_error:
            print(
                f"Stopped at layer={layer_id}, feature={feature_id}, returncode={returncode}. "
                "Use --continue-on-error to keep running.",
                file=sys.stderr,
            )
            break

    failed_count = sum(1 for r in records if int(r["returncode"]) != 0)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "total_pairs": len(DEFAULT_PAIRS),
        "executed_pairs": len(records),
        "failed_pairs": failed_count,
        "dry_run": bool(args.dry_run),
        "commands_output": str(commands_output_path) if commands_output_path else None,
        "args": {
            "model_checkpoint_path": str(args.model_checkpoint_path),
            "sae_root": str(args.sae_root),
            "llm_api_key_file": str(args.llm_api_key_file),
            "device": str(args.device),
            "history_scope": str(args.history_scope),
            "max_rounds": int(args.max_rounds),
            "num_hypothesis": int(args.num_hypothesis),
            "side": str(args.side),
            "observation_source": str(args.observation_source),
            "continue_on_error": bool(args.continue_on_error),
        },
        "records": records,
    }
    if commands_output_path is not None:
        commands_output_path.parent.mkdir(parents=True, exist_ok=True)
        commands_output_payload = "\n".join(planned_command_lines) + "\n"
        commands_output_path.write_text(commands_output_payload, encoding="utf-8")
        print(f"Saved commands: {commands_output_path}")

    manifest_path = batch_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
