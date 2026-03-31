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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_SINGLE_SCRIPT = PROJECT_ROOT / "run_single_input_history_scope_eval.py"
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "logs" / "single_run_batch"
Pair = Tuple[int, int]
Record = Dict[str, Any]

DEFAULT_PAIRS: List[Pair] = [
    (0, 414), (0, 819), (0, 2848), (0, 3358), (0, 3648), (0, 4572), (0, 7314), (0, 8024), (0, 9012), (0, 12154),
    (6, 869), (6, 976), (6, 1041), (6, 3070), (6, 6515), (6, 7164), (6, 7223), (6, 7623), (6, 13746), (6, 13825),
    (12, 212), (12, 5094), (12, 5231), (12, 7055), (12, 9105), (12, 9115), (12, 11029), (12, 11149), (12, 13848), (12, 14719),
    (18, 1423), (18, 3039), (18, 3169), (18, 3349), (18, 4090), (18, 8667), (18, 11270), (18, 11763), (18, 12449), (18, 15054),
    (24, 1501), (24, 2279), (24, 2582), (24, 6300), (24, 7467), (24, 8667), (24, 9606), (24, 11850), (24, 12403), (24, 12493),
]


def _iter_with_progress(items: List[Pair]) -> Iterable[Pair]:
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _record_pair(record: Record) -> Optional[Pair]:
    try:
        layer_id = int(record.get("layer_id"))
        feature_id = int(record.get("feature_id"))
        return layer_id, feature_id
    except (TypeError, ValueError):
        return None


def _latest_records_by_pair(records: Sequence[Record]) -> Dict[Pair, Record]:
    latest: Dict[Pair, Record] = {}
    for record in records:
        pair = _record_pair(record)
        if pair is None:
            continue
        latest[pair] = record
    return latest


def _is_success_record(record: Record) -> bool:
    if _safe_int(record.get("returncode"), default=1) != 0:
        return False
    return not bool(record.get("dry_run", False))


def _load_existing_records(manifest_path: Path) -> List[Record]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest payload must be a JSON object: {manifest_path}")
    records_raw = payload.get("records", [])
    if not isinstance(records_raw, list):
        raise ValueError(f"Manifest 'records' must be a list: {manifest_path}")
    manifest_dry_run = bool(payload.get("dry_run", False))
    records: List[Record] = []
    for item in records_raw:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        if "dry_run" not in record:
            record["dry_run"] = manifest_dry_run
        records.append(record)
    return records


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_manifest(
    *,
    args: argparse.Namespace,
    run_tag: str,
    records: Sequence[Record],
    commands_output_path: Optional[Path],
    failed_records_path: Path,
    failed_commands_path: Path,
    resume: bool,
) -> Dict[str, Any]:
    latest = _latest_records_by_pair(records)
    completed_pairs_count = sum(1 for record in latest.values() if _is_success_record(record))
    failed_pairs_count = sum(
        1
        for record in latest.values()
        if _safe_int(record.get("returncode"), default=1) != 0
    )
    pending_pairs_count = max(0, len(DEFAULT_PAIRS) - completed_pairs_count - failed_pairs_count)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "total_pairs": len(DEFAULT_PAIRS),
        "executed_pairs": len(latest),
        "completed_pairs": completed_pairs_count,
        "failed_pairs": failed_pairs_count,
        "pending_pairs": pending_pairs_count,
        "record_count": len(records),
        "dry_run": bool(args.dry_run),
        "resumed": bool(resume),
        "commands_output": str(commands_output_path) if commands_output_path else None,
        "failed_records_output": str(failed_records_path),
        "failed_commands_output": str(failed_commands_path),
        "args": {
            "model_checkpoint_path": str(args.model_checkpoint_path),
            "sae_root": str(args.sae_root),
            "llm_api_key_file": str(args.llm_api_key_file),
            "device": str(args.device),
            "history_scope": str(args.history_scope),
            "max_rounds": int(args.max_rounds),
            "num_hypothesis": int(args.num_hypothesis),
            "num_input_sentences_per_hypothesis": int(args.num_input_sentences_per_hypothesis),
            "side": str(args.side),
            "observation_source": str(args.observation_source),
            "continue_on_error": bool(args.continue_on_error),
            "run_tag": str(run_tag),
            "resume": bool(resume),
        },
        "records": list(records),
    }


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
    parser.add_argument("--num-input-sentences-per-hypothesis", type=int, default=5)
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
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Use this run tag directory under --batch-root. If omitted, use current time.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing <batch-root>/<run-tag>/batch_manifest.json. "
            "Completed pairs (returncode=0 and not dry_run) will be skipped."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_single_script = Path(str(args.run_single_script))
    if not run_single_script.exists():
        raise FileNotFoundError(f"run_single script not found: {run_single_script}")
    if args.resume and not args.run_tag:
        raise ValueError("--resume requires --run-tag.")

    run_tag = str(args.run_tag).strip() if args.run_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_tag:
        raise ValueError("--run-tag cannot be empty.")
    batch_dir = Path(str(args.batch_root)) / run_tag
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "batch_manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. "
            "Use --resume to continue this batch, or use a different --run-tag."
        )

    existing_records = _load_existing_records(manifest_path) if args.resume else []
    latest_existing = _latest_records_by_pair(existing_records)
    completed_pairs = {
        pair for pair, record in latest_existing.items() if _is_success_record(record)
    }
    if args.resume:
        print(
            f"Resume mode: loaded {len(existing_records)} records, "
            f"{len(completed_pairs)} completed pairs will be skipped."
        )

    commands_output_path = (
        Path(str(args.commands_output))
        if args.commands_output
        else (batch_dir / "dry_run_commands.sh" if args.dry_run else None)
    )
    failed_records_path = batch_dir / "failed_records.json"
    failed_commands_path = batch_dir / "failed_commands.sh"
    planned_command_lines: List[str] = []

    records: List[Record] = list(existing_records)
    for layer_id, feature_id in _iter_with_progress(DEFAULT_PAIRS):
        pair = (layer_id, feature_id)
        if pair in completed_pairs:
            print(f"Skip completed pair: layer={layer_id}, feature={feature_id}")
            continue

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
            "--num-input-sentences-per-hypothesis",
            str(args.num_input_sentences_per_hypothesis),
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
            "dry_run": bool(args.dry_run),
        }
        records.append(record)
        if _is_success_record(record):
            completed_pairs.add(pair)

        manifest = _build_manifest(
            args=args,
            run_tag=run_tag,
            records=records,
            commands_output_path=commands_output_path,
            failed_records_path=failed_records_path,
            failed_commands_path=failed_commands_path,
            resume=bool(args.resume),
        )
        _write_json(manifest_path, manifest)

        if returncode != 0 and not args.continue_on_error:
            print(
                f"Stopped at layer={layer_id}, feature={feature_id}, returncode={returncode}. "
                "Use --continue-on-error to keep running.",
                file=sys.stderr,
            )
            break

    latest_records = _latest_records_by_pair(records)
    latest_failed_records: List[Record] = []
    for pair in DEFAULT_PAIRS:
        record = latest_records.get(pair)
        if record is None:
            continue
        if _safe_int(record.get("returncode"), default=0) != 0:
            latest_failed_records.append(record)

    failed_payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "failed_count": len(latest_failed_records),
        "records": latest_failed_records,
    }
    _write_json(failed_records_path, failed_payload)

    failed_command_lines: List[str] = []
    for record in latest_failed_records:
        cmd_obj = record.get("command")
        if isinstance(cmd_obj, list) and cmd_obj:
            failed_command_lines.append(shlex.join([str(part) for part in cmd_obj]))
    failed_commands_payload = "\n".join(failed_command_lines)
    if failed_commands_payload:
        failed_commands_payload += "\n"
    failed_commands_path.write_text(failed_commands_payload, encoding="utf-8")

    if commands_output_path is not None:
        commands_output_path.parent.mkdir(parents=True, exist_ok=True)
        commands_output_payload = "\n".join(planned_command_lines)
        if commands_output_payload:
            commands_output_payload += "\n"
        commands_output_path.write_text(commands_output_payload, encoding="utf-8")
        print(f"Saved commands: {commands_output_path}")

    manifest = _build_manifest(
        args=args,
        run_tag=run_tag,
        records=records,
        commands_output_path=commands_output_path,
        failed_records_path=failed_records_path,
        failed_commands_path=failed_commands_path,
        resume=bool(args.resume),
    )
    _write_json(manifest_path, manifest)
    print(f"Saved failed records: {failed_records_path}")
    print(f"Saved failed commands: {failed_commands_path}")
    print(f"Saved manifest: {manifest_path}")

    failed_count = len(latest_failed_records)
    if failed_count > 0:
        print(f"Failed pairs ({failed_count}):")
        for record in latest_failed_records:
            print(
                f"- layer={record.get('layer_id')}, "
                f"feature={record.get('feature_id')}, "
                f"returncode={record.get('returncode')}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
