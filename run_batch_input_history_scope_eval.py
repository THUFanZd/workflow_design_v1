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

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"
)
HISTORY_SCOPES = ("same_hypothesis", "all_hypotheses")
TARGET_LAYERS = (6, 12, 18, 24)


@dataclass
class RunRecord:
    layer_id: int
    feature_id: int
    history_scope: str
    merge_enabled: bool
    sae_path: str | None
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


def _parse_pair_text(text: str) -> Tuple[int, int]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty target pair.")

    normalized = raw.replace("(", "").replace(")", "").replace(" ", "")
    if ":" in normalized:
        parts = normalized.split(":")
    elif "," in normalized:
        parts = normalized.split(",")
    else:
        raise ValueError(f"Invalid pair format: {text!r}. Use layer,feature or layer:feature.")

    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {text!r}.")
    return int(parts[0]), int(parts[1])


def _collect_manual_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    if args.target_pairs:
        for item in args.target_pairs:
            pairs.append(_parse_pair_text(item))

    if args.target_pairs_file:
        path = Path(str(args.target_pairs_file))
        if not path.exists():
            raise FileNotFoundError(f"target pairs file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pairs.append(_parse_pair_text(line))

    deduped: List[Tuple[int, int]] = []
    seen = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _build_selection(
    *,
    output_root: Path,
    sae_name: str,
    seed: int,
    sample_per_layer: int,
    manual_pairs: Sequence[Tuple[int, int]],
    target_total_features: int | None,
    width: str,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    deduped: List[Tuple[int, int]] = []
    seen = set()
    for pair in manual_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)

    def _parse_width_to_feature_count(text: str) -> int:
        raw = str(text).strip().lower()
        if not raw:
            return 16384
        match = re.fullmatch(r"(\d+)([km]?)", raw)
        if not match:
            return 16384
        base = int(match.group(1))
        suffix = match.group(2)
        if suffix == "k":
            return base * 1024
        if suffix == "m":
            return base * 1024 * 1024
        return base

    if target_total_features is not None:
        if target_total_features <= 0:
            raise ValueError("--target-total-features must be > 0.")
        if len(deduped) > target_total_features:
            raise ValueError(
                f"manual pairs count {len(deduped)} exceeds target total {target_total_features}."
            )

        feature_space_size = _parse_width_to_feature_count(width)
        if feature_space_size <= 0:
            raise ValueError(f"Invalid width/feature space: {width!r}")

        manual_layers = sorted({int(layer_id) for layer_id, _ in deduped})
        layers = manual_layers or list(TARGET_LAYERS)
        needed = target_total_features - len(deduped)
        while needed > 0:
            layer_id = layers[rng.randrange(len(layers))]
            feature_id = rng.randrange(feature_space_size)
            candidate = (layer_id, feature_id)
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
            needed -= 1
        return deduped

    if sample_per_layer <= 0:
        return deduped

    selected: List[Tuple[int, int]] = []
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

    for pair in selected:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> Tuple[int, float]:
    started = time.perf_counter()
    print("$", " ".join(cmd))
    if dry_run:
        return 0, 0.0
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return completed.returncode, time.perf_counter() - started


def _load_single_summary(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Invalid summary format in {summary_path}")
    return summary


def _save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_plan_output_path(base_path: Path, *, end_time: datetime) -> Path:
    end_tag = end_time.strftime("%Y%m%d_%H%M%S")
    return base_path.parent / end_tag / base_path.name


def _extract_average_l0_from_canonical_map(
    *,
    canonical_map_path: Path,
    layer_id: str,
    width: str,
) -> str | None:
    if not canonical_map_path.exists():
        return None

    target_id = f"layer_{layer_id}/width_{width}/canonical"
    in_target_block = False
    path_pattern = re.compile(
        rf"layer_{re.escape(layer_id)}/width_{re.escape(width)}/average_l0_([0-9]+(?:\.[0-9]+)?)"
    )
    with canonical_map_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("- id:"):
                current_id = line.split(":", 1)[1].strip()
                in_target_block = current_id == target_id
                continue
            if in_target_block and line.startswith("path:"):
                match = path_pattern.search(line.split(":", 1)[1].strip())
                if match:
                    return match.group(1)
                return None
    return None


def _extract_average_l0_from_path_suffix(path_text: str) -> str | None:
    match = re.search(r"average_l0_([0-9]+(?:\.[0-9]+)?)", path_text)
    if not match:
        return None
    return match.group(1)


def _extract_width_from_sae_path(path_text: str) -> str | None:
    match = re.search(r"width_([^/\\]+)", path_text)
    if not match:
        return None
    return match.group(1)


def _resolve_pair_sae_path(*, args: argparse.Namespace, layer_id: int) -> str | None:
    base_sae_path = args.sae_path
    if not base_sae_path:
        return None

    raw = str(base_sae_path).strip()
    if not raw:
        return None

    # Allow explicit templating for advanced usage.
    if "{layer_id}" in raw:
        return raw.format(layer_id=layer_id)

    normalized = raw.replace("\\", "/")
    layer_match = re.search(r"/layer_\d+/", normalized)
    width = _extract_width_from_sae_path(normalized)
    if layer_match is None or not width:
        return raw

    canonical_map_path = Path(str(args.sae_canonical_map)) if args.sae_canonical_map else None
    resolved_average_l0 = str(args.sae_average_l0).strip() if args.sae_average_l0 else None
    if not resolved_average_l0 and canonical_map_path:
        resolved_average_l0 = _extract_average_l0_from_canonical_map(
            canonical_map_path=canonical_map_path,
            layer_id=str(layer_id),
            width=width,
        )
    if not resolved_average_l0:
        resolved_average_l0 = _extract_average_l0_from_path_suffix(normalized)
    if not resolved_average_l0:
        return raw

    root = normalized[: layer_match.start()]
    return f"{root}/layer_{layer_id}/width_{width}/average_l0_{resolved_average_l0}"


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
    parser.add_argument(
        "--target-total-features",
        type=int,
        default=None,
        help=(
            "If set, prioritize --target-pairs/--target-pairs-file, then randomly "
            "sample remaining features from TARGET_LAYERS until this total is reached. "
            "When omitted, uses --sample-per-layer behavior."
        ),
    )
    parser.add_argument(
        "--target-pairs",
        type=str,
        nargs="*",
        default=None,
        help="Optional extra layer-feature pairs, e.g. 6,12345 6:12346.",
    )
    parser.add_argument(
        "--target-pairs-file",
        type=str,
        default=None,
        help="Optional file with one pair per line: layer,feature or layer:feature.",
    )
    parser.add_argument("--input-activation-max-rounds", type=int, default=1)
    parser.add_argument("--input-expansion-max-rounds", type=int, default=1)
    parser.add_argument("--llm-generation-model", default=None, help="Forwarded to run_single/workflow.")
    parser.add_argument("--llm-judge-model", default=None, help="Forwarded to run_single/workflow.")
    parser.add_argument("--llm-api-key-file", default=None, help="Forwarded to run_single/workflow.")
    parser.add_argument("--input-api-key-file", default=None, help="Forwarded to run_single/final-eval.")
    parser.add_argument("--output-api-key-file", default=None, help="Forwarded to run_single/final-eval.")
    parser.add_argument("--model-checkpoint-path", default=None, help="Forwarded to workflow_runner.py.")
    parser.add_argument("--sae-path", default=None, help="Forwarded to workflow_runner.py.")
    parser.add_argument("--sae-release", default=None, help="Forwarded to workflow_runner.py.")
    parser.add_argument("--sae-average-l0", default=None, help="Forwarded to workflow_runner.py.")
    parser.add_argument("--sae-canonical-map", default=None, help="Forwarded to workflow_runner.py.")
    parser.add_argument("--device", default=None, help="Forwarded to workflow_runner.py.")
    parser.add_argument("--num-hypothesis", type=int, default=3)
    parser.add_argument(
        "--generation-mode",
        choices=["single_call", "iterative"],
        default="single_call",
        help="Passed to workflow_runner.py --generation-mode.",
    )
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
    parser.add_argument(
        "--observation-source",
        choices=["neuronpedia", "bos_token"],
        default="neuronpedia",
        help="Passed to run_single_input_history_scope_eval.py --observation-source.",
    )
    parser.add_argument(
        "--bos-token-observation-root",
        default="initial_observation",
        help="Passed to run_single_input_history_scope_eval.py --bos-token-observation-root.",
    )
    parser.add_argument(
        "--enable-bos-token-semantic-cluster",
        action="store_true",
        help="Passed to run_single_input_history_scope_eval.py --enable-bos-token-semantic-cluster.",
    )
    parser.add_argument(
        "--input-non-activation-context-count",
        type=int,
        default=5,
        help="Passed to final_explanation_evaluation_runner.py --input-non-activation-context-count.",
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
        help=(
            "Base manifest path. Actual output inserts one more directory level using run-end timestamp, "
            "e.g. logs/<end_ts>/batch_input_history_scope_plan.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(str(args.output_root))
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    manual_pairs = _collect_manual_pairs(args)
    selection = _build_selection(
        output_root=output_root,
        sae_name=str(args.sae_name),
        seed=int(args.seed),
        sample_per_layer=int(args.sample_per_layer),
        manual_pairs=manual_pairs,
        target_total_features=(
            None if args.target_total_features is None else int(args.target_total_features)
        ),
        width=str(args.width),
    )

    print("Selected features:")
    for layer_id, feature_id in selection:
        print(f"- layer={layer_id}, feature={feature_id}")

    records: List[RunRecord] = []
    tasks: List[Tuple[int, int, str, bool]] = []
    for layer_id, feature_id in selection:
        tasks.extend(
            (layer_id, feature_id, history_scope, False)
            for history_scope in HISTORY_SCOPES
        )
        tasks.append((layer_id, feature_id, "all_hypotheses", True))
    for layer_id, feature_id, history_scope, merge_enabled in tqdm(
        tasks,
        desc="Batch input history scope eval",
        unit="task",
    ):
            merge_tag = "_merge" if merge_enabled else ""
            timestamp = f"{now_tag}_l{layer_id}_f{feature_id}_{history_scope}{merge_tag}"
            single_summary_path = (
                PROJECT_ROOT
                / "logs"
                / "batch_input_history_scope_eval_single_summaries"
                / now_tag
                / f"l{layer_id}_f{feature_id}_{history_scope}{merge_tag}.json"
            )
            single_cmd = [
                str(args.python_exe),
                str(PROJECT_ROOT / "run_single_input_history_scope_eval.py"),
                "--layer-id",
                str(layer_id),
                "--feature-id",
                str(feature_id),
                "--summary-output",
                str(single_summary_path),
                "--timestamp",
                timestamp,
                "--side",
                "input",
                "--history-scope",
                history_scope,
                "--input-activation-max-rounds",
                str(args.input_activation_max_rounds),
                "--input-expansion-max-rounds",
                str(args.input_expansion_max_rounds),
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
                "--observation-source",
                str(args.observation_source),
                "--bos-token-observation-root",
                str(args.bos_token_observation_root),
                "--sae-name",
                str(args.sae_name),
                "--input-non-activation-context-count",
                str(args.input_non_activation_context_count),
            ]
            if merge_enabled:
                single_cmd.append("--enable-hypothesis-merge")
            if bool(args.enable_bos_token_semantic_cluster):
                single_cmd.append("--enable-bos-token-semantic-cluster")
            if args.top_m is not None:
                single_cmd.extend(["--top-m", str(args.top_m)])
            if args.llm_generation_model:
                single_cmd.extend(["--llm-generation-model", str(args.llm_generation_model)])
            if args.llm_judge_model:
                single_cmd.extend(["--llm-judge-model", str(args.llm_judge_model)])
            if args.llm_api_key_file:
                single_cmd.extend(["--llm-api-key-file", str(args.llm_api_key_file)])
            if args.input_api_key_file:
                single_cmd.extend(["--input-api-key-file", str(args.input_api_key_file)])
            if args.output_api_key_file:
                single_cmd.extend(["--output-api-key-file", str(args.output_api_key_file)])
            if args.model_checkpoint_path:
                single_cmd.extend(["--model-checkpoint-path", str(args.model_checkpoint_path)])
            resolved_sae_path = _resolve_pair_sae_path(args=args, layer_id=layer_id)
            if resolved_sae_path:
                single_cmd.extend(["--sae-path", resolved_sae_path])
            if args.sae_release:
                single_cmd.extend(["--sae-release", str(args.sae_release)])
            if args.sae_average_l0:
                single_cmd.extend(["--sae-average-l0", str(args.sae_average_l0)])
            if args.sae_canonical_map:
                single_cmd.extend(["--sae-canonical-map", str(args.sae_canonical_map)])
            if args.device:
                single_cmd.extend(["--device", str(args.device)])
            if bool(args.force_run_input_eval):
                single_cmd.append("--force-run-input-eval")
            single_returncode, _ = _run_command(single_cmd, dry_run=bool(args.dry_run))

            workflow_code = 0 if bool(args.dry_run) else single_returncode
            evaluation_code: int | None = 0 if bool(args.dry_run) else (0 if single_returncode == 0 else None)
            workflow_seconds = 0.0
            evaluation_seconds = 0.0
            if single_summary_path.exists():
                try:
                    single_summary = _load_single_summary(single_summary_path)
                    workflow_code = int(single_summary.get("workflow_returncode", workflow_code))
                    raw_eval_code = single_summary.get("evaluation_returncode")
                    evaluation_code = None if raw_eval_code is None else int(raw_eval_code)
                    workflow_seconds = float(single_summary.get("workflow_seconds", workflow_seconds))
                    evaluation_seconds = float(single_summary.get("evaluation_seconds", evaluation_seconds))
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: failed to parse single summary {single_summary_path}: {exc}")
            record = RunRecord(
                layer_id=layer_id,
                feature_id=feature_id,
                history_scope=history_scope,
                merge_enabled=merge_enabled,
                sae_path=resolved_sae_path,
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
        "target_total_features": None if args.target_total_features is None else int(args.target_total_features),
        "target_layers": list(TARGET_LAYERS),
        "manual_pairs": [{"layer_id": l, "feature_id": f} for l, f in manual_pairs],
        "input_activation_max_rounds": int(args.input_activation_max_rounds),
        "input_expansion_max_rounds": int(args.input_expansion_max_rounds),
        "effective_max_rounds": int(args.input_activation_max_rounds) + int(args.input_expansion_max_rounds) - 1,
        "num_hypothesis": int(args.num_hypothesis),
        "generation_mode": str(args.generation_mode),
        "num_input_sentences_per_hypothesis": int(args.num_input_sentences_per_hypothesis),
        "top_m": None if args.top_m is None else int(args.top_m),
        "history_rounds": int(args.history_rounds),
        "width": str(args.width),
        "selection_method": int(args.selection_method),
        "observation_m": int(args.observation_m),
        "observation_n": int(args.observation_n),
        "observation_source": str(args.observation_source),
        "bos_token_observation_root": str(args.bos_token_observation_root),
        "enable_bos_token_semantic_cluster": bool(args.enable_bos_token_semantic_cluster),
        "input_non_activation_context_count": int(args.input_non_activation_context_count),
        "final_run_mode": "input",
        "force_run_input_eval": bool(args.force_run_input_eval),
        "selection": [{"layer_id": l, "feature_id": f} for l, f in selection],
        "records": [r.__dict__ for r in records],
    }
    plan_output_path = _build_plan_output_path(
        Path(str(args.plan_output)),
        end_time=datetime.now(),
    )
    _save_manifest(plan_output_path, summary)
    print(f"Saved manifest: {plan_output_path}")


if __name__ == "__main__":
    main()
