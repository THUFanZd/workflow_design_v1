#!/usr/bin/env python
"""Plot per-hypothesis round scores from workflow logs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROUND_DIR_PATTERN = re.compile(r"^round_(\d+)$")


@dataclass
class RoundScore:
    round_idx: int
    values: dict[str, float | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse logs/{layer-id}/{feature-id}/{timestamp}/ and draw "
            "per-hypothesis score-vs-round line charts."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory (default: logs).",
    )
    parser.add_argument(
        "--side",
        choices=["input", "output", "both"],
        default="both",
        help="Which side to draw (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate charts even if graph images already exist.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Image dpi (default: 160).",
    )
    return parser.parse_args()


def collect_run_dirs(logs_root: Path) -> list[Path]:
    if not logs_root.exists():
        return []
    candidates: list[Path] = []
    candidates.extend(sorted(p for p in logs_root.glob("*/*/*") if p.is_dir()))
    candidates.extend(sorted(p for p in logs_root.glob("*/*") if p.is_dir()))
    run_dirs: list[Path] = []
    seen: set[str] = set()
    for run_dir in candidates:
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        if any(p.is_dir() and ROUND_DIR_PATTERN.match(p.name) for p in run_dir.iterdir()):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def extract_round_json(run_dir: Path, round_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(round_dir.glob("*-experiments-execution.json"))
    if not candidates:
        return None
    path = candidates[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] invalid json, skip: {path} ({exc})")
        return None


def should_skip_side(run_dir: Path, side: str, force: bool) -> bool:
    side_dir = run_dir / "graph" / side
    if force:
        return False
    if not side_dir.exists():
        return False
    return any(side_dir.glob("*.png"))


def get_output_score_name(payload: dict[str, Any]) -> str:
    for key in ("output_score_name",):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    output_exec = payload.get("output_side_execution") or {}
    value = output_exec.get("output_score_name")
    if isinstance(value, str) and value:
        return value
    return "score_blind_accuracy"


def collect_side_round_scores(run_dir: Path, side: str) -> dict[int, list[RoundScore]]:
    series: dict[int, list[RoundScore]] = {}
    round_dirs = sorted(
        (
            p
            for p in run_dir.iterdir()
            if p.is_dir() and ROUND_DIR_PATTERN.match(p.name) is not None
        ),
        key=lambda p: int(ROUND_DIR_PATTERN.match(p.name).group(1)),  # type: ignore[union-attr]
    )
    output_score_name = "score_blind_accuracy"

    for round_dir in round_dirs:
        match = ROUND_DIR_PATTERN.match(round_dir.name)
        if match is None:
            continue
        round_idx = int(match.group(1))
        payload = extract_round_json(run_dir, round_dir)
        if payload is None:
            continue

        if side == "input":
            side_exec = payload.get("input_side_execution") or {}
            hyp_results = side_exec.get("hypothesis_results") or []
            for item in hyp_results:
                hyp_idx = item.get("hypothesis_index")
                if not isinstance(hyp_idx, int):
                    continue
                values = {
                    "score_non_zero_rate": _safe_float(
                        item.get("score_activation_rate")
                        if "score_activation_rate" in item
                        else item.get("score_non_zero_rate")
                    ),
                }
                series.setdefault(hyp_idx, []).append(
                    RoundScore(round_idx=round_idx, values=values)
                )
        else:
            output_score_name = get_output_score_name(payload)
            side_exec = payload.get("output_side_execution") or {}
            hyp_results = side_exec.get("hypothesis_results") or []
            for item in hyp_results:
                hyp_idx = item.get("hypothesis_index")
                if not isinstance(hyp_idx, int):
                    continue
                score_val = _safe_float(item.get(output_score_name))
                if score_val is None:
                    score_val = _guess_output_score(item)
                values = {output_score_name: score_val}
                series.setdefault(hyp_idx, []).append(
                    RoundScore(round_idx=round_idx, values=values)
                )
    return series


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _guess_output_score(item: dict[str, Any]) -> float | None:
    for key, value in item.items():
        if isinstance(key, str) and key.startswith("score_"):
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
    return None


def plot_input_hypothesis(
    out_path: Path, hyp_idx: int, round_scores: list[RoundScore], dpi: int
) -> None:
    round_scores = sorted(round_scores, key=lambda x: x.round_idx)
    xs = [it.round_idx for it in round_scores]
    y1 = [it.values.get("score_non_zero_rate") for it in round_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, y1, marker="o", label="score_non_zero_rate")
    plt.xticks(xs)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("round")
    plt.ylabel("score")
    plt.title(f"input hypothesis #{hyp_idx} score vs round")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_output_hypothesis(
    out_path: Path, hyp_idx: int, round_scores: list[RoundScore], dpi: int
) -> None:
    round_scores = sorted(round_scores, key=lambda x: x.round_idx)
    xs = [it.round_idx for it in round_scores]
    score_name = next(iter(round_scores[0].values.keys()))
    ys = [it.values.get(score_name) for it in round_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", label=score_name)
    plt.xticks(xs)
    plt.xlabel("round")
    plt.ylabel("score")
    plt.title(f"output hypothesis #{hyp_idx} score vs round")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def generate_for_side(run_dir: Path, side: str, dpi: int) -> tuple[int, int]:
    scores = collect_side_round_scores(run_dir, side)
    out_dir = run_dir / "graph" / side
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scores:
        return 0, 0

    generated = 0
    skipped_empty = 0
    for hyp_idx, round_scores in sorted(scores.items()):
        if not round_scores:
            skipped_empty += 1
            continue
        out_path = out_dir / f"hypothesis_{hyp_idx}_round_scores.png"
        if side == "input":
            plot_input_hypothesis(out_path, hyp_idx, round_scores, dpi)
        else:
            plot_output_hypothesis(out_path, hyp_idx, round_scores, dpi)
        generated += 1
    return generated, skipped_empty


def main() -> None:
    args = parse_args()
    run_dirs = collect_run_dirs(args.logs_root)
    if not run_dirs:
        print(f"[INFO] no run directories under: {args.logs_root}")
        return

    sides = ["input", "output"] if args.side == "both" else [args.side]
    processed_runs = 0
    skipped_runs = 0
    generated_files = 0

    for run_dir in run_dirs:
        run_has_work = False
        for side in sides:
            if should_skip_side(run_dir, side, args.force):
                print(f"[SKIP] {run_dir} side={side} (existing png files found)")
                skipped_runs += 1
                continue

            run_has_work = True
            generated, skipped_empty = generate_for_side(run_dir, side, args.dpi)
            generated_files += generated
            print(
                f"[DONE] {run_dir} side={side} generated={generated} "
                f"empty_series={skipped_empty}"
            )

        if run_has_work:
            processed_runs += 1

    print(
        f"[SUMMARY] processed_runs={processed_runs} skipped_checks={skipped_runs} "
        f"generated_files={generated_files}"
    )


if __name__ == "__main__":
    main()
