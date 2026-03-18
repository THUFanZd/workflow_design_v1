#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import re
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(__file__).resolve().parent / "outputs" / "gemmascope-res"
DEFAULT_PLOTS_DIR = DEFAULT_ROOT / "summary_plots"

DEFAULT_SCORE_FIELDS = [
    "activation_relative_quality_score",
    "activation_adherence",
    "non_activation_relative_quality_score",
    "non_activation_adherence",
    "boundary_relative_quality_score",
]

DEFAULT_TOKEN_FIELDS = [
    "workflow_prompt_tokens",
    "workflow_completion_tokens",
    "workflow_total_tokens",
]

MODE_ORDER = ["same_hypothesis", "all_hypotheses", "all_hypotheses_merge"]

LAYER_PATTERN = re.compile(r"^layer-(\d+)$")
FEATURE_PATTERN = re.compile(r"^feature-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize gemmascope input-side evaluation results, compare modes, "
            "and visualize score/token usage."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder that contains layer-*/feature-*/.../evaluation_record.json.",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="CSV for per-run details. Default: <root>/summary_details.csv",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="CSV for overall stats (scores + token usage). Default: <root>/summary_stats.csv",
    )
    parser.add_argument(
        "--mode-score-out",
        type=Path,
        default=None,
        help="CSV for score comparison by hypothesis mode. Default: <root>/summary_mode_score_stats.csv",
    )
    parser.add_argument(
        "--mode-token-out",
        type=Path,
        default=None,
        help="CSV for token-usage comparison by hypothesis mode. Default: <root>/summary_mode_token_stats.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Output folder for generated figures.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_SCORE_FIELDS,
        help="Score fields to aggregate and visualize.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return None


def percentile(sorted_values: list[float], ratio: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * ratio
    left_index = int(math.floor(position))
    right_index = int(math.ceil(position))
    if left_index == right_index:
        return sorted_values[left_index]
    left = sorted_values[left_index]
    right = sorted_values[right_index]
    return left + (right - left) * (position - left_index)


def extract_layer_feature(path: Path) -> tuple[int | None, int | None]:
    layer: int | None = None
    feature: int | None = None
    for part in path.parts:
        layer_match = LAYER_PATTERN.match(part)
        if layer_match:
            layer = int(layer_match.group(1))
        feature_match = FEATURE_PATTERN.match(part)
        if feature_match:
            feature = int(feature_match.group(1))
    return layer, feature


def detect_hypothesis_mode(run_dir: str) -> str:
    if "all_hypotheses_merge" in run_dir:
        return "all_hypotheses_merge"
    if "all_hypotheses" in run_dir:
        return "all_hypotheses"
    if "same_hypothesis" in run_dir:
        return "same_hypothesis"
    return "unknown"


def expected_workflow_result_path(layer: int | None, feature: int | None, run_dir: str) -> Path | None:
    if layer is None or feature is None:
        return None
    return (
        PROJECT_ROOT
        / "logs"
        / f"{layer}_{feature}"
        / run_dir
        / "final_result"
        / f"layer{layer}-feature{feature}-final-result.json"
    )


def load_workflow_result(
    layer: int | None,
    feature: int | None,
    run_dir: str,
    cache: dict[str, dict[str, Any] | None],
) -> tuple[dict[str, Any] | None, str | None]:
    path = expected_workflow_result_path(layer, feature, run_dir)
    if path is None:
        return None, None
    cache_key = str(path)
    if cache_key in cache:
        payload = cache[cache_key]
        return payload, cache_key if payload is not None else None

    if not path.exists():
        cache[cache_key] = None
        return None, None

    try:
        payload = load_json(path)
    except Exception:  # noqa: BLE001
        cache[cache_key] = None
        return None, None

    cache[cache_key] = payload
    return payload, cache_key


def extract_workflow_fields(workflow_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(workflow_payload, dict):
        return {
            "workflow_prompt_tokens": None,
            "workflow_completion_tokens": None,
            "workflow_total_tokens": None,
            "workflow_best_hypothesis": "",
            "workflow_executed_rounds": None,
        }

    usage = workflow_payload.get("token_usage_this_run", {})
    if not isinstance(usage, dict):
        usage = {}
    best_hyp = workflow_payload.get("input_side_best_hypothesis", {})
    if not isinstance(best_hyp, dict):
        best_hyp = {}

    return {
        "workflow_prompt_tokens": coerce_float(usage.get("prompt_tokens")),
        "workflow_completion_tokens": coerce_float(usage.get("completion_tokens")),
        "workflow_total_tokens": coerce_float(usage.get("total_tokens")),
        "workflow_best_hypothesis": str(best_hyp.get("hypothesis", "")),
        "workflow_executed_rounds": workflow_payload.get("executed_rounds"),
    }


def build_records(eval_record_paths: list[Path]) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[tuple[Path, str]] = []
    workflow_cache: dict[str, dict[str, Any] | None] = {}

    for eval_path in eval_record_paths:
        try:
            payload = load_json(eval_path)
        except Exception as exc:  # noqa: BLE001
            errors.append((eval_path, str(exc)))
            continue

        scores = payload.get("scores")
        if not isinstance(scores, dict):
            scores = {}

        layer, feature = extract_layer_feature(eval_path)
        feature_payload = payload.get("feature")
        if isinstance(feature_payload, dict):
            source_text = str(feature_payload.get("source", ""))
            source_head = source_text.split("-")[0]
            index_text = str(feature_payload.get("index", ""))
            if layer is None and source_head.isdigit():
                layer = int(source_head)
            if feature is None and index_text.isdigit():
                feature = int(index_text)

        run_dir = eval_path.parent.name
        mode = detect_hypothesis_mode(run_dir)

        workflow_payload, workflow_path = load_workflow_result(layer, feature, run_dir, workflow_cache)
        workflow_fields = extract_workflow_fields(workflow_payload)

        hypothesis = payload.get("hypothesis", "")
        reference_hypotheses = payload.get("reference_explanations", [])
        if reference_hypotheses is None:
            reference_hypotheses = []
        if not isinstance(reference_hypotheses, list):
            reference_hypotheses = [reference_hypotheses]

        row: dict[str, Any] = {
            "layer": layer,
            "feature": feature,
            "run_dir": run_dir,
            "hypothesis_mode": mode,
            "my_hypothesis": str(hypothesis) if hypothesis is not None else "",
            "neuronpedia_hypotheses": safe_json_text(reference_hypotheses),
            "scores": safe_json_text(scores),
            "evaluation_record_path": str(eval_path),
            "workflow_final_result_path": workflow_path or "",
            "__scores_obj__": scores,
        }

        row.update(workflow_fields)

        for score_key, score_value in scores.items():
            row[score_key] = score_value

        rows.append(row)

    return rows, errors


def build_stats(rows: list[dict[str, Any]], fields: list[str], *, field_type: str) -> list[dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    total_count = len(rows)

    for field in fields:
        numeric_values: list[float] = []
        for row in rows:
            value = row.get(field)
            numeric = coerce_float(value)
            if numeric is not None:
                numeric_values.append(numeric)

        valid_count = len(numeric_values)
        null_count = total_count - valid_count
        if valid_count > 0:
            sorted_values = sorted(numeric_values)
            mean_val = statistics.fmean(numeric_values)
            std_val = statistics.stdev(numeric_values) if valid_count > 1 else 0.0
            min_val = sorted_values[0]
            p25_val = percentile(sorted_values, 0.25)
            median_val = percentile(sorted_values, 0.50)
            p75_val = percentile(sorted_values, 0.75)
            max_val = sorted_values[-1]
        else:
            mean_val = None
            std_val = None
            min_val = None
            p25_val = None
            median_val = None
            p75_val = None
            max_val = None

        stats_rows.append(
            {
                "field_type": field_type,
                "field": field,
                "count": total_count,
                "valid_count": valid_count,
                "null_count": null_count,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "p25": p25_val,
                "median": median_val,
                "p75": p75_val,
                "max": max_val,
            }
        )

    return stats_rows


def build_mode_stats(rows: list[dict[str, Any]], fields: list[str], *, field_type: str) -> list[dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        mode_rows = [row for row in rows if row.get("hypothesis_mode") == mode]
        total_count = len(mode_rows)
        for field in fields:
            numeric_values: list[float] = []
            for row in mode_rows:
                numeric = coerce_float(row.get(field))
                if numeric is not None:
                    numeric_values.append(numeric)

            valid_count = len(numeric_values)
            null_count = total_count - valid_count
            if valid_count > 0:
                sorted_values = sorted(numeric_values)
                mean_val = statistics.fmean(numeric_values)
                std_val = statistics.stdev(numeric_values) if valid_count > 1 else 0.0
                min_val = sorted_values[0]
                p25_val = percentile(sorted_values, 0.25)
                median_val = percentile(sorted_values, 0.50)
                p75_val = percentile(sorted_values, 0.75)
                max_val = sorted_values[-1]
            else:
                mean_val = None
                std_val = None
                min_val = None
                p25_val = None
                median_val = None
                p75_val = None
                max_val = None

            stats_rows.append(
                {
                    "field_type": field_type,
                    "hypothesis_mode": mode,
                    "field": field,
                    "count": total_count,
                    "valid_count": valid_count,
                    "null_count": null_count,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "p25": p25_val,
                    "median": median_val,
                    "p75": p75_val,
                    "max": max_val,
                }
            )
    return stats_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def print_stats_table(stats_rows: list[dict[str, Any]], *, title: str) -> None:
    print(title)
    headers = ["field_type", "field", "count", "valid", "null", "mean", "std", "min", "p25", "median", "p75", "max"]
    print(" | ".join(headers))
    print("-|-|-|-|-|-|-|-|-|-|-|-")
    for row in stats_rows:
        print(
            " | ".join(
                [
                    str(row.get("field_type", "")),
                    str(row["field"]),
                    str(row["count"]),
                    str(row["valid_count"]),
                    str(row["null_count"]),
                    format_number(row["mean"]),
                    format_number(row["std"]),
                    format_number(row["min"]),
                    format_number(row["p25"]),
                    format_number(row["median"]),
                    format_number(row["p75"]),
                    format_number(row["max"]),
                ]
            )
        )


def _mode_label(mode: str) -> str:
    mapping = {
        "same_hypothesis": "same",
        "all_hypotheses": "all",
        "all_hypotheses_merge": "all+merge",
    }
    return mapping.get(mode, mode)


def plot_score_boxplots(rows: list[dict[str, Any]], score_fields: list[str], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_modes = [mode for mode in MODE_ORDER if any(row.get("hypothesis_mode") == mode for row in rows)]
    if not valid_modes:
        return False

    n = len(score_fields)
    cols = 2
    rows_n = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 4 * rows_n))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, field in enumerate(score_fields):
        ax = axes_flat[idx]
        series: list[list[float]] = []
        labels: list[str] = []
        for mode in valid_modes:
            values = [coerce_float(row.get(field)) for row in rows if row.get("hypothesis_mode") == mode]
            numeric_values = [v for v in values if v is not None]
            if numeric_values:
                series.append(numeric_values)
                labels.append(_mode_label(mode))

        if not series:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(field)
            ax.set_xticks([])
            continue

        ax.boxplot(series, labels=labels, showmeans=True)
        ax.set_title(field)
        ax.set_xlabel("Hypothesis mode")
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Score Comparison Across Hypothesis Modes", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_token_usage_bars(rows: list[dict[str, Any]], token_fields: list[str], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_modes = [mode for mode in MODE_ORDER if any(row.get("hypothesis_mode") == mode for row in rows)]
    if not valid_modes:
        return False

    means_by_mode: dict[str, list[float]] = {}
    for mode in valid_modes:
        mode_means: list[float] = []
        mode_rows = [row for row in rows if row.get("hypothesis_mode") == mode]
        for field in token_fields:
            values = [coerce_float(row.get(field)) for row in mode_rows]
            numeric_values = [v for v in values if v is not None]
            mode_means.append(statistics.fmean(numeric_values) if numeric_values else 0.0)
        means_by_mode[mode] = mode_means

    x_labels = ["prompt", "completion", "total"]
    x = list(range(len(token_fields)))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, mode in enumerate(valid_modes):
        y = means_by_mode[mode]
        offset = (idx - (len(valid_modes) - 1) / 2) * width
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, y, width=width, label=_mode_label(mode))
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean tokens")
    ax.set_xlabel("Token usage type")
    ax.set_title("Workflow Token Usage Comparison Across Hypothesis Modes")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    details_out = args.details_out or (root / "summary_details.csv")
    stats_out = args.stats_out or (root / "summary_stats.csv")
    mode_score_out = args.mode_score_out or (root / "summary_mode_score_stats.csv")
    mode_token_out = args.mode_token_out or (root / "summary_mode_token_stats.csv")
    plots_dir = args.plots_dir.resolve()

    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    eval_record_paths = sorted(root.rglob("evaluation_record.json"))
    if not eval_record_paths:
        raise SystemExit(f"No evaluation_record.json found under: {root}")

    rows, errors = build_records(eval_record_paths)
    if not rows:
        raise SystemExit("No valid evaluation records were parsed.")

    score_keys = sorted(
        {
            key
            for row in rows
            for key in row["__scores_obj__"].keys()
        }
    )

    detail_headers = [
        "layer",
        "feature",
        "run_dir",
        "hypothesis_mode",
        "my_hypothesis",
        "neuronpedia_hypotheses",
        "workflow_best_hypothesis",
        "workflow_executed_rounds",
        "workflow_prompt_tokens",
        "workflow_completion_tokens",
        "workflow_total_tokens",
        "workflow_final_result_path",
        "scores",
        *score_keys,
        "evaluation_record_path",
    ]
    write_csv(details_out, rows, detail_headers)

    overall_score_stats = build_stats(rows, args.fields, field_type="score")
    overall_token_stats = build_stats(rows, DEFAULT_TOKEN_FIELDS, field_type="token")
    overall_stats = overall_score_stats + overall_token_stats
    write_csv(
        stats_out,
        overall_stats,
        [
            "field_type",
            "field",
            "count",
            "valid_count",
            "null_count",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
        ],
    )

    mode_score_stats = build_mode_stats(rows, args.fields, field_type="score")
    mode_token_stats = build_mode_stats(rows, DEFAULT_TOKEN_FIELDS, field_type="token")
    write_csv(
        mode_score_out,
        mode_score_stats,
        [
            "field_type",
            "hypothesis_mode",
            "field",
            "count",
            "valid_count",
            "null_count",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
        ],
    )
    write_csv(
        mode_token_out,
        mode_token_stats,
        [
            "field_type",
            "hypothesis_mode",
            "field",
            "count",
            "valid_count",
            "null_count",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
        ],
    )

    score_plot_path = plots_dir / "score_mode_boxplots.png"
    token_plot_path = plots_dir / "token_usage_mode_bars.png"
    score_plot_ok = plot_score_boxplots(rows, args.fields, score_plot_path)
    token_plot_ok = plot_token_usage_bars(rows, DEFAULT_TOKEN_FIELDS, token_plot_path)

    print(f"Scanned records: {len(eval_record_paths)}")
    print(f"Valid records: {len(rows)}")
    print(f"Detail CSV: {details_out}")
    print(f"Overall stats CSV: {stats_out}")
    print(f"Mode score stats CSV: {mode_score_out}")
    print(f"Mode token stats CSV: {mode_token_out}")
    if score_plot_ok:
        print(f"Score comparison plot: {score_plot_path}")
    else:
        print("Score comparison plot: skipped (matplotlib unavailable or no valid mode data)")
    if token_plot_ok:
        print(f"Token usage plot: {token_plot_path}")
    else:
        print("Token usage plot: skipped (matplotlib unavailable or no valid mode data)")

    if errors:
        print(f"Parse errors: {len(errors)}")
        for path, reason in errors[:10]:
            print(f"  - {path}: {reason}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print()
    print_stats_table(overall_stats, title="Overall statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
