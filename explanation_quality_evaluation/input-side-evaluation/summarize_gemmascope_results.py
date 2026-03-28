#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(__file__).resolve().parent / "outputs" / "gemmascope-res"
DEFAULT_PLOTS_DIR = DEFAULT_ROOT / "summary_plots"
DEFAULT_ROUND_PLOTS_DIR = DEFAULT_PLOTS_DIR / "round_trends"

DEFAULT_METRIC_FIELDS = [
    "relative_quality_score_non_zero_rate",
    "relative_quality_score_boundary_non_activation_rate",
]

DEFAULT_TOKEN_FIELDS = [
    "workflow_prompt_tokens",
    "workflow_completion_tokens",
    "workflow_total_tokens",
]

MODE_ORDER = ["same_hypothesis", "all_hypotheses", "all_hypotheses_merge"]

LAYER_PATTERN = re.compile(r"^layer-(\d+)$")
FEATURE_PATTERN = re.compile(r"^feature-(\d+)$")

# summarize for existing results
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize gemmascope input-side final-evaluation results, compare modes, "
            "and visualize score/token usage and per-round score trends."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder that contains layer-*/feature-*/.../result.json.",
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
        "--round-plots-dir",
        type=Path,
        default=DEFAULT_ROUND_PLOTS_DIR,
        help="Output folder for per-run round trend figures.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_METRIC_FIELDS,
        help="Relative-quality score fields to aggregate and visualize.",
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
        / str(layer)
        / str(feature)
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


def detect_hypothesis_mode_from_workflow(
    run_dir: str,
    workflow_payload: dict[str, Any] | None,
) -> str:
    mode = detect_hypothesis_mode(run_dir)
    if mode != "unknown":
        return mode
    if not isinstance(workflow_payload, dict):
        return "unknown"

    merge_mode = str(workflow_payload.get("hypothesis_merge_mode", "")).strip().lower()
    if merge_mode not in {"", "off", "none"}:
        return "all_hypotheses_merge"

    history_rounds = workflow_payload.get("history_rounds")
    if isinstance(history_rounds, int):
        return "all_hypotheses" if history_rounds > 0 else "same_hypothesis"

    return "unknown"


def _extract_relative_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    non_zero = coerce_float(payload.get("relative_quality_score_non_zero_rate"))
    boundary = coerce_float(payload.get("relative_quality_score_boundary_non_activation_rate"))

    return {
        "relative_quality_score_non_zero_rate": non_zero,
        "relative_quality_score_boundary_non_activation_rate": boundary,
    }


def _extract_round_scores_from_execution(execution_payload: dict[str, Any]) -> tuple[float | None, float | None]:
    input_side = execution_payload.get("input_side_execution", {})
    if not isinstance(input_side, dict):
        return None, None

    hypothesis_results_raw = input_side.get("hypothesis_results", [])
    hypothesis_results = [item for item in hypothesis_results_raw if isinstance(item, dict)]
    if not hypothesis_results:
        return (
            coerce_float(input_side.get("overall_score_non_zero_rate")),
            coerce_float(input_side.get("overall_score_boundary_non_activation_rate")),
        )

    def rank_key(item: dict[str, Any]) -> tuple[float, int]:
        score_non_zero = coerce_float(item.get("score_non_zero_rate")) or 0.0
        score_boundary = coerce_float(item.get("score_boundary_non_activation_rate")) or 0.0
        combined = score_non_zero + score_boundary
        hypothesis_index = int(item.get("hypothesis_index", 10**9))
        return combined, -hypothesis_index

    best = max(hypothesis_results, key=rank_key)
    return (
        coerce_float(best.get("score_non_zero_rate")),
        coerce_float(best.get("score_boundary_non_activation_rate")),
    )


def load_round_series(
    *,
    layer: int | None,
    feature: int | None,
    run_dir: str,
    workflow_payload: dict[str, Any] | None,
    cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if layer is None or feature is None:
        return []
    workflow_path = expected_workflow_result_path(layer, feature, run_dir)
    if workflow_path is None:
        return []
    cache_key = str(workflow_path)
    if cache_key in cache:
        return cache[cache_key]
    if not workflow_path.exists():
        cache[cache_key] = []
        return []

    workflow_dir = workflow_path.parents[1]
    executed_rounds = 0
    if isinstance(workflow_payload, dict):
        maybe_executed = workflow_payload.get("executed_rounds")
        if isinstance(maybe_executed, int):
            executed_rounds = max(0, maybe_executed)

    series: list[dict[str, Any]] = []
    for round_index in range(0, executed_rounds + 1):
        execution_path = (
            workflow_dir
            / f"round_{round_index}"
            / f"layer{layer}-feature{feature}-experiments-execution.json"
        )
        if not execution_path.exists():
            continue
        try:
            execution_payload = load_json(execution_path)
        except Exception:  # noqa: BLE001
            continue

        score_non_zero, score_boundary = _extract_round_scores_from_execution(execution_payload)
        series.append(
            {
                "round_index": round_index,
                "score_non_zero_rate": score_non_zero,
                "score_boundary_non_activation_rate": score_boundary,
            }
        )

    cache[cache_key] = series
    return series


def build_records(result_paths: list[Path]) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[tuple[Path, str]] = []
    workflow_cache: dict[str, dict[str, Any] | None] = {}
    round_series_cache: dict[str, list[dict[str, Any]]] = {}

    for result_path in result_paths:
        try:
            payload = load_json(result_path)
        except Exception as exc:  # noqa: BLE001
            errors.append((result_path, str(exc)))
            continue

        # Only summarize the new workflow-style evaluation results.
        if (
            "relative_quality_score_non_zero_rate" not in payload
            and "relative_quality_score_boundary_non_activation_rate" not in payload
        ):
            continue

        layer, feature = extract_layer_feature(result_path)
        run_dir = result_path.parent.name

        workflow_payload, workflow_path = load_workflow_result(layer, feature, run_dir, workflow_cache)
        mode = detect_hypothesis_mode_from_workflow(run_dir, workflow_payload)
        workflow_fields = extract_workflow_fields(workflow_payload)
        relative_metrics = _extract_relative_metrics(payload)
        round_series = load_round_series(
            layer=layer,
            feature=feature,
            run_dir=run_dir,
            workflow_payload=workflow_payload,
            cache=round_series_cache,
        )

        row: dict[str, Any] = {
            "layer": layer,
            "feature": feature,
            "run_dir": run_dir,
            "hypothesis_mode": mode,
            "evaluation_result_path": str(result_path),
            "workflow_final_result_path": workflow_path or "",
            "relative_quality_score_non_zero_rate": relative_metrics["relative_quality_score_non_zero_rate"],
            "relative_quality_score_boundary_non_activation_rate": relative_metrics[
                "relative_quality_score_boundary_non_activation_rate"
            ],
            "round_score_series": safe_json_text(round_series),
            "__round_series_obj__": round_series,
        }
        row.update(workflow_fields)
        rows.append(row)

    return rows, errors


def build_stats(rows: list[dict[str, Any]], fields: list[str], *, field_type: str) -> list[dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    total_count = len(rows)

    for field in fields:
        numeric_values: list[float] = []
        for row in rows:
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


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", text)
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "unknown"


def feature_round_plot_dir(evaluation_result_path: str) -> Path | None:
    result_path = Path(evaluation_result_path)
    feature_dir = result_path.parent.parent
    if feature_dir == result_path.parent:
        return None
    return feature_dir / "round_trends"


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
        ax.set_ylabel("Relative quality")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Relative-Quality Score Comparison Across Hypothesis Modes", fontsize=14)
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


def plot_round_trends_per_run(rows: list[dict[str, Any]]) -> int:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return 0

    created = 0
    for row in rows:
        series_raw = row.get("__round_series_obj__")
        series = [item for item in series_raw if isinstance(item, dict)] if isinstance(series_raw, list) else []
        if not series:
            continue
        output_dir = feature_round_plot_dir(str(row.get("evaluation_result_path", "")))
        if output_dir is None:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        rounds: list[int] = []
        non_zero_vals: list[float | None] = []
        boundary_vals: list[float | None] = []
        for item in sorted(series, key=lambda x: int(x.get("round_index", 0))):
            rounds.append(int(item.get("round_index", 0)))
            non_zero_vals.append(coerce_float(item.get("score_non_zero_rate")))
            boundary_vals.append(coerce_float(item.get("score_boundary_non_activation_rate")))

        if not rounds:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(rounds, non_zero_vals, marker="o", label="score_non_zero_rate")
        ax.plot(rounds, boundary_vals, marker="o", label="score_boundary_non_activation_rate")
        ax.set_xlabel("Round index")
        ax.set_ylabel("Score")
        ax.set_title(
            f"Round Score Trend | layer={row.get('layer')} feature={row.get('feature')} "
            f"mode={_mode_label(str(row.get('hypothesis_mode', 'unknown')))}"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        ax.set_xticks(rounds)
        fig.tight_layout()

        filename = _safe_filename(
            f"l{row.get('layer')}_f{row.get('feature')}_{row.get('run_dir')}_round_trend.png"
        )
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        created += 1
    return created


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    details_out = args.details_out or (root / "summary_details.csv")
    stats_out = args.stats_out or (root / "summary_stats.csv")
    mode_score_out = args.mode_score_out or (root / "summary_mode_score_stats.csv")
    mode_token_out = args.mode_token_out or (root / "summary_mode_token_stats.csv")
    plots_dir = args.plots_dir.resolve()
    round_plots_dir = args.round_plots_dir.resolve()

    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    result_paths = sorted(root.rglob("result.json"))
    if not result_paths:
        raise SystemExit(f"No result.json found under: {root}")

    rows, errors = build_records(result_paths)
    if not rows:
        raise SystemExit("No valid result records were parsed.")

    detail_headers = [
        "layer",
        "feature",
        "run_dir",
        "hypothesis_mode",
        "relative_quality_score_non_zero_rate",
        "relative_quality_score_boundary_non_activation_rate",
        "workflow_best_hypothesis",
        "workflow_executed_rounds",
        "workflow_prompt_tokens",
        "workflow_completion_tokens",
        "workflow_total_tokens",
        "round_score_series",
        "workflow_final_result_path",
        "evaluation_result_path",
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

    score_plot_path = plots_dir / "relative_quality_mode_boxplots.png"
    token_plot_path = plots_dir / "token_usage_mode_bars.png"
    score_plot_ok = plot_score_boxplots(rows, args.fields, score_plot_path)
    token_plot_ok = plot_token_usage_bars(rows, DEFAULT_TOKEN_FIELDS, token_plot_path)
    round_plot_count = plot_round_trends_per_run(rows)

    print(f"Scanned result files: {len(result_paths)}")
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
    print("Per-run round trend plots are written under each feature directory as round_trends/*.png")
    print(f"Per-run round trend plots: {round_plot_count}")

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
