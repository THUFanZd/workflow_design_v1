#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "logs" / "single_run_batch"

RELATIVE_QUALITY_FIELDS = [
    "relative_quality_score_non_zero_rate",
    "relative_quality_score_boundary_non_activation_rate",
    "relative_quality_combined_input_score",
]

TOKEN_USAGE_FIELDS = [
    "input_tokens",
    "output_tokens",
    "total_tokens",
]

ROUND_SCORE_FIELDS = [
    "score_non_zero_rate",
    "score_boundary_non_activation_rate",
    "combined_score",
]


@dataclass
class BatchContext:
    batch_dir: Path
    manifest_path: Path
    analysis_dir: Path
    tables_dir: Path
    figures_dir: Path
    debug_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one run_single_pairs_batch result set. "
            "Input is a batch_manifest.json, outputs are written under <batch_dir>/analysis/."
        )
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Batch directory that contains batch_manifest.json. Default: latest under logs/single_run_batch.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to batch_manifest.json. If set, --batch-dir is ignored.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit with code 0 even if no valid final-evaluation files were found.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def pick_latest_batch_dir(batch_root: Path) -> Path:
    candidates = sorted([p for p in batch_root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No batch directory found under: {batch_root}")
    return candidates[-1]


def build_context(args: argparse.Namespace) -> BatchContext:
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        batch_dir = manifest_path.parent
    else:
        batch_dir = (
            args.batch_dir.resolve()
            if args.batch_dir is not None
            else pick_latest_batch_dir(DEFAULT_BATCH_ROOT.resolve())
        )
        manifest_path = batch_dir / "batch_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    analysis_dir = batch_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures"
    debug_dir = analysis_dir / "debug"
    return BatchContext(
        batch_dir=batch_dir,
        manifest_path=manifest_path,
        analysis_dir=analysis_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        debug_dir=debug_dir,
    )


def extract_timestamp_from_single_summary(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    try:
        payload = load_json(summary_path)
    except Exception:
        return None

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    ts = summary.get("timestamp")
    if isinstance(ts, str) and ts.strip():
        return ts.strip()
    return None


def final_evaluation_path(layer: int, feature: int, timestamp: str) -> Path:
    return (
        PROJECT_ROOT
        / "logs"
        / f"layer-{layer}"
        / f"feature-{feature}"
        / timestamp
        / "final_result"
        / f"layer{layer}-feature{feature}-final-evaluation.json"
    )


def final_result_path(layer: int, feature: int, timestamp: str) -> Path:
    return (
        PROJECT_ROOT
        / "logs"
        / f"layer-{layer}"
        / f"feature-{feature}"
        / timestamp
        / "final_result"
        / f"layer{layer}-feature{feature}-final-result.json"
    )


def execution_round_path(layer: int, feature: int, timestamp: str, round_index: int) -> Path:
    return (
        PROJECT_ROOT
        / "logs"
        / f"layer-{layer}"
        / f"feature-{feature}"
        / timestamp
        / f"round_{round_index}"
        / f"layer{layer}-feature{feature}-experiments-execution.json"
    )


def parse_token_usage(token_usage_obj: Any) -> tuple[float | None, float | None, float | None]:
    if not isinstance(token_usage_obj, dict):
        return None, None, None
    input_tokens = to_float(token_usage_obj.get("prompt_tokens"))
    output_tokens = to_float(token_usage_obj.get("completion_tokens"))
    total_tokens = to_float(token_usage_obj.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def extract_token_usage_for_feature(
    *,
    layer: int,
    feature: int,
    timestamp: str,
    final_eval_payload: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    # Prefer full-run usage from final-result JSON.
    final_result = final_result_path(layer, feature, timestamp)
    if final_result.exists():
        try:
            final_result_payload = load_json(final_result)
        except Exception:
            final_result_payload = {}
        if isinstance(final_result_payload, dict):
            for key in ["token_usage_this_run", "token_usage_total", "token_usage"]:
                input_tokens, output_tokens, total_tokens = parse_token_usage(final_result_payload.get(key))
                if any(v is not None for v in [input_tokens, output_tokens, total_tokens]):
                    return input_tokens, output_tokens, total_tokens

    # Fallback to token usage carried by final-evaluation JSON.
    for key in ["token_usage_this_run", "token_usage_total", "token_usage"]:
        input_tokens, output_tokens, total_tokens = parse_token_usage(final_eval_payload.get(key))
        if any(v is not None for v in [input_tokens, output_tokens, total_tokens]):
            return input_tokens, output_tokens, total_tokens

    input_eval = final_eval_payload.get("input_evaluation")
    if isinstance(input_eval, dict):
        raw_result = input_eval.get("raw_result")
        if isinstance(raw_result, dict):
            input_tokens, output_tokens, total_tokens = parse_token_usage(raw_result.get("token_usage"))
            if any(v is not None for v in [input_tokens, output_tokens, total_tokens]):
                return input_tokens, output_tokens, total_tokens

    output_eval = final_eval_payload.get("output_evaluation")
    if isinstance(output_eval, dict):
        raw_result = output_eval.get("raw_result")
        if isinstance(raw_result, dict):
            input_tokens, output_tokens, total_tokens = parse_token_usage(raw_result.get("token_usage"))
            if any(v is not None for v in [input_tokens, output_tokens, total_tokens]):
                return input_tokens, output_tokens, total_tokens

    return None, None, None


def extract_best_round_score(execution_payload: dict[str, Any]) -> tuple[float | None, float | None]:
    input_side = execution_payload.get("input_side_execution")
    if not isinstance(input_side, dict):
        return None, None

    raw_results = input_side.get("hypothesis_results", [])
    hypothesis_results = [item for item in raw_results if isinstance(item, dict)]
    if not hypothesis_results:
        return (
            to_float(input_side.get("overall_score_non_zero_rate")),
            to_float(input_side.get("overall_score_boundary_non_activation_rate")),
        )

    def key_fn(item: dict[str, Any]) -> tuple[float, int]:
        non_zero = to_float(item.get("score_non_zero_rate")) or 0.0
        boundary = to_float(item.get("score_boundary_non_activation_rate")) or 0.0
        combined = non_zero + boundary
        hyp_idx = int(item.get("hypothesis_index", 10**9))
        return combined, -hyp_idx

    best = max(hypothesis_results, key=key_fn)
    return (
        to_float(best.get("score_non_zero_rate")),
        to_float(best.get("score_boundary_non_activation_rate")),
    )


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def build_relative_quality_stats(
    feature_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    by_layer_rows: list[dict[str, Any]] = []

    for field in RELATIVE_QUALITY_FIELDS:
        values = [to_float(row.get(field)) for row in feature_rows]
        numeric = [v for v in values if v is not None]
        summary = summarize_numeric(numeric)
        overall_rows.append(
            {
                "metric": field,
                "valid_count": summary["count"],
                "mean": summary["mean"],
                "median": summary["median"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
            }
        )

    layer_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        layer_groups[int(row["layer"])].append(row)

    for layer in sorted(layer_groups.keys()):
        rows = layer_groups[layer]
        for field in RELATIVE_QUALITY_FIELDS:
            values = [to_float(row.get(field)) for row in rows]
            numeric = [v for v in values if v is not None]
            summary = summarize_numeric(numeric)
            by_layer_rows.append(
                {
                    "layer": layer,
                    "metric": field,
                    "valid_count": summary["count"],
                    "mean": summary["mean"],
                    "median": summary["median"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                }
            )

    return overall_rows, by_layer_rows


def build_token_usage_stats(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in TOKEN_USAGE_FIELDS:
        values = [to_float(row.get(field)) for row in feature_rows]
        numeric = [v for v in values if v is not None]
        summary = summarize_numeric(numeric)
        rows.append(
            {
                "metric": field,
                "valid_count": summary["count"],
                "sum": sum(numeric) if numeric else None,
                "mean": summary["mean"],
                "median": summary["median"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
            }
        )
    return rows


def build_round_mean_stats(
    round_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_out: list[dict[str, Any]] = []
    by_layer_out: list[dict[str, Any]] = []

    overall_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in round_rows:
        overall_group[int(row["round_index"])].append(row)

    for round_index in sorted(overall_group.keys()):
        rows = overall_group[round_index]
        for metric in ROUND_SCORE_FIELDS:
            vals = [to_float(row.get(metric)) for row in rows]
            numeric = [v for v in vals if v is not None]
            summary = summarize_numeric(numeric)
            overall_out.append(
                {
                    "round_index": round_index,
                    "metric": metric,
                    "n_features": summary["count"],
                    "mean": summary["mean"],
                    "median": summary["median"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                }
            )

    by_layer_group: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in round_rows:
        key = (int(row["layer"]), int(row["round_index"]))
        by_layer_group[key].append(row)

    for key in sorted(by_layer_group.keys()):
        layer, round_index = key
        rows = by_layer_group[key]
        for metric in ROUND_SCORE_FIELDS:
            vals = [to_float(row.get(metric)) for row in rows]
            numeric = [v for v in vals if v is not None]
            summary = summarize_numeric(numeric)
            by_layer_out.append(
                {
                    "layer": layer,
                    "round_index": round_index,
                    "metric": metric,
                    "n_features": summary["count"],
                    "mean": summary["mean"],
                    "median": summary["median"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                }
            )

    return overall_out, by_layer_out


def _plot_round_lines(
    *,
    out_path: Path,
    x_values: list[int],
    series: dict[str, list[float | None]],
    title: str,
    y_label: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, y_values in series.items():
        plt.plot(x_values, y_values, marker="o", label=name)
    plt.xticks(x_values)
    plt.xlabel("round_index")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_histogram(
    *,
    out_path: Path,
    values: list[float],
    title: str,
    x_label: str,
) -> bool:
    if not values:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = max(5, min(30, int(math.sqrt(len(values))) + 1))
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    mean_v = statistics.fmean(values)
    median_v = statistics.median(values)
    plt.axvline(mean_v, color="red", linestyle="--", linewidth=1.6, label=f"mean={mean_v:.4f}")
    plt.axvline(median_v, color="green", linestyle="-.", linewidth=1.6, label=f"median={median_v:.4f}")
    plt.xlabel(x_label)
    plt.ylabel("feature count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def render_token_usage_plot(*, token_usage_overall: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    metric_to_row = {str(row.get("metric")): row for row in token_usage_overall}
    labels = TOKEN_USAGE_FIELDS
    sums: list[float] = []
    means: list[float | None] = []
    for metric in labels:
        row = metric_to_row.get(metric, {})
        sum_v = to_float(row.get("sum"))
        sums.append(sum_v if sum_v is not None else 0.0)
        means.append(to_float(row.get("mean")))

    if not any(v > 0 for v in sums):
        return []

    out_path = figures_dir / "token_usage_overall.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pretty_label = {
        "input_tokens": "input tokens",
        "output_tokens": "output tokens",
        "total_tokens": "total tokens",
    }
    x_labels = [pretty_label[name] for name in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_labels, sums)
    plt.ylabel("sum across features")
    plt.title("Token Usage Summary (All Features)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        mean_v = means[i]
        if mean_v is None:
            label = f"sum={height:.0f}"
        else:
            label = f"sum={height:.0f}\nmean={mean_v:.2f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return [str(out_path)]


def render_relative_quality_distribution_plots(
    *, feature_rows: list[dict[str, Any]], figures_dir: Path
) -> list[str]:
    generated: list[str] = []
    metric_to_name = {
        "relative_quality_score_non_zero_rate": "Relative Quality (Non-Zero Rate)",
        "relative_quality_score_boundary_non_activation_rate": "Relative Quality (Boundary Non-Activation Rate)",
        "relative_quality_combined_input_score": "Relative Quality (Combined Input Score)",
    }
    metric_to_filename = {
        "relative_quality_score_non_zero_rate": "relative_quality_non_zero_rate_distribution.png",
        "relative_quality_score_boundary_non_activation_rate": "relative_quality_boundary_non_activation_rate_distribution.png",
        "relative_quality_combined_input_score": "relative_quality_combined_input_score_distribution.png",
    }
    for metric in RELATIVE_QUALITY_FIELDS:
        values = [to_float(row.get(metric)) for row in feature_rows]
        numeric = [v for v in values if v is not None]
        out_path = figures_dir / metric_to_filename[metric]
        if _plot_histogram(
            out_path=out_path,
            values=numeric,
            title=metric_to_name[metric],
            x_label="score",
        ):
            generated.append(str(out_path))
    return generated


def render_round_plots(
    *,
    round_mean_overall: list[dict[str, Any]],
    round_mean_by_layer: list[dict[str, Any]],
    figures_dir: Path,
) -> list[str]:
    generated: list[str] = []

    # Overall: one figure with three metrics.
    round_indices = sorted({int(row["round_index"]) for row in round_mean_overall})
    if round_indices:
        metric_to_points: dict[str, dict[int, float | None]] = {}
        for metric in ROUND_SCORE_FIELDS:
            metric_to_points[metric] = {}
        for row in round_mean_overall:
            metric = str(row["metric"])
            if metric in metric_to_points:
                metric_to_points[metric][int(row["round_index"])] = to_float(row["mean"])

        series: dict[str, list[float | None]] = {}
        for metric in ROUND_SCORE_FIELDS:
            series[metric] = [metric_to_points[metric].get(r) for r in round_indices]

        out_path = figures_dir / "round_trend_overall.png"
        if _plot_round_lines(
            out_path=out_path,
            x_values=round_indices,
            series=series,
            title="Average Round Score Trend (All Features)",
            y_label="mean score",
        ):
            generated.append(str(out_path))

    # Per-layer: one figure per layer with three metrics.
    layer_set = sorted({int(row["layer"]) for row in round_mean_by_layer})
    for layer in layer_set:
        rows = [row for row in round_mean_by_layer if int(row["layer"]) == layer]
        rounds = sorted({int(row["round_index"]) for row in rows})
        if not rounds:
            continue
        metric_to_points: dict[str, dict[int, float | None]] = {}
        for metric in ROUND_SCORE_FIELDS:
            metric_to_points[metric] = {}
        for row in rows:
            metric = str(row["metric"])
            if metric in metric_to_points:
                metric_to_points[metric][int(row["round_index"])] = to_float(row["mean"])
        series: dict[str, list[float | None]] = {}
        for metric in ROUND_SCORE_FIELDS:
            series[metric] = [metric_to_points[metric].get(r) for r in rounds]

        out_path = figures_dir / f"round_trend_layer_{layer}.png"
        if _plot_round_lines(
            out_path=out_path,
            x_values=rounds,
            series=series,
            title=f"Average Round Score Trend (Layer {layer})",
            y_label="mean score",
        ):
            generated.append(str(out_path))

    return generated


def main() -> int:
    args = parse_args()
    ctx = build_context(args)

    manifest = load_json(ctx.manifest_path)
    records_raw = manifest.get("records", [])
    records = [item for item in records_raw if isinstance(item, dict)]

    feature_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for record in records:
        layer = int(record.get("layer_id"))
        feature = int(record.get("feature_id"))

        summary_output_text = str(record.get("summary_output", "")).strip()
        summary_output_path = Path(summary_output_text) if summary_output_text else None
        timestamp = (
            extract_timestamp_from_single_summary(summary_output_path)
            if summary_output_path is not None
            else None
        )
        if not timestamp:
            debug_rows.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "timestamp": "",
                    "reason": "missing_timestamp_from_run_single_summary",
                    "path": summary_output_text,
                }
            )
            continue

        final_eval = final_evaluation_path(layer, feature, timestamp)
        if not final_eval.exists():
            debug_rows.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "timestamp": timestamp,
                    "reason": "missing_final_evaluation_json",
                    "path": str(final_eval),
                }
            )
            continue

        try:
            final_payload = load_json(final_eval)
        except Exception as exc:
            debug_rows.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "timestamp": timestamp,
                    "reason": f"invalid_final_evaluation_json: {exc}",
                    "path": str(final_eval),
                }
            )
            continue

        input_eval = final_payload.get("input_evaluation")
        if not isinstance(input_eval, dict):
            debug_rows.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "timestamp": timestamp,
                    "reason": "missing_input_evaluation_object",
                    "path": str(final_eval),
                }
            )
            continue

        input_tokens, output_tokens, total_tokens = extract_token_usage_for_feature(
            layer=layer,
            feature=feature,
            timestamp=timestamp,
            final_eval_payload=final_payload,
        )

        feature_rows.append(
            {
                "layer": layer,
                "feature": feature,
                "timestamp": timestamp,
                "final_evaluation_path": str(final_eval),
                "relative_quality_score_non_zero_rate": to_float(
                    input_eval.get("relative_quality_score_non_zero_rate")
                ),
                "relative_quality_score_boundary_non_activation_rate": to_float(
                    input_eval.get("relative_quality_score_boundary_non_activation_rate")
                ),
                "relative_quality_combined_input_score": to_float(
                    input_eval.get("relative_quality_combined_input_score")
                ),
                "workflow_score_non_zero_rate": to_float(input_eval.get("workflow_score_non_zero_rate")),
                "workflow_score_boundary_non_activation_rate": to_float(
                    input_eval.get("workflow_score_boundary_non_activation_rate")
                ),
                "workflow_combined_input_score": to_float(input_eval.get("workflow_combined_input_score")),
                "neuronpedia_mean_combined_input_score": to_float(
                    input_eval.get("neuronpedia_mean_combined_input_score")
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )

        # Load all existing round_{i} execution files and take each round's best hypothesis score.
        round_index = 0
        while True:
            execution_path = execution_round_path(layer, feature, timestamp, round_index)
            if not execution_path.exists():
                break
            try:
                execution_payload = load_json(execution_path)
            except Exception:
                debug_rows.append(
                    {
                        "layer": layer,
                        "feature": feature,
                        "timestamp": timestamp,
                        "reason": "invalid_round_execution_json",
                        "path": str(execution_path),
                    }
                )
                round_index += 1
                continue

            non_zero, boundary = extract_best_round_score(execution_payload)
            combined = None
            if non_zero is not None and boundary is not None:
                combined = non_zero + boundary
            round_rows.append(
                {
                    "layer": layer,
                    "feature": feature,
                    "timestamp": timestamp,
                    "round_index": round_index,
                    "score_non_zero_rate": non_zero,
                    "score_boundary_non_activation_rate": boundary,
                    "combined_score": combined,
                    "execution_path": str(execution_path),
                }
            )
            round_index += 1

    if not feature_rows:
        ctx.debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = ctx.debug_dir / "missing_or_invalid_records.csv"
        write_csv(debug_path, debug_rows, ["layer", "feature", "timestamp", "reason", "path"])
        print(f"Manifest: {ctx.manifest_path}")
        print(f"Valid feature records: 0 / {len(records)}")
        print(f"Debug CSV: {debug_path}")
        if args.allow_empty:
            return 0
        return 1

    relative_overall, relative_by_layer = build_relative_quality_stats(feature_rows)
    token_usage_overall = build_token_usage_stats(feature_rows)
    round_mean_overall, round_mean_by_layer = build_round_mean_stats(round_rows)

    # Raw tables
    feature_metrics_path = ctx.tables_dir / "feature_metrics.csv"
    round_scores_path = ctx.tables_dir / "round_scores_per_feature.csv"
    write_csv(
        feature_metrics_path,
        feature_rows,
        [
            "layer",
            "feature",
            "timestamp",
            "relative_quality_score_non_zero_rate",
            "relative_quality_score_boundary_non_activation_rate",
            "relative_quality_combined_input_score",
            "workflow_score_non_zero_rate",
            "workflow_score_boundary_non_activation_rate",
            "workflow_combined_input_score",
            "neuronpedia_mean_combined_input_score",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "final_evaluation_path",
        ],
    )
    write_csv(
        round_scores_path,
        round_rows,
        [
            "layer",
            "feature",
            "timestamp",
            "round_index",
            "score_non_zero_rate",
            "score_boundary_non_activation_rate",
            "combined_score",
            "execution_path",
        ],
    )

    # Aggregated tables
    relative_overall_path = ctx.tables_dir / "relative_quality_overall.csv"
    relative_by_layer_path = ctx.tables_dir / "relative_quality_by_layer.csv"
    token_usage_overall_path = ctx.tables_dir / "token_usage_overall.csv"
    round_mean_overall_path = ctx.tables_dir / "round_mean_overall.csv"
    round_mean_by_layer_path = ctx.tables_dir / "round_mean_by_layer.csv"
    write_csv(
        relative_overall_path,
        relative_overall,
        ["metric", "valid_count", "mean", "median", "std", "min", "max"],
    )
    write_csv(
        relative_by_layer_path,
        relative_by_layer,
        ["layer", "metric", "valid_count", "mean", "median", "std", "min", "max"],
    )
    write_csv(
        token_usage_overall_path,
        token_usage_overall,
        ["metric", "valid_count", "sum", "mean", "median", "std", "min", "max"],
    )
    write_csv(
        round_mean_overall_path,
        round_mean_overall,
        ["round_index", "metric", "n_features", "mean", "median", "std", "min", "max"],
    )
    write_csv(
        round_mean_by_layer_path,
        round_mean_by_layer,
        ["layer", "round_index", "metric", "n_features", "mean", "median", "std", "min", "max"],
    )

    debug_path = ctx.debug_dir / "missing_or_invalid_records.csv"
    write_csv(debug_path, debug_rows, ["layer", "feature", "timestamp", "reason", "path"])

    figure_paths: list[str] = []
    figure_paths.extend(render_round_plots(
        round_mean_overall=round_mean_overall,
        round_mean_by_layer=round_mean_by_layer,
        figures_dir=ctx.figures_dir,
    ))
    figure_paths.extend(
        render_relative_quality_distribution_plots(
            feature_rows=feature_rows,
            figures_dir=ctx.figures_dir,
        )
    )
    figure_paths.extend(
        render_token_usage_plot(
            token_usage_overall=token_usage_overall,
            figures_dir=ctx.figures_dir,
        )
    )

    analysis_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(ctx.batch_dir),
        "batch_manifest": str(ctx.manifest_path),
        "record_count": len(records),
        "valid_feature_record_count": len(feature_rows),
        "valid_round_record_count": len(round_rows),
        "debug_record_count": len(debug_rows),
        "outputs": {
            "feature_metrics_csv": str(feature_metrics_path),
            "round_scores_per_feature_csv": str(round_scores_path),
            "relative_quality_overall_csv": str(relative_overall_path),
            "relative_quality_by_layer_csv": str(relative_by_layer_path),
            "token_usage_overall_csv": str(token_usage_overall_path),
            "round_mean_overall_csv": str(round_mean_overall_path),
            "round_mean_by_layer_csv": str(round_mean_by_layer_path),
            "debug_csv": str(debug_path),
            "figure_paths": figure_paths,
        },
    }
    analysis_manifest_path = ctx.analysis_dir / "analysis_manifest.json"
    analysis_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_manifest_path.write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Batch manifest: {ctx.manifest_path}")
    print(f"Analysis dir: {ctx.analysis_dir}")
    print(f"Valid feature records: {len(feature_rows)} / {len(records)}")
    print(f"Valid round records: {len(round_rows)}")
    print(f"Debug records: {len(debug_rows)}")
    print(f"Feature metrics CSV: {feature_metrics_path}")
    print(f"Round mean overall CSV: {round_mean_overall_path}")
    print(f"Round mean by layer CSV: {round_mean_by_layer_path}")
    if figure_paths:
        print("Round trend figures:")
        for p in figure_paths:
            print(f"  - {p}")
    else:
        print("Round trend figures: not generated (matplotlib unavailable or no round data)")
    print(f"Analysis manifest: {analysis_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
