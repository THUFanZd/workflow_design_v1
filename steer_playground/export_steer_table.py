from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "steer_playground" / "output"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export steer batch_results.json to CSV. "
            "Default mode scans output_root for batch_results.json and writes CSV into each job directory "
            "when missing."
        )
    )
    parser.add_argument("--results-file", type=str, nargs="*", default=None)
    parser.add_argument("--target-pairs", type=str, nargs="*", default=None)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--csv-name", type=str, default="batch_results_table.csv")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate CSV even if it already exists in the job directory.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=0,
        help="Max chars for output text columns. 0 means no truncation.",
    )
    return parser.parse_args()


def _collect_result_files(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []

    if args.results_file:
        for item in args.results_file:
            p = Path(item)
            if not p.exists():
                raise FileNotFoundError(f"results file not found: {p}")
            paths.append(p)

    if args.target_pairs:
        for item in args.target_pairs:
            layer, feature = _parse_pair_text(item)
            p = Path(args.output_root) / str(layer) / str(feature) / "batch_results.json"
            if not p.exists():
                raise FileNotFoundError(f"results file not found: {p}")
            paths.append(p)

    # Default mode: discover all batch_results.json files and only process those
    # whose per-job CSV does not exist yet.
    if not paths:
        output_root = Path(args.output_root)
        if not output_root.exists():
            raise FileNotFoundError(f"output root not found: {output_root}")
        for p in output_root.rglob("batch_results.json"):
            csv_path = p.parent / str(args.csv_name)
            if args.overwrite_existing or (not csv_path.exists()):
                paths.append(p)
        if not paths:
            raise ValueError(
                "No eligible batch_results.json found. "
                "Either all directories already contain CSV, or no results exist."
            )

    # Deduplicate in input order.
    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _shorten(text: str, max_chars: int) -> str:
    s = str(text).replace("\n", "\\n").strip()
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def _fmt_number(v: Any, digits: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return f"{v:.{digits}g}"
    return str(v)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return payload


def _rows_from_result(path: Path, payload: Dict[str, Any], preview_chars: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    neuronpedia = payload.get("neuronpedia", {}) if isinstance(payload.get("neuronpedia"), dict) else {}
    layer_id = metadata.get("layer_id", "")
    feature_id = metadata.get("feature_id", "")
    np_source = str(neuronpedia.get("source", ""))
    feature_explanation = str(neuronpedia.get("feature_explanation", ""))
    np_max_token = str(neuronpedia.get("max_activation_token", ""))
    np_max_value = _fmt_number(neuronpedia.get("max_activation_value"))

    prompt_results = payload.get("prompt_results", [])
    if not isinstance(prompt_results, list):
        return rows

    for prompt_item in prompt_results:
        if not isinstance(prompt_item, dict):
            continue
        prompt_id = str(prompt_item.get("prompt_id", ""))
        prompt_text = str(prompt_item.get("prompt_text", ""))
        clean_output = prompt_item.get("clean_output", {})
        clean_text = ""
        if isinstance(clean_output, dict):
            clean_text = str(clean_output.get("completion_text", ""))

        interventions = prompt_item.get("interventions", [])
        if not isinstance(interventions, list):
            continue
        for iv in interventions:
            if not isinstance(iv, dict):
                continue
            steered_output = iv.get("steered_output", {})
            steered_text = ""
            if isinstance(steered_output, dict):
                steered_text = str(steered_output.get("completion_text", ""))

            rows.append(
                {
                    "layer_id": str(layer_id),
                    "feature_id": str(feature_id),
                    "feature_explanation": feature_explanation,
                    "neuronpedia_source": np_source,
                    "max_activation_token": np_max_token,
                    "max_activation_value": np_max_value,
                    "prompt_id": prompt_id,
                    "prompt_text": _shorten(prompt_text, preview_chars),
                    "method": str(iv.get("intervention_method", "")),
                    "scale": _fmt_number(iv.get("scale")),
                    "steer_value": _fmt_number(iv.get("steer_value")),
                    "achieved_kl": _fmt_number(iv.get("achieved_kl")),
                    "clean_output": _shorten(clean_text, preview_chars),
                    # Keep steered output as the last column for readability.
                    "steered_output": _shorten(steered_text, preview_chars),
                }
            )

    return rows


def _write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_id",
        "feature_id",
        "feature_explanation",
        "neuronpedia_source",
        "max_activation_token",
        "max_activation_value",
        "prompt_id",
        "prompt_text",
        "method",
        "scale",
        "steer_value",
        "achieved_kl",
        "clean_output",
        "steered_output",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    result_files = _collect_result_files(args)

    generated: List[Dict[str, Any]] = []
    for p in result_files:
        payload = _load_json(p)
        rows = _rows_from_result(p, payload, preview_chars=int(args.preview_chars))
        if not rows:
            continue
        csv_path = p.parent / str(args.csv_name)
        _write_csv(rows, csv_path)
        generated.append(
            {
                "results_file": str(p),
                "csv": str(csv_path),
                "rows": len(rows),
            }
        )

    if not generated:
        raise ValueError("No intervention rows found in eligible batch_results files.")

    summary = {
        "generated_count": len(generated),
        "generated": generated,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
