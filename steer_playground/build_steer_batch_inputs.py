from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuronpedia_feature_api import extract_explanations, fetch_feature_json

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "steer_playground" / "output"
DEFAULT_STEER_OUTPUT_ROOT = PROJECT_ROOT / "steer_playground" / "output"

FIXED_PROMPTS = [
    "I think",
    "We",
    #"The explanation is simple",
    "This is",
    #"<bos>",
]

DEFAULT_SCALE_VALUES = [0.0, 0.5, -0.5, 2.0]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_neuronpedia_source(layer_id: int, width: str) -> str:
    return f"{int(layer_id)}-gemmascope-res-{str(width)}"


def _extract_global_max_activation(payload: Dict[str, Any]) -> Tuple[float, str]:
    activations = payload.get("activations", [])
    if not isinstance(activations, list):
        return 0.0, ""

    best_value = float("-inf")
    best_token = ""
    for item in activations:
        if not isinstance(item, dict):
            continue

        max_value = _to_float(item.get("maxValue"), default=float("-inf"))
        if max_value <= best_value:
            continue

        token = ""
        tokens = item.get("tokens")
        max_idx = item.get("maxValueTokenIndex")
        if isinstance(tokens, list) and isinstance(max_idx, int) and 0 <= max_idx < len(tokens):
            token = str(tokens[max_idx])
        elif "max_token" in item:
            token = str(item.get("max_token"))

        best_value = max_value
        best_token = token

    if best_value == float("-inf"):
        return 0.0, ""
    return float(best_value), str(best_token)


def _build_prompt_specs(feature_explanation: str) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    for idx, text in enumerate(FIXED_PROMPTS):
        prompt_kind = "bos" if text == "<bos>" else "text"
        prompts.append(
            {
                "prompt_id": f"fixed_{idx}",
                "prompt_source": "fixed_template",
                "prompt_kind": prompt_kind,
                "prompt_text": text,
            }
        )

    explanation = str(feature_explanation).strip()
    if explanation:
        prompts.append(
            {
                "prompt_id": "neuronpedia_explanation",
                "prompt_source": "neuronpedia_explanation",
                "prompt_kind": "text",
                "prompt_text": explanation,
            }
        )
    return prompts


def _build_scaled_interventions(max_activation_value: float, scale_values: List[float]) -> List[Dict[str, float]]:
    items: List[Dict[str, float]] = []
    for scale in scale_values:
        scale_float = float(scale)
        items.append(
            {
                "scale": scale_float,
                "steer_value": float(max_activation_value) * scale_float,
            }
        )
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build batch input spec for steer.py-style runs: "
            "fixed prompts + Neuronpedia explanation prompt, and scaled interventions."
        )
    )
    parser.add_argument("--layer-id", type=int, default=None)
    parser.add_argument("--feature-id", type=int, default=None)
    parser.add_argument(
        "--target-pairs",
        type=str,
        nargs="*",
        default=None,
        help="List of layer-feature pairs, e.g. 6,12345 6,12346 or 6:12345.",
    )
    parser.add_argument(
        "--target-pairs-file",
        type=str,
        default=None,
        help="Optional file with one pair per line: layer,feature (or layer:feature).",
    )
    parser.add_argument("--width", type=str, default="16k")
    parser.add_argument("--model-id", type=str, default="gemma-2-2b", help="Neuronpedia model id.")
    parser.add_argument("--llm-name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--scale-values",
        type=float,
        nargs="*",
        default=list(DEFAULT_SCALE_VALUES),
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--steer-output-root", type=str, default=str(DEFAULT_STEER_OUTPUT_ROOT))
    parser.add_argument("--neuronpedia-api-key", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--explanation-limit", type=int, default=1)
    return parser.parse_args()


def _parse_pair_text(text: str) -> Tuple[int, int]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty target tuple.")

    normalized = raw.replace("(", "").replace(")", "").replace(" ", "")
    if ":" in normalized:
        parts = normalized.split(":")
    elif "," in normalized:
        parts = normalized.split(",")
    else:
        raise ValueError(f"Invalid tuple format: {text!r}. Use (layer,feature) or layer:feature.")

    if len(parts) != 2:
        raise ValueError(f"Invalid tuple format: {text!r}.")
    return int(parts[0]), int(parts[1])


def _collect_targets(args: argparse.Namespace) -> List[Tuple[int, int]]:
    targets: List[Tuple[int, int]] = []
    if args.target_pairs:
        for item in args.target_pairs:
            targets.append(_parse_pair_text(item))
    if args.target_pairs_file:
        path = Path(args.target_pairs_file)
        if not path.exists():
            raise FileNotFoundError(f"target pairs file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            targets.append(_parse_pair_text(line))

    if args.layer_id is not None or args.feature_id is not None:
        if args.layer_id is None or args.feature_id is None:
            raise ValueError("When using --layer-id/--feature-id, both must be provided.")
        targets.append((int(args.layer_id), int(args.feature_id)))

    deduped: List[Tuple[int, int]] = []
    seen = set()
    for pair in targets:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)

    if not deduped:
        raise ValueError("Provide --layer-id/--feature-id or --target-pairs / --target-pairs-file.")
    return deduped


def main() -> None:
    args = _parse_args()
    width = str(args.width)
    targets = _collect_targets(args)
    runs: List[Dict[str, Any]] = []

    for layer_id, feature_id in targets:
        output_dir = Path(args.output_root) / str(layer_id) / str(feature_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = Path(args.steer_output_root) / str(layer_id) / str(feature_id)
        raw_dir.mkdir(parents=True, exist_ok=True)

        source = _build_neuronpedia_source(layer_id=layer_id, width=width)
        payload = fetch_feature_json(
            model_id=str(args.model_id),
            source=source,
            feature_id=str(feature_id),
            api_key=args.neuronpedia_api_key or os.getenv("NEURONPEDIA_API_KEY"),
            timeout=int(args.timeout),
        )

        raw_path = raw_dir / "neuronpedia_feature_raw.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        explanations = extract_explanations(payload, limit=max(1, int(args.explanation_limit)))
        feature_explanation = explanations[0] if explanations else ""
        max_activation_value, max_activation_token = _extract_global_max_activation(payload)

        jobs_payload: Dict[str, Any] = {
            "metadata": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "layer_id": layer_id,
                "feature_id": feature_id,
                "width": width,
            },
            "neuronpedia": {
                "model_id": str(args.model_id),
                "source": source,
                "raw_path": str(raw_path),
                "feature_explanation": feature_explanation,
                "max_activation_value": max_activation_value,
                "max_activation_token": max_activation_token,
            },
            "model_config": {
                "llm_name": str(args.llm_name),
                "sae_release": str(args.sae_release),
                "temperature": float(args.temperature),
                "max_new_tokens": min(max(1, int(args.max_new_tokens)), 100),
            },
            "prompt_specs": _build_prompt_specs(feature_explanation=feature_explanation),
            "interventions": {
                "target_kl_values": [],  # Too heavy, leave empty now.
                "scaled_by_max_activation": _build_scaled_interventions(
                    max_activation_value=max_activation_value,
                    scale_values=[float(x) for x in args.scale_values],
                ),
            },
        }

        jobs_path = output_dir / "batch_jobs.json"
        jobs_path.write_text(json.dumps(jobs_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        runs.append(
            {
                "layer_id": layer_id,
                "feature_id": feature_id,
                "jobs_path": str(jobs_path),
                "raw_path": str(raw_path),
                "prompt_count": len(jobs_payload["prompt_specs"]),
                "target_kl_count": len(jobs_payload["interventions"]["target_kl_values"]),
                "scaled_count": len(jobs_payload["interventions"]["scaled_by_max_activation"]),
            }
        )

    print(json.dumps({"runs": runs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
