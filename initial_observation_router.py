from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from function import build_round_dir, normalize_round_id
from neuronpedia_feature_api import fetch_and_parse_feature_observation

ObservationSource = Literal["neuronpedia", "bos_token"]
SUPPORTED_OBSERVATION_SOURCES = ("neuronpedia", "bos_token")


def _load_json_dict_or_raise(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in file: {path}")
    return payload


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bos_top_tokens_path(*, layer_id: str, feature_id: str, bos_token_observation_root: str) -> Path:
    return (
        Path(bos_token_observation_root)
        / f"layer-{layer_id}"
        / f"feature-{feature_id}"
        / "bos_token"
        / "top_tokens.json"
    )


def _convert_bos_top_tokens_to_observation(top_tokens_payload: Dict[str, Any]) -> Dict[str, Any]:
    top_tokens_raw = top_tokens_payload.get("top_tokens", [])
    top_tokens = [item for item in top_tokens_raw if isinstance(item, dict)] if isinstance(top_tokens_raw, list) else []

    activation_examples = []
    compact_top_tokens = []
    for item in top_tokens:
        token_id = _safe_int(item.get("token_id"))
        token_text_raw = item.get("token_text")
        token_text = str(token_text_raw).strip() if token_text_raw is not None else ""
        activation = _safe_float(item.get("activation"))
        rank = _safe_int(item.get("rank"))
        token_label = token_text if token_text else (f"<token_id:{token_id}>" if token_id is not None else "<unknown>")

        activation_token: Dict[str, Any] = {
            "token": token_label,
            "value": activation,
        }
        if token_id is not None:
            activation_token["token_id"] = token_id
        if rank is not None:
            activation_token["rank"] = rank

        activation_examples.append(
            {
                "sentence": f"<bos>{token_label}",
                "activation_tokens": [activation_token],
                "maxValue": activation,
                "max_token": token_label,
            }
        )

        compact_item: Dict[str, Any] = {
            "token": token_label,
            "activation": activation,
        }
        if token_id is not None:
            compact_item["token_id"] = token_id
        if rank is not None:
            compact_item["rank"] = rank
        compact_top_tokens.append(compact_item)

    return {
        "input_side_observation": {
            "source": "bos_token",
            "selected_count": len(activation_examples),
            "activation_examples": activation_examples,
            "bos_token_top_tokens": compact_top_tokens,
            "bos_token_scan_meta": {
                "scoring_position": top_tokens_payload.get("scoring_position"),
                "top_k": _safe_int(top_tokens_payload.get("top_k")),
                "activation_threshold": _safe_float(top_tokens_payload.get("activation_threshold")),
                "evaluated_token_count": _safe_int(top_tokens_payload.get("evaluated_token_count")),
                "activated_token_count": _safe_int(top_tokens_payload.get("activated_token_count")),
                "max_activation_seen": _safe_float(top_tokens_payload.get("max_activation_seen")),
            },
        },
        "output_side_observation": {
            "source": "bos_token",
            "note": "Output-side observation is unavailable for bos_token source.",
            "pos_pairs": [],
            "neg_pairs": [],
        },
    }


def _write_observation_input_artifact(
    *,
    observation: Dict[str, Any],
    layer_id: str,
    feature_id: str,
    timestamp: str,
    round_id: Optional[str],
) -> Path:
    resolved_round_id = normalize_round_id(round_id, round_index=0)
    out_path = (
        build_round_dir(
            layer_id=layer_id,
            feature_id=feature_id,
            timestamp=timestamp,
            round_id=resolved_round_id,
            round_index=0,
        )
        / f"layer{layer_id}-feature{feature_id}-observation-input.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def collect_initial_observation(
    *,
    observation_source: ObservationSource,
    model_id: str,
    layer_id: str,
    feature_id: str,
    timestamp: Optional[str] = None,
    round_id: Optional[str] = "round_0",
    width: str = "16k",
    selection_method: int = 1,
    observation_m: int = 2,
    observation_n: int = 2,
    neuronpedia_api_key: Optional[str] = None,
    neuronpedia_timeout: int = 30,
    bos_token_observation_root: str = "initial_observation",
) -> Dict[str, Any]:
    if observation_source not in SUPPORTED_OBSERVATION_SOURCES:
        raise ValueError(
            f"Unsupported observation source: {observation_source}. "
            f"Supported: {', '.join(SUPPORTED_OBSERVATION_SOURCES)}"
        )

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    if observation_source == "neuronpedia":
        return fetch_and_parse_feature_observation(
            model_id=model_id,
            layer_id=layer_id,
            feature_id=feature_id,
            width=width,
            selection_method=selection_method,
            m=observation_m,
            n=observation_n,
            api_key=neuronpedia_api_key,
            timeout=neuronpedia_timeout,
            timestamp=ts,
            round_id=round_id,
        )

    top_tokens_path = _bos_top_tokens_path(
        layer_id=layer_id,
        feature_id=feature_id,
        bos_token_observation_root=bos_token_observation_root,
    )
    if not top_tokens_path.exists():
        raise FileNotFoundError(
            "Cannot find bos_token top_tokens.json file. "
            f"Expected path: {top_tokens_path}. "
            "Please run input_bos_token_scan.py first."
        )

    top_tokens_payload = _load_json_dict_or_raise(top_tokens_path)
    observation = _convert_bos_top_tokens_to_observation(top_tokens_payload)
    _write_observation_input_artifact(
        observation=observation,
        layer_id=layer_id,
        feature_id=feature_id,
        timestamp=ts,
        round_id=round_id,
    )
    return observation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect initial observation from configured source.")
    parser.add_argument("--observation-source", choices=SUPPORTED_OBSERVATION_SOURCES, default="neuronpedia")
    parser.add_argument("--model-id", default="gemma-2-2b")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--round-id", default="round_0")
    parser.add_argument("--width", default="16k")
    parser.add_argument("--selection-method", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--observation-m", type=int, default=2)
    parser.add_argument("--observation-n", type=int, default=2)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--neuronpedia-timeout", type=int, default=30)
    parser.add_argument("--bos-token-observation-root", default="initial_observation")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    observation = collect_initial_observation(
        observation_source=args.observation_source,
        model_id=str(args.model_id),
        layer_id=str(args.layer_id),
        feature_id=str(args.feature_id),
        timestamp=args.timestamp,
        round_id=args.round_id,
        width=args.width,
        selection_method=int(args.selection_method),
        observation_m=int(args.observation_m),
        observation_n=int(args.observation_n),
        neuronpedia_api_key=args.neuronpedia_api_key,
        neuronpedia_timeout=int(args.neuronpedia_timeout),
        bos_token_observation_root=str(args.bos_token_observation_root),
    )
    print(
        json.dumps(
            {
                "observation_source": args.observation_source,
                "input_selected_count": observation.get("input_side_observation", {}).get("selected_count", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
