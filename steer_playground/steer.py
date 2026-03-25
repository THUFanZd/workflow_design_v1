from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_with_sae import ModelWithSAEModule
from function import DEFAULT_CANONICAL_MAP_PATH, extract_average_l0_from_canonical_map


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Steer a specified SAE feature and compare clean vs steered generation. "
            "Provide either --steer-value or --target-kl."
        )
    )
    # "sae-lens://release=gemma-scope-2b-pt-res;sae_id=layer_6/width_16k/average_l0_70"
    parser.add_argument("--layer-id", required=True, type=int, help="Target SAE layer id.")
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2b-pt-res",
        help="SAE release used to build sae-lens URI.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default="16k",
        help="SAE width used to build sae-lens URI (e.g., 16k).",
    )
    parser.add_argument("--feature-id", required=True, type=int, help="Target SAE feature id.")
    parser.add_argument(
        "--steer-value",
        type=float,
        default=None,
        help="Direct intervention strength (add mode).",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Target KL divergence for automatic strength search. Signed value controls direction.",
    )
    parser.add_argument("--max-token", required=True, type=int, help="Max new tokens to generate.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me a short story about a cat learning to cook.",
        help="Prompt used for clean/steered generation and KL calibration.",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or local model path for model_with_sae.py",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run model on.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 means greedy decoding.",
    )
    parser.add_argument(
        "--kl-tolerance",
        type=float,
        default=0.01,
        help="Tolerance for KL-guided strength search.",
    )
    parser.add_argument(
        "--kl-max-steps",
        type=int,
        default=16,
        help="Max bisection steps for KL-guided strength search.",
    )
    parser.add_argument(
        "--kl-max-abs-value",
        type=float,
        default=1024.0,
        help="Max absolute steering value allowed during KL search.",
    )
    parser.add_argument(
        "--canonical-map-path",
        type=str,
        default=str(DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt used for average_l0 consistency check.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    has_steer = args.steer_value is not None
    has_kl = args.target_kl is not None
    if has_steer == has_kl:
        raise ValueError("Provide exactly one of --steer-value or --target-kl.")
    if int(args.max_token) <= 0:
        raise ValueError("--max-token must be > 0.")
    if int(args.layer_id) < 0:
        raise ValueError("--layer-id must be >= 0.")
    if has_kl and float(args.target_kl) == 0.0:
        raise ValueError("--target-kl cannot be 0.")


def _build_sae_uri(args: argparse.Namespace) -> Tuple[str, str]:
    canonical_map_path = Path(str(args.canonical_map_path))
    resolved_average_l0 = extract_average_l0_from_canonical_map(
        canonical_map_path=canonical_map_path,
        layer_id=str(int(args.layer_id)),
        width=str(args.width),
    )

    if resolved_average_l0 is None:
        raise ValueError(
            "Cannot resolve average_l0 from canonical map for "
            f"layer_{int(args.layer_id)}/width_{str(args.width)}. "
            f"canonical_map_path={canonical_map_path}"
        )

    sae_uri = (
        "sae-lens://"
        f"release={str(args.sae_release)};"
        f"sae_id=layer_{int(args.layer_id)}/width_{str(args.width)}/average_l0_{str(resolved_average_l0)}"
    )
    return sae_uri, str(resolved_average_l0)


def _prepare_prompt_tensors(module: ModelWithSAEModule, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = module.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(module.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=module.device)
    else:
        attention_mask = attention_mask.to(module.device)
    return input_ids, attention_mask


@torch.no_grad()
def _mean_logits_kl(clean_logits: torch.Tensor, steered_logits: torch.Tensor, attention_mask: torch.Tensor) -> float:
    eps = 1e-10
    clean = clean_logits.float()
    steered = steered_logits.float()
    p = torch.softmax(clean, dim=-1).clamp_min(eps)
    q = torch.softmax(steered, dim=-1).clamp_min(eps)
    token_kl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    valid = attention_mask.bool()
    if valid.any():
        return float(token_kl[valid].mean().item())
    return float(token_kl.mean().item())


@torch.no_grad()
def _compute_kl_for_value(
    module: ModelWithSAEModule,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    feature_id: int,
    steer_value: float,
) -> float:
    clean_logits = module.run_logits(input_ids=input_ids, attention_mask=attention_mask)
    steered_logits = module.run_logits_with_feature_intervention(
        input_ids=input_ids,
        feature_index=feature_id,
        value=float(steer_value),
        mode="add",
        attention_mask=attention_mask,
    )
    return _mean_logits_kl(clean_logits, steered_logits, attention_mask)


@torch.no_grad()
def _find_steer_value_for_target_kl(
    module: ModelWithSAEModule,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    feature_id: int,
    target_kl: float,
    tolerance: float,
    max_steps: int,
    max_abs_value: float,
) -> Tuple[float, float]:
    direction = 1.0 if float(target_kl) > 0 else -1.0
    target = abs(float(target_kl))

    low_mag = 0.0
    high_mag = 1.0
    high_kl = _compute_kl_for_value(
        module,
        input_ids=input_ids,
        attention_mask=attention_mask,
        feature_id=feature_id,
        steer_value=direction * high_mag,
    )

    while high_kl < target and high_mag < max_abs_value:
        low_mag = high_mag
        high_mag = min(high_mag * 2.0, float(max_abs_value))
        high_kl = _compute_kl_for_value(
            module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_id=feature_id,
            steer_value=direction * high_mag,
        )
        if high_mag >= max_abs_value:
            break

    best_mag = high_mag
    best_kl = high_kl

    for _ in range(max(0, int(max_steps))):
        mid_mag = 0.5 * (low_mag + high_mag)
        mid_kl = _compute_kl_for_value(
            module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_id=feature_id,
            steer_value=direction * mid_mag,
        )
        best_mag = mid_mag
        best_kl = mid_kl
        if abs(mid_kl - target) <= float(tolerance):
            break
        if mid_kl < target:
            low_mag = mid_mag
        else:
            high_mag = mid_mag

    return direction * float(best_mag), float(best_kl)


def _sample_next_token(last_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if float(temperature) <= 0.0:
        return torch.argmax(last_logits, dim=-1, keepdim=True)
    probs = torch.softmax(last_logits / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def _generate_text(
    module: ModelWithSAEModule,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    feature_id: Optional[int] = None,
    steer_value: Optional[float] = None,
) -> dict[str, Any]:
    input_ids, attention_mask = _prepare_prompt_tensors(module, prompt)
    prompt_len = int(input_ids.shape[1])
    eos_id = module.tokenizer.eos_token_id

    for _ in range(int(max_new_tokens)):
        if feature_id is None or steer_value is None:
            logits = module.run_logits(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = module.run_logits_with_feature_intervention(
                input_ids=input_ids,
                feature_index=int(feature_id),
                value=float(steer_value),
                mode="add",
                attention_mask=attention_mask,
            )

        next_logits = logits[:, -1, :]
        next_token = _sample_next_token(next_logits, temperature=temperature)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    full_text = module.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    completion_ids = input_ids[0, prompt_len:]
    completion_text = module.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "full_text": full_text,
        "completion_text": completion_text,
        "generated_token_count": int(completion_ids.numel()),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    sae_uri, resolved_average_l0 = _build_sae_uri(args)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    module = ModelWithSAEModule(
        llm_name=str(args.llm_name),
        sae_path=sae_uri,
        sae_layer=int(args.layer_id),
        feature_index=int(args.feature_id),
        device=str(args.device),
        debug=False,
    )

    input_ids, attention_mask = _prepare_prompt_tensors(module, str(args.prompt))

    selected_steer_value: float
    achieved_kl: Optional[float] = None

    if args.steer_value is not None:
        selected_steer_value = float(args.steer_value)
        achieved_kl = _compute_kl_for_value(
            module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_id=int(args.feature_id),
            steer_value=selected_steer_value,
        )
    else:
        selected_steer_value, achieved_kl = _find_steer_value_for_target_kl(
            module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_id=int(args.feature_id),
            target_kl=float(args.target_kl),
            tolerance=float(args.kl_tolerance),
            max_steps=int(args.kl_max_steps),
            max_abs_value=float(args.kl_max_abs_value),
        )

    clean = _generate_text(
        module,
        prompt=str(args.prompt),
        max_new_tokens=int(args.max_token),
        temperature=float(args.temperature),
    )
    steered = _generate_text(
        module,
        prompt=str(args.prompt),
        max_new_tokens=int(args.max_token),
        temperature=float(args.temperature),
        feature_id=int(args.feature_id),
        steer_value=float(selected_steer_value),
    )

    result = {
        "config": {
            "llm_name": str(args.llm_name),
            "sae_release": str(args.sae_release),
            "width": str(args.width),
            "resolved_average_l0": str(resolved_average_l0),
            "sae": sae_uri,
            "layer_id": int(args.layer_id),
            "feature_id": int(args.feature_id),
            "prompt": str(args.prompt),
            "max_token": int(args.max_token),
            "temperature": float(args.temperature),
            "steer_value": float(selected_steer_value),
            "target_kl": (None if args.target_kl is None else float(args.target_kl)),
            "achieved_kl_on_prompt": achieved_kl,
        },
        "clean_output": clean,
        "steered_output": steered,
    }

    print("=" * 80)
    print("Steering Config")
    print("=" * 80)
    for k, v in result["config"].items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("Without Intervention")
    print("=" * 80)
    print(clean["full_text"])

    print("\n" + "=" * 80)
    print("With Intervention")
    print("=" * 80)
    print(steered["full_text"])


if __name__ == "__main__":
    main()
