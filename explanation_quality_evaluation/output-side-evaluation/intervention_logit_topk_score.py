from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
from openai import OpenAI

# Allow importing project modules when this script is launched from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_design import OUTPUT_SIDE_PLACEHOLDER
from function import DEFAULT_CANONICAL_MAP_PATH, build_default_sae_path, call_llm
from support_info.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE

if TYPE_CHECKING:
    from model_with_sae import ModelWithSAEModule

SAE_RELEASE_BY_NAME: Dict[str, str] = {
    "gemmascope-res": "gemma-scope-2b-pt-res",
}

TOKEN_JUDGE_SYSTEM_PROMPT = """You evaluate whether token logits should move under a feature hypothesis.

You will receive:
1) A hypothesis describing what one model feature represents.
2) Candidate tokens.

For each token, classify the expected effect:
- increase: token logit should be clearly increased by this feature
- decrease: token logit should be clearly decreased by this feature
- no_change: no clear directional expectation

Return strict JSON only (no markdown, no extra text):
{
  "judgments": [
    {
      "token_id": 0,
      "token": "example",
      "expected_effect": "increase|decrease|no_change",
      "reason": "short reason"
    }
  ]
}
"""


@dataclass
class TokenDeltaJudgment:
    rank: int
    token_id: int
    token: str
    clean_logit: float
    steered_logit: float
    delta_logit: float
    llm_expected_effect: str
    llm_reason: str
    llm_is_correct: bool


def _resolve_sae(
    *,
    sae_name: str,
    sae_release: Optional[str],
    layer_id: str,
    width: str,
    average_l0: Optional[str],
    canonical_map_path: Path,
) -> Tuple[str, str, str]:
    release = (sae_release or SAE_RELEASE_BY_NAME.get(sae_name) or sae_name).strip()
    sae_uri, resolved_average_l0 = build_default_sae_path(
        layer_id=layer_id,
        width=width,
        release=release,
        average_l0=average_l0,
        canonical_map_path=canonical_map_path,
    )
    return sae_uri, release, resolved_average_l0


def _resolve_api_key(api_key: Optional[str], api_key_file: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    if api_key_file:
        key_path = Path(api_key_file)
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
    return None


def _hash_prompts(prompts: Sequence[str]) -> str:
    payload = json.dumps([str(x) for x in prompts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_clamp_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": []}
    if not isinstance(payload, dict):
        return {"version": 1, "entries": []}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        payload["entries"] = []
    payload.setdefault("version", 1)
    return payload


def _find_cached_clamp(
    *,
    cache_payload: Dict[str, Any],
    target_kl: float,
    prompt_hash: str,
) -> Optional[Dict[str, Any]]:
    entries = cache_payload.get("entries", [])
    if not isinstance(entries, list):
        return None
    for item in entries:
        if not isinstance(item, dict):
            continue
        try:
            item_target_kl = float(item.get("target_kl"))
        except Exception:
            continue
        if abs(item_target_kl - float(target_kl)) > 1e-9:
            continue
        if str(item.get("prompt_hash", "")) != prompt_hash:
            continue
        if "clamp_value" not in item:
            continue
        return item
    return None


def _save_clamp_cache(path: Path, cache_payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None

    candidates: List[str] = [raw]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    brace_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    bracket_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if bracket_match:
        candidates.append(bracket_match.group(0))

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
        if isinstance(loaded, list):
            return {"judgments": loaded}
    return None


def _normalize_effect(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"increase", "up", "higher", "promote", "positive"}:
        return "increase"
    if text in {"decrease", "down", "lower", "suppress", "negative"}:
        return "decrease"
    if text in {"no_change", "no-change", "neutral", "none", "unchanged"}:
        return "no_change"
    if "increase" in text or "up" in text or "promot" in text:
        return "increase"
    if "decrease" in text or "down" in text or "suppress" in text:
        return "decrease"
    return "no_change"


def _judgment_map_from_payload(payload: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    result: Dict[int, Dict[str, str]] = {}
    if not payload:
        return result
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        return result

    for item in judgments:
        if not isinstance(item, dict):
            continue
        try:
            token_id = int(item.get("token_id"))
        except Exception:
            continue
        effect = _normalize_effect(item.get("expected_effect"))
        reason = str(item.get("reason", "")).strip()
        result[token_id] = {
            "expected_effect": effect,
            "reason": reason,
        }
    return result


def _safe_prompt_list(prompts: Sequence[str]) -> List[str]:
    cleaned = [str(p).strip() for p in prompts if str(p).strip()]
    if not cleaned:
        raise ValueError("prompts must contain at least one non-empty string.")
    return cleaned


def _select_topk_token_ids(
    *,
    delta: torch.Tensor,
    top_k: int,
    tokenizer: Any,
    skip_special_tokens: bool,
) -> Tuple[List[int], List[int]]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    increase_scores = delta.clone()
    decrease_scores = delta.clone()

    if skip_special_tokens and tokenizer is not None and hasattr(tokenizer, "all_special_ids"):
        for special_id in tokenizer.all_special_ids:
            if 0 <= int(special_id) < int(delta.shape[0]):
                increase_scores[int(special_id)] = float("-inf")
                decrease_scores[int(special_id)] = float("inf")

    vocab_size = int(delta.shape[0])
    k = min(int(top_k), vocab_size)

    _, top_inc_ids = torch.topk(increase_scores, k=k)
    _, top_dec_ids = torch.topk(decrease_scores, k=k, largest=False)
    return [int(x) for x in top_inc_ids.tolist()], [int(x) for x in top_dec_ids.tolist()]


def _build_token_rows(
    *,
    token_ids: Sequence[int],
    clean_mean_logits: torch.Tensor,
    steered_mean_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    tokenizer: Any,
    judgments: Dict[int, Dict[str, str]],
    expected_effect: str,
) -> List[TokenDeltaJudgment]:
    if tokenizer is not None:
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    else:
        tokens = [str(x) for x in token_ids]

    rows: List[TokenDeltaJudgment] = []
    for idx, token_id in enumerate(token_ids, start=1):
        judgment = judgments.get(int(token_id), {"expected_effect": "no_change", "reason": ""})
        llm_effect = _normalize_effect(judgment.get("expected_effect"))
        llm_reason = str(judgment.get("reason", "")).strip()
        rows.append(
            TokenDeltaJudgment(
                rank=idx,
                token_id=int(token_id),
                token=str(tokens[idx - 1]),
                clean_logit=float(clean_mean_logits[token_id].item()),
                steered_logit=float(steered_mean_logits[token_id].item()),
                delta_logit=float(delta_logits[token_id].item()),
                llm_expected_effect=llm_effect,
                llm_reason=llm_reason,
                llm_is_correct=(llm_effect == expected_effect),
            )
        )
    return rows


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    runs = payload.get("runs", [])
    positive_ratios = [
        float(run.get("scores", {}).get("positive_topk_increase_ratio", 0.0))
        for run in runs
        if isinstance(run, dict)
    ]
    negative_ratios = [
        float(run.get("scores", {}).get("negative_topk_decrease_ratio", 0.0))
        for run in runs
        if isinstance(run, dict)
    ]

    lines: List[str] = []
    lines.append("# Output-side Logit Top-K Evaluation")
    lines.append("")
    lines.append("## Metadata")
    for key in [
        "generated_at",
        "sae_name",
        "sae_release",
        "sae_uri",
        "layer_id",
        "width",
        "feature_id",
        "target_kls",
        "top_k",
    ]:
        lines.append(f"- {key}: {metadata.get(key)}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("```text")
    lines.append(payload.get("explanation", ""))
    lines.append("```")
    lines.append("")
    lines.append("## Scores")
    lines.append(
        f"- mean_positive_topk_increase_ratio: "
        f"{(sum(positive_ratios) / len(positive_ratios)) if positive_ratios else None}"
    )
    lines.append(
        f"- mean_negative_topk_decrease_ratio: "
        f"{(sum(negative_ratios) / len(negative_ratios)) if negative_ratios else None}"
    )
    lines.append(f"- run_count: {len(runs)}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("| target_kl | clamp_value | actual_kl | cache_hit | positive_ratio | negative_ratio |")
    lines.append("| ---: | ---: | ---: | --- | ---: | ---: |")
    for run in payload.get("runs", []):
        scores = run.get("scores", {})
        lines.append(
            f"| {run.get('target_kl')} | {run.get('clamp_value')} | {run.get('actual_kl')} | {run.get('clamp_cache_hit')} "
            f"| {scores.get('positive_topk_increase_ratio')} | {scores.get('negative_topk_decrease_ratio')} |"
        )
    lines.append("")

    for run in payload.get("runs", []):
        lines.append(f"## KL {run.get('target_kl'):+g}")
        lines.append("")
        scores = run.get("scores", {})
        lines.append(f"- positive_topk_increase_ratio: {scores.get('positive_topk_increase_ratio')}")
        lines.append(f"- negative_topk_decrease_ratio: {scores.get('negative_topk_decrease_ratio')}")
        lines.append(f"- clamp_value: {run.get('clamp_value')}")
        lines.append(f"- actual_kl: {run.get('actual_kl')}")
        lines.append(f"- clamp_cache_hit: {run.get('clamp_cache_hit')}")
        lines.append("")

        lines.append("### Positive Top-K (Largest Delta Increase)")
        lines.append("| rank | token_id | token | clean_logit | steered_logit | delta_logit | llm_expected_effect | correct |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |")
        for row in run.get("positive_topk_tokens", []):
            lines.append(
                f"| {row.get('rank')} | {row.get('token_id')} | {str(row.get('token', '')).replace('|', '\\|')} "
                f"| {row.get('clean_logit'):.6f} | {row.get('steered_logit'):.6f} | {row.get('delta_logit'):.6f} "
                f"| {row.get('llm_expected_effect')} | {row.get('llm_is_correct')} |"
            )
        lines.append("")

        lines.append("### Negative Top-K (Largest Delta Decrease)")
        lines.append("| rank | token_id | token | clean_logit | steered_logit | delta_logit | llm_expected_effect | correct |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: | --- | --- |")
        for row in run.get("negative_topk_tokens", []):
            lines.append(
                f"| {row.get('rank')} | {row.get('token_id')} | {str(row.get('token', '')).replace('|', '\\|')} "
                f"| {row.get('clean_logit'):.6f} | {row.get('steered_logit'):.6f} | {row.get('delta_logit'):.6f} "
                f"| {row.get('llm_expected_effect')} | {row.get('llm_is_correct')} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score output-side explanation by top-k unembedding-logit changes after KL-guided intervention. "
            "Outputs are saved under outputs/{sae}/layer-{layer}/feature-{feature}."
        )
    )
    parser.add_argument("--layer-id", required=True, help="SAE layer id")
    parser.add_argument("--width", default="16k", help="SAE width, e.g. 16k")
    parser.add_argument("--feature-id", required=True, type=int, help="Feature id to intervene")
    parser.add_argument("--sae-name", default="gemmascope-res", help="SAE family name for output path")
    parser.add_argument("--sae-release", default=None, help="Optional explicit SAE release override")
    parser.add_argument("--sae-average-l0", default=None, help="Optional average_l0 override")
    parser.add_argument(
        "--sae-canonical-map",
        default=str(PROJECT_ROOT / DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt for resolving canonical average_l0",
    )
    parser.add_argument("--model-checkpoint-path", default="google/gemma-2-2b", help="Base model checkpoint path")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--target-kl", type=float, nargs="+", required=True, help="One or more signed target KL divergence values")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k size for positive/negative logit delta tokens")
    parser.add_argument("--kl-tolerance", type=float, default=0.1)
    parser.add_argument("--kl-max-steps", type=int, default=12)
    parser.add_argument("--force-refresh-kl-cache", action="store_true")

    parser.add_argument("--explanation", type=str, default=None, help="Hypothesis text used for token judgment")
    parser.add_argument("--explanation-file", type=str, default=None, help="Path to file containing hypothesis text")
    parser.add_argument("--prompts", nargs="*", default=list(OUTPUT_SIDE_PLACEHOLDER), help="Prompt list in English")
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Include special tokens in top-k selection (default: excluded)",
    )

    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-key-file", type=str, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--openai-model", type=str, default="zai-org/glm-4.7")
    parser.add_argument("--openai-base-url", type=str, default="https://api.ppio.com/openai")
    parser.add_argument("--judge-max-tokens", type=int, default=10000, help="Max tokens for each LLM judge call")

    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"),
        help="Root output directory",
    )
    parser.add_argument("--json-filename", default="intervention_logit_topk_score.json")
    parser.add_argument("--md-filename", default="intervention_logit_topk_score.md")
    parser.add_argument(
        "--prefer-existing",
        dest="prefer_existing",
        action="store_true",
        help="Prefer existing intervention_logit_topk_score.json under output directory.",
    )
    parser.add_argument(
        "--no-prefer-existing",
        dest="prefer_existing",
        action="store_false",
        help="Do not load existing score json before recomputing.",
    )
    parser.add_argument(
        "--force-refresh-score",
        action="store_true",
        help="Force recomputing score even if existing score json is present.",
    )
    parser.set_defaults(prefer_existing=True)
    return parser


def _resolve_explanation(args: argparse.Namespace) -> str:
    if args.explanation and str(args.explanation).strip():
        return str(args.explanation).strip()
    if args.explanation_file:
        text = Path(str(args.explanation_file)).read_text(encoding="utf-8").strip()
        if text:
            return text
    raise ValueError("Missing explanation: provide --explanation or --explanation-file.")


def _resolve_target_dir(*, output_root: str, sae_name: str, layer_id: str, feature_id: int) -> Path:
    target_dir = Path(output_root) / sae_name / f"layer-{layer_id}" / f"feature-{feature_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _try_load_existing_result(
    *,
    target_dir: Path,
    json_filename: str,
) -> Optional[Dict[str, Any]]:
    json_path = target_dir / json_filename
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def main() -> None:
    args = _build_parser().parse_args()
    layer_id = str(args.layer_id)
    width = str(args.width)
    feature_id = int(args.feature_id)
    target_kls = [float(x) for x in args.target_kl]
    top_k = int(args.top_k)
    prompts = _safe_prompt_list(args.prompts)
    target_dir = _resolve_target_dir(
        output_root=str(args.output_root),
        sae_name=str(args.sae_name),
        layer_id=layer_id,
        feature_id=feature_id,
    )

    if bool(args.prefer_existing) and not bool(args.force_refresh_score):
        existing_payload = _try_load_existing_result(
            target_dir=target_dir,
            json_filename=str(args.json_filename),
        )
        if existing_payload is not None:
            md_path = target_dir / str(args.md_filename)
            _write_markdown(md_path, existing_payload)
            print(
                json.dumps(
                    {
                        "output_dir": str(target_dir),
                        "json_file": str(target_dir / str(args.json_filename)),
                        "md_file": str(md_path),
                        "loaded_from_cache": True,
                        "runs": [
                            {
                                "target_kl": run.get("target_kl"),
                                "positive_topk_increase_ratio": run.get("scores", {}).get("positive_topk_increase_ratio"),
                                "negative_topk_decrease_ratio": run.get("scores", {}).get("negative_topk_decrease_ratio"),
                            }
                            for run in existing_payload.get("runs", [])
                            if isinstance(run, dict)
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

    explanation = _resolve_explanation(args)

    sae_uri, sae_release, resolved_average_l0 = _resolve_sae(
        sae_name=str(args.sae_name),
        sae_release=args.sae_release,
        layer_id=layer_id,
        width=width,
        average_l0=args.sae_average_l0,
        canonical_map_path=Path(args.sae_canonical_map),
    )

    cache_path = target_dir / "kl_clamp_cache.json"

    from model_with_sae import ModelWithSAEModule

    module = ModelWithSAEModule(
        llm_name=str(args.model_checkpoint_path),
        sae_path=sae_uri,
        sae_layer=int(layer_id),
        feature_index=feature_id,
        device=str(args.device),
    )
    if module.model is None or module.tokenizer is None:
        raise RuntimeError("Model/tokenizer failed to load.")
    if not module.use_hooked_transformer:
        raise RuntimeError("This score requires HookedSAETransformer mode (sae-lens URI).")
    if "__sae_lens_obj__" not in module.sae:
        raise RuntimeError("SAE object not found. Please use sae-lens SAE.")

    prompt_tokens = module.model.to_tokens(prompts)
    if not isinstance(prompt_tokens, torch.Tensor) or prompt_tokens.ndim != 2:
        raise RuntimeError("Unexpected prompt token shape from model.to_tokens.")

    prompt_hash = _hash_prompts(prompts)
    cache_payload = _load_clamp_cache(cache_path)
    api_key = _resolve_api_key(api_key=args.api_key, api_key_file=args.api_key_file)
    if not api_key:
        raise ValueError("Missing API key: set --api-key or --api-key-file.")
    judge_client = OpenAI(
        base_url=str(args.openai_base_url),
        api_key=api_key,
    )
    sae_obj = module.sae["__sae_lens_obj__"]
    runs: List[Dict[str, Any]] = []
    clean_logits = module.run_logits(prompt_tokens)
    clean_mean_logits = clean_logits.mean(dim=(0, 1)).detach().float().cpu()
    # Note all tokens or next token?

    for target_kl in target_kls:
        cache_hit_entry: Optional[Dict[str, Any]] = None
        if not bool(args.force_refresh_kl_cache):
            cache_hit_entry = _find_cached_clamp(
                cache_payload=cache_payload,
                target_kl=target_kl,
                prompt_hash=prompt_hash,
            )

        if cache_hit_entry is not None:
            clamp_value = float(cache_hit_entry["clamp_value"])
            actual_kl = float(cache_hit_entry.get("actual_kl", 0.0))
            clamp_cache_hit = True
        else:
            clamp_values, kl_values = module._find_clamp_values_for_kl(
                prompt_tokens,
                feature_id,
                sae_obj,
                target_kl=target_kl,
                tolerance=float(args.kl_tolerance),
                max_steps=int(args.kl_max_steps),
            )
            if not clamp_values:
                raise RuntimeError(f"Failed to find clamp value for target KL={target_kl}.")
            clamp_value = float(clamp_values[0])
            actual_kl = float(kl_values[0]) if kl_values else 0.0
            clamp_cache_hit = False

            cache_payload.setdefault("entries", [])
            cache_payload["entries"].append(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "target_kl": target_kl,
                    "prompt_hash": prompt_hash,
                    "prompts": prompts,
                    "clamp_value": clamp_value,
                    "actual_kl": actual_kl,
                    "kl_tolerance": float(args.kl_tolerance),
                    "kl_max_steps": int(args.kl_max_steps),
                }
            )

        steered_logits = module.run_logits_with_feature_intervention(
            input_ids=prompt_tokens,
            feature_index=feature_id,
            value=float(clamp_value),
            mode="clamp",
        )
        steered_mean_logits = steered_logits.mean(dim=(0, 1)).detach().float().cpu()
        delta_logits = steered_mean_logits - clean_mean_logits

        pos_ids, neg_ids = _select_topk_token_ids(
            delta=delta_logits,
            top_k=top_k,
            tokenizer=module.tokenizer,
            skip_special_tokens=not bool(args.include_special_tokens),
        )

        unique_ordered_ids: List[int] = []
        for token_id in pos_ids + neg_ids:
            if token_id not in unique_ordered_ids:
                unique_ordered_ids.append(token_id)
        candidate_tokens = (
            module.tokenizer.convert_ids_to_tokens(unique_ordered_ids)
            if module.tokenizer is not None
            else [str(x) for x in unique_ordered_ids]
        )
        token_items = [{"token_id": int(i), "token": str(t)} for i, t in zip(unique_ordered_ids, candidate_tokens)]

        judge_user_prompt = (
            "Hypothesis:\n"
            f"{explanation}\n\n"
            f"Target KL: {target_kl}\n"
            "Candidate tokens (id + token text):\n"
            f"{json.dumps(token_items, ensure_ascii=False, indent=2)}\n\n"
            "Classify every token with expected_effect in {increase, decrease, no_change}."
        )
        judge_messages = [
            {"role": "system", "content": TOKEN_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_user_prompt},
        ]
        judge_raw_output, _, judge_debug = call_llm(
            client=judge_client,
            model=str(args.openai_model),
            messages=judge_messages,
            temperature=0.0,
            max_tokens=int(args.judge_max_tokens),
            stream=False,
            response_format_text=True,
            return_debug=True,
        )
        judge_payload = _extract_json_payload(judge_raw_output)
        judgment_map = _judgment_map_from_payload(judge_payload)

        positive_rows = _build_token_rows(
            token_ids=pos_ids,
            clean_mean_logits=clean_mean_logits,
            steered_mean_logits=steered_mean_logits,
            delta_logits=delta_logits,
            tokenizer=module.tokenizer,
            judgments=judgment_map,
            expected_effect="increase",
        )
        negative_rows = _build_token_rows(
            token_ids=neg_ids,
            clean_mean_logits=clean_mean_logits,
            steered_mean_logits=steered_mean_logits,
            delta_logits=delta_logits,
            tokenizer=module.tokenizer,
            judgments=judgment_map,
            expected_effect="decrease",
        )

        pos_total = max(1, len(positive_rows))
        neg_total = max(1, len(negative_rows))
        pos_correct = sum(1 for row in positive_rows if row.llm_is_correct)
        neg_correct = sum(1 for row in negative_rows if row.llm_is_correct)

        runs.append(
            {
                "target_kl": target_kl,
                "clamp_value": clamp_value,
                "actual_kl": actual_kl,
                "clamp_cache_hit": clamp_cache_hit,
                "scores": {
                    "positive_topk_increase_ratio": pos_correct / pos_total,
                    "negative_topk_decrease_ratio": neg_correct / neg_total,
                    "positive_correct_count": pos_correct,
                    "positive_total": len(positive_rows),
                    "negative_correct_count": neg_correct,
                    "negative_total": len(negative_rows),
                },
                "positive_topk_tokens": [row.__dict__ for row in positive_rows],
                "negative_topk_tokens": [row.__dict__ for row in negative_rows],
                "llm_judge": {
                    "messages": judge_messages,
                    "raw_output": judge_raw_output,
                    "parsed_output": judge_payload,
                    "debug": judge_debug,
                },
            }
        )

    _save_clamp_cache(cache_path, cache_payload)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": now,
            "sae_name": str(args.sae_name),
            "sae_release": sae_release,
            "sae_uri": sae_uri,
            "resolved_average_l0": resolved_average_l0,
            "layer_id": layer_id,
            "width": width,
            "feature_id": feature_id,
            "target_kls": target_kls,
            "top_k": top_k,
            "prompts": prompts,
            "prompt_hash": prompt_hash,
        },
        "explanation": explanation,
        "runs": runs,
    }

    json_path = target_dir / str(args.json_filename)
    md_path = target_dir / str(args.md_filename)
    json_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, result_payload)

    print(
        json.dumps(
            {
                "output_dir": str(target_dir),
                "json_file": str(json_path),
                "md_file": str(md_path),
                "runs": [
                    {
                        "target_kl": run.get("target_kl"),
                        "positive_topk_increase_ratio": run.get("scores", {}).get("positive_topk_increase_ratio"),
                        "negative_topk_decrease_ratio": run.get("scores", {}).get("negative_topk_decrease_ratio"),
                        "clamp_value": run.get("clamp_value"),
                        "actual_kl": run.get("actual_kl"),
                        "clamp_cache_hit": run.get("clamp_cache_hit"),
                    }
                    for run in runs
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
