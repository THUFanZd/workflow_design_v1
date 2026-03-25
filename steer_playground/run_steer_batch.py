from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from function import DEFAULT_CANONICAL_MAP_PATH, build_default_sae_path

if TYPE_CHECKING:
    from model_with_sae import ModelWithSAEModule

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "steer_playground" / "output"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run steer.py-style batch jobs from batch_jobs.json. "
            "Supports single target or multiple (layer_id, feature_id) pairs."
        )
    )
    parser.add_argument("--jobs-file", type=str, default=None)
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
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--llm-name", type=str, default=None, help="Optional override of jobs.model_config.llm_name")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sae-release", type=str, default=None)
    parser.add_argument("--width", type=str, default=None)
    parser.add_argument(
        "--canonical-map-path",
        type=str,
        default=str(PROJECT_ROOT / DEFAULT_CANONICAL_MAP_PATH),
    )
    parser.add_argument("--kl-tolerance", type=float, default=0.1)
    parser.add_argument("--kl-max-steps", type=int, default=12)
    parser.add_argument("--result-filename", type=str, default="batch_results.json")
    parser.add_argument(
        "--enable-target-kl",
        action="store_true",
        help="Enable target_kl runs from jobs file. Disabled by default.",
    )
    parser.add_argument("--progress-filename", type=str, default="batch_progress.json")
    parser.add_argument("--partial-filename", type=str, default="batch_results.partial.json")
    return parser.parse_args()


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


def _collect_target_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    if args.target_pairs:
        for item in args.target_pairs:
            pairs.append(_parse_pair_text(item))

    if args.target_pairs_file:
        path = Path(args.target_pairs_file)
        if not path.exists():
            raise FileNotFoundError(f"target pairs file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pairs.append(_parse_pair_text(line))

    if args.layer_id is not None or args.feature_id is not None:
        if args.layer_id is None or args.feature_id is None:
            raise ValueError("When using --layer-id/--feature-id, both must be provided.")
        pairs.append((int(args.layer_id), int(args.feature_id)))

    # de-duplicate in input order
    deduped: List[Tuple[int, int]] = []
    seen = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _resolve_jobs_files(args: argparse.Namespace) -> List[Path]:
    if args.jobs_file:
        path = Path(args.jobs_file)
        if not path.exists():
            raise FileNotFoundError(f"jobs file not found: {path}")
        return [path]

    pairs = _collect_target_pairs(args)
    if not pairs:
        raise ValueError(
            "Provide --jobs-file, or provide --layer-id/--feature-id, "
            "or provide --target-pairs / --target-pairs-file."
        )

    paths: List[Path] = []
    for layer_id, feature_id in pairs:
        path = Path(args.output_root) / str(layer_id) / str(feature_id) / "batch_jobs.json"
        if not path.exists():
            raise FileNotFoundError(f"jobs file not found: {path}")
        paths.append(path)
    return paths


def _load_jobs(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid jobs payload: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _sample_next_token(last_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if float(temperature) <= 0.0:
        return torch.argmax(last_logits, dim=-1, keepdim=True)
    probs = torch.softmax(last_logits / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _prepare_prompt_tensors(
    module: ModelWithSAEModule,
    *,
    prompt_kind: str,
    prompt_text: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prompt_kind == "bos":
        bos_id = getattr(module.tokenizer, "bos_token_id", None)
        if bos_id is not None:
            input_ids = torch.tensor([[int(bos_id)]], device=module.device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=module.device)
            return input_ids, attention_mask

        fallback = getattr(module.tokenizer, "bos_token", None) or ""
        encoded = module.tokenizer(str(fallback), return_tensors="pt")
    else:
        encoded = module.tokenizer(str(prompt_text), return_tensors="pt")

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
    if module.use_hooked_transformer and "__sae_lens_obj__" in module.sae:
        sae_obj = module.sae["__sae_lens_obj__"]
        return float(
            module._compute_kl_for_value(
                tokens=input_ids,
                value=float(steer_value),
                feature_index=int(feature_id),
                sae_obj=sae_obj,
            )
        )

    clean_logits = module.run_logits(input_ids=input_ids, attention_mask=attention_mask)
    steered_logits = module.run_logits_with_feature_intervention(
        input_ids=input_ids,
        feature_index=int(feature_id),
        value=float(steer_value),
        mode="add",
        attention_mask=attention_mask,
    )
    return _mean_logits_kl(clean_logits, steered_logits, attention_mask)


@torch.no_grad()
def _run_base_logits(
    module: ModelWithSAEModule,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if module.model is None:
        raise RuntimeError("Model not loaded.")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be shape [batch, seq].")

    input_ids = input_ids.to(module.device)
    attention_mask = attention_mask.to(module.device)

    # For HookedSAETransformer, prefer plain model forward without attaching SAE.
    if module.use_hooked_transformer:
        try:
            logits = module.model(input_ids, return_type="logits")
            if isinstance(logits, torch.Tensor):
                return logits
        except Exception:
            pass

        try:
            logits = module.model.run_with_saes(input_ids, saes=[])
            if isinstance(logits, torch.Tensor):
                return logits
        except Exception:
            pass

    outputs = module.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    if not hasattr(outputs, "logits") or outputs.logits is None:
        raise RuntimeError("Model output does not contain logits.")
    return outputs.logits


@torch.no_grad()
def _find_steer_value_for_target_kl(
    module: ModelWithSAEModule,
    *,
    input_ids: torch.Tensor,
    feature_id: int,
    target_kl: float,
    tolerance: float,
    max_steps: int,
) -> Tuple[float, float]:
    if not (module.use_hooked_transformer and "__sae_lens_obj__" in module.sae):
        raise RuntimeError(
            "KL target search requires HookedSAETransformer + SAE-Lens object "
            "(same KL definition as intervention_blind_score.py)."
        )
    sae_obj = module.sae["__sae_lens_obj__"]
    clamp_values, kl_values = module._find_clamp_values_for_kl(
        input_ids,
        int(feature_id),
        sae_obj,
        target_kl=float(target_kl),
        tolerance=float(tolerance),
        max_steps=int(max_steps),
    )
    if not clamp_values:
        raise RuntimeError(f"Failed to find steer value for target KL={target_kl}.")
    steer_value = float(clamp_values[0])
    achieved_kl = float(kl_values[0]) if kl_values else float("nan")
    return steer_value, achieved_kl


@torch.no_grad()
def _generate_text_from_input(
    module: ModelWithSAEModule,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    feature_id: Optional[int] = None,
    steer_value: Optional[float] = None,
) -> Dict[str, Any]:
    tokens = input_ids.clone()
    mask = attention_mask.clone()
    prompt_len = int(tokens.shape[1])
    eos_id = getattr(module.tokenizer, "eos_token_id", None)

    for _ in range(int(max_new_tokens)):
        if feature_id is None or steer_value is None:
            logits = _run_base_logits(module, input_ids=tokens, attention_mask=mask)
        else:
            logits = module.run_logits_with_feature_intervention(
                input_ids=tokens,
                feature_index=int(feature_id),
                value=float(steer_value),
                mode="add",
                attention_mask=mask,
            )
        next_logits = logits[:, -1, :]
        next_token = _sample_next_token(next_logits, temperature=float(temperature))

        tokens = torch.cat([tokens, next_token], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    completion_ids = tokens[0, prompt_len:]
    completion_text = module.tokenizer.decode(completion_ids, skip_special_tokens=True)
    full_text = module.tokenizer.decode(tokens[0], skip_special_tokens=True)
    return {
        "completion_text": completion_text,
        "full_text": full_text,
        "generated_token_count": int(completion_ids.numel()),
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from model_with_sae import ModelWithSAEModule

    jobs_paths = _resolve_jobs_files(args)
    run_summaries: List[Dict[str, Any]] = []
    shared_model: Any = None
    shared_tokenizer: Any = None
    shared_model_key: Optional[Tuple[str, str, bool]] = None

    for jobs_path in tqdm(jobs_paths, desc="Jobs"):
        jobs = _load_jobs(jobs_path)

        metadata = jobs.get("metadata", {})
        layer_id = int(metadata.get("layer_id"))
        feature_id = int(metadata.get("feature_id"))
        width = str(args.width or metadata.get("width") or "16k")

        model_cfg = jobs.get("model_config", {})
        llm_name = str(args.llm_name or model_cfg.get("llm_name") or "google/gemma-2-2b")
        sae_release = str(args.sae_release or model_cfg.get("sae_release") or "gemma-scope-2b-pt-res")
        temperature = float(
            args.temperature if args.temperature is not None else model_cfg.get("temperature", 0.0)
        )
        max_new_tokens = min(
            100,
            int(args.max_new_tokens if args.max_new_tokens is not None else model_cfg.get("max_new_tokens", 500)),
        )
        if max_new_tokens <= 0:
            max_new_tokens = 1

        sae_uri, resolved_average_l0 = build_default_sae_path(
            layer_id=str(layer_id),
            width=width,
            release=sae_release,
            average_l0=None,
            canonical_map_path=Path(args.canonical_map_path),
        )
        requested_hooked = bool(isinstance(sae_uri, str) and sae_uri.startswith("sae-lens://"))
        model_key = (llm_name, str(args.device), requested_hooked)

        if shared_model is not None and shared_tokenizer is not None and shared_model_key == model_key:
            module = ModelWithSAEModule(
                llm_name=llm_name,
                sae_path=sae_uri,
                sae_layer=layer_id,
                feature_index=feature_id,
                device=str(args.device),
                model=shared_model,
                tokenizer=shared_tokenizer,
            )
        else:
            module = ModelWithSAEModule(
                llm_name=llm_name,
                sae_path=sae_uri,
                sae_layer=layer_id,
                feature_index=feature_id,
                device=str(args.device),
            )
            shared_model = module.model
            shared_tokenizer = module.tokenizer
            shared_model_key = model_key

        prompt_specs = jobs.get("prompt_specs", [])
        if not isinstance(prompt_specs, list) or not prompt_specs:
            raise ValueError(f"No prompt_specs in jobs file: {jobs_path}")

        interventions = jobs.get("interventions", {})
        target_kl_values = interventions.get("target_kl_values", [])
        scaled_values = interventions.get("scaled_by_max_activation", [])
        if not isinstance(target_kl_values, list):
            target_kl_values = []
        if not isinstance(scaled_values, list):
            scaled_values = []
        if not bool(args.enable_target_kl):
            target_kl_values = []

        result_payload: Dict[str, Any] = {
            "metadata": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "jobs_file": str(jobs_path),
                "layer_id": layer_id,
                "feature_id": feature_id,
                "width": width,
                "llm_name": llm_name,
                "device": str(args.device),
                "sae_release": sae_release,
                "sae_uri": sae_uri,
                "resolved_average_l0": resolved_average_l0,
                "clean_mode": "base_model",
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "kl_tolerance": float(args.kl_tolerance),
                "kl_max_steps": int(args.kl_max_steps),
                "enable_target_kl": bool(args.enable_target_kl),
            },
            "neuronpedia": jobs.get("neuronpedia", {}),
            "prompt_results": [],
        }
        progress_path = jobs_path.parent / str(args.progress_filename)
        partial_path = jobs_path.parent / str(args.partial_filename)
        steps_per_prompt = 1 + len(target_kl_values) + len(scaled_values)
        total_steps = max(1, len(prompt_specs) * steps_per_prompt)
        completed_steps = 0

        progress_payload: Dict[str, Any] = {
            "metadata": {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "jobs_file": str(jobs_path),
                "layer_id": layer_id,
                "feature_id": feature_id,
                "steps_per_prompt": steps_per_prompt,
                "total_steps": total_steps,
            },
            "progress": {
                "completed_steps": completed_steps,
                "percent": 0.0,
                "current_prompt_id": None,
                "status": "running",
            },
        }
        _write_json_atomic(progress_path, progress_payload)
        _write_json_atomic(partial_path, result_payload)

        for prompt_spec in tqdm(prompt_specs, desc="Prompts"):
            if not isinstance(prompt_spec, dict):
                continue
            prompt_id = str(prompt_spec.get("prompt_id", "unknown"))
            prompt_kind = str(prompt_spec.get("prompt_kind", "text"))
            prompt_text = str(prompt_spec.get("prompt_text", ""))
            prompt_source = str(prompt_spec.get("prompt_source", "unknown"))

            input_ids, attention_mask = _prepare_prompt_tensors(
                module,
                prompt_kind=prompt_kind,
                prompt_text=prompt_text,
            )

            clean_output = _generate_text_from_input(
                module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            completed_steps += 1
            progress_payload["metadata"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            progress_payload["progress"] = {
                "completed_steps": completed_steps,
                "percent": round(100.0 * completed_steps / total_steps, 2),
                "current_prompt_id": prompt_id,
                "status": "running",
            }
            _write_json_atomic(progress_path, progress_payload)

            run_results: List[Dict[str, Any]] = []

            for target_kl in target_kl_values:
                target_kl_float = float(target_kl)
                steer_value, achieved_kl = _find_steer_value_for_target_kl(
                    module,
                    input_ids=input_ids,
                    feature_id=feature_id,
                    target_kl=target_kl_float,
                    tolerance=float(args.kl_tolerance),
                    max_steps=int(args.kl_max_steps),
                )
                steered_output = _generate_text_from_input(
                    module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    feature_id=feature_id,
                    steer_value=steer_value,
                )
                run_results.append(
                    {
                        "intervention_method": "target_kl",
                        "target_kl": target_kl_float,
                        "steer_value": steer_value,
                        "achieved_kl": achieved_kl,
                        "steered_output": steered_output,
                    }
                )
                completed_steps += 1
                progress_payload["metadata"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
                progress_payload["progress"] = {
                    "completed_steps": completed_steps,
                    "percent": round(100.0 * completed_steps / total_steps, 2),
                    "current_prompt_id": prompt_id,
                    "status": "running",
                }
                _write_json_atomic(progress_path, progress_payload)

            for item in scaled_values:
                if not isinstance(item, dict):
                    continue
                scale = float(item.get("scale", 0.0))
                steer_value = float(item.get("steer_value", 0.0))
                achieved_kl = _compute_kl_for_value(
                    module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    feature_id=feature_id,
                    steer_value=steer_value,
                )
                steered_output = _generate_text_from_input(
                    module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    feature_id=feature_id,
                    steer_value=steer_value,
                )
                run_results.append(
                    {
                        "intervention_method": "scaled_max_activation",
                        "scale": scale,
                        "steer_value": steer_value,
                        "achieved_kl": achieved_kl,
                        "steered_output": steered_output,
                    }
                )
                completed_steps += 1
                progress_payload["metadata"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
                progress_payload["progress"] = {
                    "completed_steps": completed_steps,
                    "percent": round(100.0 * completed_steps / total_steps, 2),
                    "current_prompt_id": prompt_id,
                    "status": "running",
                }
                _write_json_atomic(progress_path, progress_payload)

            result_payload["prompt_results"].append(
                {
                    "prompt_id": prompt_id,
                    "prompt_kind": prompt_kind,
                    "prompt_source": prompt_source,
                    "prompt_text": prompt_text,
                    "clean_output": clean_output,
                    "interventions": run_results,
                }
            )
            _write_json_atomic(partial_path, result_payload)

        result_path = jobs_path.parent / str(args.result_filename)
        _write_json_atomic(result_path, result_payload)
        progress_payload["metadata"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        progress_payload["progress"] = {
            "completed_steps": completed_steps,
            "percent": round(100.0 * completed_steps / total_steps, 2),
            "current_prompt_id": None,
            "status": "done",
        }
        _write_json_atomic(progress_path, progress_payload)
        run_summaries.append(
            {
                "jobs_file": str(jobs_path),
                "result_path": str(result_path),
                "progress_path": str(progress_path),
                "partial_path": str(partial_path),
                "layer_id": layer_id,
                "feature_id": feature_id,
                "prompt_count": len(result_payload["prompt_results"]),
                "intervention_per_prompt": (
                    len(result_payload["prompt_results"][0]["interventions"])
                    if result_payload["prompt_results"]
                    else 0
                ),
            }
        )

        # Release per-job tensors/SAE refs aggressively to reduce transient RAM spikes.
        if hasattr(module, "sae"):
            module.sae = {}
        del module
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    print(json.dumps({"runs": run_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
