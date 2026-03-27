from __future__ import annotations

import argparse
import hashlib
import re
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from function import DEFAULT_CANONICAL_MAP_PATH, build_default_sae_path
from neuronpedia_feature_api import extract_explanations, fetch_feature_json

Decision = Literal["ACTIVATE", "DO_NOT_ACTIVATE"]

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_API_KEY_FILE = (
    "C:\\Users\\lzx\\Desktop\\\u7814\u4e00\u4e0b\\keys\\ali_api_key.txt"
)
DEBUG_LLM_IO_PATH = Path("./outputs/llm_inout.md")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "explanation_quality_evaluation" / "input-side-evaluation" / "outputs"
DEFAULT_REFERENCE_CACHE_ROOT = DEFAULT_OUTPUT_ROOT

SAE_RELEASE_BY_NAME: Dict[str, str] = {
    "gemmascope-res": "gemma-scope-2b-pt-res",
}


# DEBUG LOGGING BLOCK (easy to remove): write full LLM input/output for each judge call.
def _append_llm_io_log(system_prompt: str, user_prompt: str, raw_output: str) -> None:
    DEBUG_LLM_IO_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with DEBUG_LLM_IO_PATH.open("a", encoding="utf-8") as f:
        f.write(f"## LLM Call {ts}\n\n")
        f.write("### System Prompt\n\n")
        f.write("```text\n")
        f.write(system_prompt)
        f.write("\n```\n\n")
        f.write("### User Prompt\n\n")
        f.write("```text\n")
        f.write(user_prompt)
        f.write("\n```\n\n")
        f.write("### Raw Output\n\n")
        f.write("```text\n")
        f.write(raw_output)
        f.write("\n```\n\n---\n\n")


@dataclass
class ExplanationScore:
    reference_explanation: str
    sample_count: int
    adherence: float
    neuronpedia_activation_accuracy: float
    my_activation_accuracy: float
    relative_quality: Optional[float]


@dataclass
class NonActivationExplanationScore:
    reference_explanation: str
    sample_count: int
    adherence: float
    neuronpedia_non_activation_accuracy: float
    my_non_activation_accuracy: float
    relative_quality: Optional[float]


@dataclass
class BoundaryCaseResult:
    context: str
    activation_max: float
    is_non_activated: bool


@dataclass
class BoundaryScore:
    explanation: str
    sample_count: int
    activation_threshold: float
    non_activation_rate: float
    details: List[BoundaryCaseResult]


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "unknown"


def _extract_output_layout_from_source(source: str) -> Tuple[str, str]:
    parts = [p for p in source.split("-") if p]
    if parts and parts[0].isdigit():
        layer_id = parts[0]
        if len(parts) >= 3:
            sae_name = "-".join(parts[1:-1]) or parts[1]
            return layer_id, sae_name
        if len(parts) >= 2:
            return layer_id, parts[1]
        return layer_id, source
    return "unknown", source


def _prepare_run_output_paths(
    source: str,
    feature_id: str,
    *,
    output_root: Path,
    output_timestamp: Optional[str] = None,
) -> Dict[str, Path]:
    layer_id, sae_name = _extract_output_layout_from_source(source)
    run_dir_base = (
        output_root
        / _sanitize_path_component(sae_name)
        / f"layer-{_sanitize_path_component(layer_id)}"
        / f"feature-{_sanitize_path_component(str(feature_id))}"
    )
    run_dir = (
        run_dir_base / _sanitize_path_component(str(output_timestamp))
        if output_timestamp and str(output_timestamp).strip()
        else run_dir_base
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "llm_io_log": run_dir / "llm_inout.md",
        "evaluation_record": run_dir / "evaluation_record.json",
        "result_json": run_dir / "result.json",
        "result_md": run_dir / "result.md",
    }


def _prepare_reference_cache_path(
    source: str,
    feature_id: str,
    *,
    cache_root: Path,
) -> Path:
    layer_id, sae_name = _extract_output_layout_from_source(source)
    cache_dir = (
        cache_root
        / _sanitize_path_component(sae_name)
        / f"layer-{_sanitize_path_component(layer_id)}"
        / f"feature-{_sanitize_path_component(str(feature_id))}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "neuronpedia_reference_cache.json"


def _load_reference_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _is_cache_compatible(cache_data: Dict[str, Any], expected_signature: Dict[str, Any]) -> bool:
    return cache_data.get("signature") == expected_signature


def _cache_signature_hash(signature: Dict[str, Any]) -> str:
    canonical = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _iter_reference_cache_candidates(base_path: Path) -> List[Path]:
    ordered_candidates: List[Path] = []
    if base_path.exists():
        ordered_candidates.append(base_path)
    pattern = f"{base_path.stem}-*{base_path.suffix}"
    ordered_candidates.extend(
        sorted(
            base_path.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    candidates: List[Path] = []
    seen: set[str] = set()
    for path in ordered_candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)
    return candidates


def _find_compatible_reference_cache(
    *,
    base_path: Path,
    expected_signature: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    for candidate in _iter_reference_cache_candidates(base_path):
        payload = _load_reference_cache(candidate)
        if payload and _is_cache_compatible(payload, expected_signature):
            return payload, candidate
    return None, None


def _to_boundary_score(payload: Dict[str, Any]) -> BoundaryScore:
    details_raw = payload.get("details")
    details: List[BoundaryCaseResult] = []
    if isinstance(details_raw, list):
        for item in details_raw:
            if not isinstance(item, dict):
                continue
            details.append(
                BoundaryCaseResult(
                    context=str(item.get("context", "")),
                    activation_max=float(item.get("activation_max", 0.0)),
                    is_non_activated=bool(item.get("is_non_activated", False)),
                )
            )
    return BoundaryScore(
        explanation=str(payload.get("explanation", "")),
        sample_count=int(payload.get("sample_count", len(details))),
        activation_threshold=float(payload.get("activation_threshold", 0.0)),
        non_activation_rate=float(payload.get("non_activation_rate", 0.0)),
        details=details,
    )


def _read_api_key(api_key_file: str) -> str:
    key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {api_key_file}")
    return key


def build_client(api_key_file: str, base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=_read_api_key(api_key_file))


def _extract_json_any(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except Exception:
            continue
    return None


def _extract_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return [normalized] if normalized else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_extract_string_list(item))
        return out
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(_extract_string_list(item))
        return out
    return []


def generate_boundary_contexts(
    client: OpenAI,
    model: str,
    explanation: str,
    boundary_case_count: int = 5,
    max_tokens: int = 10000,
    temperature: float = 0.2,
    llm_io_logger: Optional[Any] = None,
) -> List[str]:
    if boundary_case_count <= 0:
        raise ValueError("boundary_case_count must be a positive integer.")

    system_prompt = (
        "You are an expert at designing adversarial boundary test cases for SAE feature explanations."
        " Return JSON only."
    )
    user_prompt = (
        "Task: generate boundary contexts for the hypothesis below.\n\n"
        "Definition of boundary case (critical):\n"
        "- A boundary case is near the edge of the explained set by the feature explanation:\\\n"
        "  it looks lexically/semantically similar,\n"
        "  but should still fall OUTSIDE the true activation set, which means: \n"
        "- The case should be tempting and confusable, but as long as it sticks closely to the explanation,\n"
        "  the feature should NOT activate strongly on that case.\n"
        "- Use multiple near-miss types when possible (context shift, minimal lexical edits,\n"
        "  homophone/orthographic variants, same surface form in a different domain, etc.).\n\n"
        f"Hypothesis / explanation:\n{explanation}\n\n"
        f"Generate exactly {boundary_case_count} boundary contexts.\n"
        "Again, boundary cases should try NOT to activate the SAE feature."
        "Each context should be a single natural sentence or short snippet.\n"
        "Return JSON only in this format:\n"
        "{\n"
        '  "boundary_cases": ["case 1", "case 2", "case 3", "case 4", "case 5"]\n'
        "}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )

    content = (response.choices[0].message.content or "").strip().strip("`")
    if llm_io_logger is not None:
        llm_io_logger(system_prompt, user_prompt, content)

    parsed = _extract_json_any(content)
    candidates: List[str] = []
    if isinstance(parsed, dict):
        for key in ("boundary_cases", "cases", "examples", "contexts", "items"):
            if key in parsed:
                candidates.extend(_extract_string_list(parsed[key]))
        if not candidates:
            candidates.extend(_extract_string_list(parsed))
    elif parsed is not None:
        candidates.extend(_extract_string_list(parsed))

    if not candidates:
        for line in content.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                candidates.append(" ".join(cleaned.split()))

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)

    if len(deduped) < boundary_case_count:
        raise ValueError(
            f"Boundary case generator returned {len(deduped)} cases, "
            f"but {boundary_case_count} are required. Raw output: {content}"
        )
    return deduped[:boundary_case_count]


def restore_sentence(tokens: Sequence[Any]) -> str:
    text = "".join(str(token) for token in tokens)
    replacements = {
        "\u2581": " ",  # sentencepiece marker
        "\u0120": " ",  # GPT-2 style marker
        "\u010a": "\n",  # newline marker
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.split())


def _safe_max_token(activation: Dict[str, Any]) -> str:
    tokens = activation.get("tokens")
    max_idx = activation.get("maxValueTokenIndex")
    if not isinstance(tokens, list) or not isinstance(max_idx, int):
        return ""
    if max_idx < 0 or max_idx >= len(tokens):
        return ""
    token = tokens[max_idx]
    return token if isinstance(token, str) else str(token)


def _select_activations_method_1(
    activations: List[Dict[str, Any]],
    m: int,
    n: int,
) -> List[Dict[str, Any]]:
    total = len(activations)
    first_count = min(m, total)
    selected_indices: List[int] = list(range(first_count))

    for idx in range(first_count, total):
        if len(selected_indices) >= first_count + n:
            break
        current_token = _safe_max_token(activations[idx])
        last_token = _safe_max_token(activations[selected_indices[-1]]) if selected_indices else ""
        if current_token != last_token:
            selected_indices.append(idx)

    target = first_count + n
    if len(selected_indices) < target:
        for idx in range(first_count, total):
            if len(selected_indices) >= target:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

    return [activations[i] for i in selected_indices]


def _select_activations_method_2(
    activations: List[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    selected_indices: List[int] = []
    for idx, item in enumerate(activations):
        if not selected_indices:
            selected_indices.append(idx)
        else:
            current_token = _safe_max_token(item)
            last_token = _safe_max_token(activations[selected_indices[-1]])
            if current_token != last_token:
                selected_indices.append(idx)
        if len(selected_indices) >= n:
            break

    return [activations[i] for i in selected_indices]


def _select_activations_method_3(
    activations: List[Dict[str, Any]],
    m: int,
) -> List[Dict[str, Any]]:
    count = min(m, len(activations))
    return [activations[i] for i in range(count)]


def select_activation_contexts(
    feature_payload: Dict[str, Any],
    selection_method: int = 1,
    m: int = 5,
    n: int = 5,
) -> List[str]:
    if selection_method not in (1, 2, 3):
        raise ValueError("selection_method must be 1, 2, or 3.")
    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative integers.")

    activations_raw = feature_payload.get("activations") or []
    activations: List[Dict[str, Any]] = [item for item in activations_raw if isinstance(item, dict)]
    if selection_method == 1:
        selected_activations = _select_activations_method_1(activations, m=m, n=n)
    elif selection_method == 2:
        selected_activations = _select_activations_method_2(activations, n=n)
    else:
        selected_activations = _select_activations_method_3(activations, m=m)

    contexts: List[str] = []
    for item in selected_activations:
        tokens = item.get("tokens") or []
        context = restore_sentence(tokens)
        if context:
            contexts.append(context)
    return contexts


def select_non_activation_contexts(
    feature_payload: Dict[str, Any],
    non_activation_context_count: int = 5,
) -> List[str]:
    if non_activation_context_count <= 0:
        raise ValueError("non_activation_context_count must be a positive integer.")

    activations = feature_payload.get("activations") or []
    tail = activations[-non_activation_context_count:]

    contexts: List[str] = []
    for item in tail:
        tokens = item.get("tokens") or []
        context = restore_sentence(tokens)
        if context:
            contexts.append(context)
    return contexts


def _build_source(layer_id: str, width: str) -> str:
    return f"{layer_id}-gemmascope-res-{width}"


def _normalize_decision(raw: str) -> Optional[Decision]:
    text = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"ACTIVATE", "YES", "SHOULD_ACTIVATE"}:
        return "ACTIVATE"
    if text in {"DO_NOT_ACTIVATE", "NO", "SHOULD_NOT_ACTIVATE", "DONT_ACTIVATE"}:
        return "DO_NOT_ACTIVATE"
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Directly change into json or extract json from text then parse it.
    """
    text = text.strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
        return None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return None
    return None


def judge_should_activate(
    client: OpenAI,
    model: str,
    explanation: str,
    context: str,
    max_tokens: int = 10000,
) -> Decision:
    system_prompt = (
        "You are a strict evaluator of SAE feature explanations."
        " You must decide whether the feature should have high activation in a given context."
        " Return JSON only."
    )
    user_prompt = (
        "Task: decide whether this feature should have high activation in the given context.\n"
        "Evaluation rules:\n"
        "1) Use both literal lexical overlap and semantic/theme match.\n"
        "2) Ignore noisy tokens and unrelated artifacts.\n"  # , symbols, multilingual fragments,
        "3) Prefer conservative decisions; if evidence is weak, choose DO_NOT_ACTIVATE.\n"
        "4) Do not use output fluency/style as evidence; only concept alignment matters.\n\n"
        f"Explanation:\n{explanation}\n\n"
        f"Context:\n{context}\n\n"
        "Return JSON only, format: {\"decision\":\"ACTIVATE\"} or {\"decision\":\"DO_NOT_ACTIVATE\"}."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )
    content = (response.choices[0].message.content or "").strip().strip("`")
    _append_llm_io_log(system_prompt=system_prompt, user_prompt=user_prompt, raw_output=content)
    decision = _normalize_decision(content.strip("\"'"))
    if decision is not None:
        return decision

    try:
        parsed = _extract_json(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unexpected judge output: {content}") from exc

    if isinstance(parsed, dict):
        fields = ("decision", "label", "answer")
        for field in fields:
            value = parsed.get(field)
            if value is not None:
                decision = _normalize_decision(str(value))
                if decision is not None:
                    return decision

    raise ValueError(f"Unexpected judge output: {content}")


def _parse_source_layout(source: str) -> Tuple[int, str, str]:
    parts = [p for p in source.split("-") if p]
    if len(parts) < 3 or not parts[0].isdigit():
        raise ValueError(
            "Cannot infer SAE layer/sae/width from source. "
            "Provide --sae-path and --sae-layer explicitly."
        )
    layer = int(parts[0])
    width = parts[-1]
    sae_name = "-".join(parts[1:-1]).strip() or "gemmascope-res"
    return layer, sae_name, width


def _normalize_hf_model_name(model_id: str, hf_model_name: Optional[str]) -> str:
    if hf_model_name:
        return hf_model_name
    mapping = {
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-27b": "google/gemma-2-27b",
    }
    return mapping.get(model_id, model_id)


def _infer_gemma_scope_release(model_id: str) -> Optional[str]:
    normalized = model_id.lower()
    if "2b" in normalized and "gemma" in normalized:
        return "gemma-scope-2b-pt-res"
    if "9b" in normalized and "gemma" in normalized:
        return "gemma-scope-9b-pt-res"
    if "27b" in normalized and "gemma" in normalized:
        return "gemma-scope-27b-pt-res"
    return None


def _resolve_sae_path(
    model_id: str,
    source: str,
    sae_path: Optional[str],
    sae_variant: str,
    sae_name: Optional[str],
    sae_release: Optional[str],
    sae_average_l0: Optional[str],
    sae_canonical_map: Optional[str],
) -> str:
    if sae_path:
        return sae_path
    layer, source_sae_name, width = _parse_source_layout(source)
    resolved_sae_name = (sae_name or source_sae_name).strip()
    release = (
        (sae_release or "").strip()
        or SAE_RELEASE_BY_NAME.get(resolved_sae_name)
        or _infer_gemma_scope_release(model_id)
        or resolved_sae_name
    )

    resolved_average_l0 = (sae_average_l0 or "").strip() or None
    if not resolved_average_l0:
        match = re.search(r"average_l0_([0-9]+(?:\.[0-9]+)?)", str(sae_variant or ""))
        if match:
            resolved_average_l0 = match.group(1)

    canonical_map = Path(sae_canonical_map) if sae_canonical_map else (PROJECT_ROOT / DEFAULT_CANONICAL_MAP_PATH)
    sae_uri, _ = build_default_sae_path(
        layer_id=str(layer),
        width=width,
        release=release,
        average_l0=resolved_average_l0,
        canonical_map_path=canonical_map,
    )
    return sae_uri


def _resolve_sae_device(sae_device: str) -> str:
    if sae_device != "auto":
        return sae_device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_boundary_sae_module(
    model_id: str,
    source: str,
    feature_index: int,
    *,
    hf_model_name: Optional[str] = None,
    sae_path: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_variant: Optional[str] = None,
    sae_name: Optional[str] = None,
    sae_release: Optional[str] = None,
    sae_average_l0: Optional[str] = None,
    sae_canonical_map: Optional[str] = None,
    sae_device: str = "auto",
) -> Any:
    try:
        from model_with_sae import ModelWithSAEModule  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import model_with_sae.py. "
            "Please ensure model/SAE dependencies are installed."
        ) from exc

    resolved_sae_path = _resolve_sae_path(
        model_id,
        source,
        sae_path=sae_path,
        sae_variant=sae_variant,
        sae_name=sae_name,
        sae_release=sae_release,
        sae_average_l0=sae_average_l0,
        sae_canonical_map=sae_canonical_map,
    )
    resolved_sae_layer = sae_layer
    if resolved_sae_layer is None:
        try:
            inferred_layer, _, _ = _parse_source_layout(source)
            resolved_sae_layer = inferred_layer
        except ValueError:
            resolved_sae_layer = None
    resolved_hf_model_name = _normalize_hf_model_name(model_id, hf_model_name)
    resolved_device = _resolve_sae_device(sae_device)

    module = ModelWithSAEModule(
        llm_name=resolved_hf_model_name,
        sae_path=resolved_sae_path,
        sae_layer=resolved_sae_layer,
        feature_index=feature_index,
        device=resolved_device,
        debug=False,
    )
    if not getattr(module, "sae", None):
        raise RuntimeError(
            "SAE failed to load for boundary evaluation. "
            "Please check --sae-path / --sae-layer / model compatibility."
        )
    return module


def evaluate_boundary_with_sae(
    contexts: Sequence[str],
    explanation: str,
    model_id: str,
    source: str,
    feature_index: int,
    *,
    activation_threshold: float = 0.0,
    hf_model_name: Optional[str] = None,
    sae_path: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_variant: Optional[str] = None,
    sae_device: str = "auto",
    module: Optional[Any] = None,
) -> BoundaryScore:
    if not contexts:
        raise ValueError("No boundary contexts available for evaluation.")

    sae_module = module or build_boundary_sae_module(
        model_id=model_id,
        source=source,
        feature_index=feature_index,
        hf_model_name=hf_model_name,
        sae_path=sae_path,
        sae_layer=sae_layer,
        sae_variant=sae_variant,
        sae_device=sae_device,
    )

    details: List[BoundaryCaseResult] = []
    non_activation_hits = 0
    for ctx in contexts:
        trace = sae_module.get_activation_trace(ctx)
        activation = float(trace.get("summary_activation", 0.0))
        is_non_activated = activation <= activation_threshold
        if is_non_activated:
            non_activation_hits += 1
        details.append(
            BoundaryCaseResult(
                context=ctx,
                activation_max=activation,
                is_non_activated=is_non_activated,
            )
        )

    non_activation_rate = non_activation_hits / len(contexts)
    return BoundaryScore(
        explanation=explanation,
        sample_count=len(contexts),
        activation_threshold=activation_threshold,
        non_activation_rate=non_activation_rate,
        details=details,
    )


def _activation_accuracy(decisions: Sequence[Decision]) -> float:
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d == "ACTIVATE") / len(decisions)


def _non_activation_accuracy(decisions: Sequence[Decision]) -> float:
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d == "DO_NOT_ACTIVATE") / len(decisions)


def _score_activation_against_reference_decisions(
    *,
    reference_explanation: str,
    ref_decisions: Sequence[Decision],
    my_decisions: Sequence[Decision],
) -> ExplanationScore:
    if not ref_decisions:
        raise ValueError("No reference decisions available for activation evaluation.")
    if len(my_decisions) != len(ref_decisions):
        raise ValueError("Decision count mismatch.")

    sample_count = len(ref_decisions)
    adherence = (
        sum(1 for left, right in zip(ref_decisions, my_decisions) if left == right) / sample_count
    )
    np_acc = _activation_accuracy(ref_decisions)
    my_acc = _activation_accuracy(my_decisions)
    relative_quality = (my_acc / np_acc) if np_acc > 0 else None
    return ExplanationScore(
        reference_explanation=reference_explanation,
        sample_count=sample_count,
        adherence=adherence,
        neuronpedia_activation_accuracy=np_acc,
        my_activation_accuracy=my_acc,
        relative_quality=relative_quality,
    )


def _score_non_activation_against_reference_decisions(
    *,
    reference_explanation: str,
    ref_decisions: Sequence[Decision],
    my_decisions: Sequence[Decision],
) -> NonActivationExplanationScore:
    if not ref_decisions:
        raise ValueError("No reference decisions available for non-activation evaluation.")
    if len(my_decisions) != len(ref_decisions):
        raise ValueError("Decision count mismatch.")

    sample_count = len(ref_decisions)
    adherence = (
        sum(1 for left, right in zip(ref_decisions, my_decisions) if left == right) / sample_count
    )
    np_acc = _non_activation_accuracy(ref_decisions)
    my_acc = _non_activation_accuracy(my_decisions)
    relative_quality = (my_acc / np_acc) if np_acc > 0 else None
    return NonActivationExplanationScore(
        reference_explanation=reference_explanation,
        sample_count=sample_count,
        adherence=adherence,
        neuronpedia_non_activation_accuracy=np_acc,
        my_non_activation_accuracy=my_acc,
        relative_quality=relative_quality,
    )


def evaluate_against_reference(
    client: OpenAI,
    model: str,
    contexts: Sequence[str],
    reference_explanation: str,
    my_explanation: str,
    my_decisions: Optional[Sequence[Decision]] = None,
) -> ExplanationScore:
    if not contexts:
        raise ValueError("No contexts available for evaluation.")

    ref_decisions: List[Decision] = [
        judge_should_activate(client, model, reference_explanation, ctx) for ctx in contexts
    ]
    own_decisions: Sequence[Decision] = (
        my_decisions
        if my_decisions is not None
        else [judge_should_activate(client, model, my_explanation, ctx) for ctx in contexts]
    )

    if len(own_decisions) != len(ref_decisions):
        raise ValueError("Decision count mismatch.")

    sample_count = len(contexts)
    adherence = (
        sum(1 for left, right in zip(ref_decisions, own_decisions) if left == right) / sample_count
    )
    np_acc = _activation_accuracy(ref_decisions)
    my_acc = _activation_accuracy(list(own_decisions))
    relative_quality = (my_acc / np_acc) if np_acc > 0 else None

    return ExplanationScore(
        reference_explanation=reference_explanation,
        sample_count=sample_count,
        adherence=adherence,
        neuronpedia_activation_accuracy=np_acc,
        my_activation_accuracy=my_acc,
        relative_quality=relative_quality,
    )


def evaluate_non_activation_against_reference(
    client: OpenAI,
    model: str,
    contexts: Sequence[str],
    reference_explanation: str,
    my_explanation: str,
    my_decisions: Optional[Sequence[Decision]] = None,
) -> NonActivationExplanationScore:
    if not contexts:
        raise ValueError("No contexts available for evaluation.")

    ref_decisions: List[Decision] = [
        judge_should_activate(client, model, reference_explanation, ctx) for ctx in contexts
    ]
    own_decisions: Sequence[Decision] = (
        my_decisions
        if my_decisions is not None
        else [judge_should_activate(client, model, my_explanation, ctx) for ctx in contexts]
    )

    if len(own_decisions) != len(ref_decisions):
        raise ValueError("Decision count mismatch.")

    sample_count = len(contexts)
    adherence = (
        sum(1 for left, right in zip(ref_decisions, own_decisions) if left == right) / sample_count
    )
    np_acc = _non_activation_accuracy(ref_decisions)
    my_acc = _non_activation_accuracy(list(own_decisions))
    relative_quality = (my_acc / np_acc) if np_acc > 0 else None

    return NonActivationExplanationScore(
        reference_explanation=reference_explanation,
        sample_count=sample_count,
        adherence=adherence,
        neuronpedia_non_activation_accuracy=np_acc,
        my_non_activation_accuracy=my_acc,
        relative_quality=relative_quality,
    )


def compare_with_neuronpedia_explanations(
    model_id: str,
    source: str,
    index: str,
    my_explanation: str,
    *,
    max_explanations: int = 3,
    selection_method: int = 1,
    m: int = 5,
    n: int = 5,
    non_activation_context_count: int = 5,
    neuronpedia_api_key: Optional[str] = None,
    neuronpedia_timeout: int = 30,
    llm_model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key_file: str = DEFAULT_API_KEY_FILE,
    enable_activation_score: bool = True,
    enable_non_activation_score: bool = True,
    enable_boundary_score: bool = True,
    boundary_case_count: int = 5,
    boundary_max_tokens: int = 1200,
    boundary_llm_model: Optional[str] = None,
    boundary_activation_threshold: float = 0.0,
    hf_model_name: Optional[str] = None,
    sae_path: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_variant: Optional[str] = None,
    sae_name: Optional[str] = None,
    sae_release: Optional[str] = None,
    sae_average_l0: Optional[str] = None,
    sae_canonical_map: Optional[str] = None,
    sae_device: str = "auto",
    output_root: Optional[Path] = None,
    output_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    global DEBUG_LLM_IO_PATH
    output_paths = _prepare_run_output_paths(
        source=source,
        feature_id=index,
        output_root=Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT,
        output_timestamp=output_timestamp,
    )
    DEBUG_LLM_IO_PATH = output_paths["llm_io_log"]

    reference_cache_base_path = _prepare_reference_cache_path(
        source=source,
        feature_id=index,
        cache_root=DEFAULT_REFERENCE_CACHE_ROOT,
    )
    cache_signature = {
        "model_id": model_id,
        "source": source,
        "feature_id": str(index),
        "max_explanations": int(max_explanations),
        "selection_method": int(selection_method),
        "m": int(m),
        "n": int(n),
        "non_activation_context_count": int(non_activation_context_count),
        "enable_activation_score": bool(enable_activation_score),
        "enable_non_activation_score": bool(enable_non_activation_score),
        "enable_boundary_score": bool(enable_boundary_score),
        "boundary_case_count": int(boundary_case_count),
        "boundary_max_tokens": int(boundary_max_tokens),
        "boundary_llm_model": boundary_llm_model or llm_model,
        "boundary_activation_threshold": float(boundary_activation_threshold),
        "llm_model": llm_model,
    }
    cached_reference, matched_cache_path = _find_compatible_reference_cache(
        base_path=reference_cache_base_path,
        expected_signature=cache_signature,
    )
    use_reference_cache = cached_reference is not None
    cache_hash = _cache_signature_hash(cache_signature)
    reference_cache_path = (
        matched_cache_path
        if matched_cache_path is not None
        else reference_cache_base_path.with_name(
            f"{reference_cache_base_path.stem}-{cache_hash}{reference_cache_base_path.suffix}"
        )
    )

    if use_reference_cache:
        reference_explanations = [
            str(item) for item in (cached_reference.get("reference_explanations") or [])
        ]
        if not reference_explanations:
            raise ValueError("Reference cache is invalid: no reference explanations found.")
        contexts = [str(item) for item in (cached_reference.get("activation_contexts") or [])]
        non_activation_contexts = [
            str(item) for item in (cached_reference.get("non_activation_contexts") or [])
        ]
        activation_reference_decisions_raw = cached_reference.get("activation_reference_decisions") or {}
        non_activation_reference_decisions_raw = (
            cached_reference.get("non_activation_reference_decisions") or {}
        )
        boundary_reference_scores_raw = cached_reference.get("boundary_reference_scores") or []
    else:
        payload = fetch_feature_json(
            model_id=model_id,
            source=source,
            feature_id=index,
            api_key=neuronpedia_api_key,
            timeout=neuronpedia_timeout,
        )
        reference_explanations = extract_explanations(payload, limit=max_explanations)
        if not reference_explanations:
            raise ValueError("No explanation found in Neuronpedia response.")

        contexts: List[str] = []
        non_activation_contexts: List[str] = []
        if enable_activation_score:
            contexts = select_activation_contexts(
                payload,
                selection_method=selection_method,
                m=m,
                n=n,
            )
            if not contexts:
                raise ValueError("No contexts selected from activations.")

        if enable_non_activation_score:
            non_activation_contexts = select_non_activation_contexts(
                payload,
                non_activation_context_count=non_activation_context_count,
            )
            if not non_activation_contexts:
                raise ValueError("No non-activation contexts selected from activations.")

        activation_reference_decisions_raw: Dict[str, List[Decision]] = {}
        non_activation_reference_decisions_raw: Dict[str, List[Decision]] = {}
        boundary_reference_scores_raw: List[Dict[str, Any]] = []

    if enable_activation_score and not contexts:
        raise ValueError("No activation contexts available for evaluation.")
    if enable_non_activation_score and not non_activation_contexts:
        raise ValueError("No non-activation contexts available for evaluation.")

    need_llm = enable_activation_score or enable_non_activation_score or enable_boundary_score
    client: Optional[OpenAI] = build_client(api_key_file, base_url) if need_llm else None

    my_decisions: List[Decision] = []
    my_non_activation_decisions: List[Decision] = []
    if enable_activation_score:
        assert client is not None
        my_decisions = [judge_should_activate(client, llm_model, my_explanation, ctx) for ctx in contexts]
    if enable_non_activation_score:
        assert client is not None
        my_non_activation_decisions = [
            judge_should_activate(client, llm_model, my_explanation, ctx)
            for ctx in non_activation_contexts
        ]
    boundary_contexts: List[str] = []
    boundary_score: Optional[BoundaryScore] = None
    boundary_reference_scores: List[BoundaryScore] = []
    boundary_warning: Optional[str] = None
    if enable_boundary_score:
        assert client is not None
        try:
            feature_index = int(index)
        except ValueError as exc:
            raise ValueError(f"feature index must be an integer for SAE evaluation, got: {index}") from exc

        try:
            boundary_module = build_boundary_sae_module(
                model_id=model_id,
                source=source,
                feature_index=feature_index,
                hf_model_name=hf_model_name,
                sae_path=sae_path,
                sae_layer=sae_layer,
                sae_variant=sae_variant,
                sae_name=sae_name,
                sae_release=sae_release,
                sae_average_l0=sae_average_l0,
                sae_canonical_map=sae_canonical_map,
                sae_device=sae_device,
            )

            boundary_contexts = generate_boundary_contexts(
                client=client,
                model=boundary_llm_model or llm_model,
                explanation=my_explanation,
                boundary_case_count=boundary_case_count,
                max_tokens=boundary_max_tokens,
                llm_io_logger=_append_llm_io_log,
            )
            boundary_score = evaluate_boundary_with_sae(
                contexts=boundary_contexts,
                explanation=my_explanation,
                model_id=model_id,
                source=source,
                feature_index=feature_index,
                activation_threshold=boundary_activation_threshold,
                hf_model_name=hf_model_name,
                sae_path=sae_path,
                sae_layer=sae_layer,
                sae_variant=sae_variant,
                sae_device=sae_device,
                module=boundary_module,
            )

            if use_reference_cache:
                boundary_reference_scores = [
                    _to_boundary_score(item)
                    for item in boundary_reference_scores_raw
                    if isinstance(item, dict)
                ]
            else:
                for exp in reference_explanations:
                    ref_boundary_contexts = generate_boundary_contexts(
                        client=client,
                        model=boundary_llm_model or llm_model,
                        explanation=exp,
                        boundary_case_count=boundary_case_count,
                        max_tokens=boundary_max_tokens,
                        llm_io_logger=_append_llm_io_log,
                    )
                    ref_boundary_score = evaluate_boundary_with_sae(
                        contexts=ref_boundary_contexts,
                        explanation=exp,
                        model_id=model_id,
                        source=source,
                        feature_index=feature_index,
                        activation_threshold=boundary_activation_threshold,
                        hf_model_name=hf_model_name,
                        sae_path=sae_path,
                        sae_layer=sae_layer,
                        sae_variant=sae_variant,
                        sae_device=sae_device,
                        module=boundary_module,
                    )
                    boundary_reference_scores.append(ref_boundary_score)
        except Exception as exc:
            boundary_warning = (
                "Boundary scoring disabled because SAE failed to initialize: "
                f"{exc}"
            )
            print(f"[WARN] {boundary_warning}")
            enable_boundary_score = False

    details: List[ExplanationScore] = []
    non_activation_details: List[NonActivationExplanationScore] = []
    if enable_activation_score:
        assert client is not None
        for exp in reference_explanations:
            ref_decisions = activation_reference_decisions_raw.get(exp)
            if not use_reference_cache:
                ref_decisions = [judge_should_activate(client, llm_model, exp, ctx) for ctx in contexts]
                activation_reference_decisions_raw[exp] = ref_decisions
            if not ref_decisions:
                raise ValueError(f"Missing activation reference decisions for explanation: {exp}")
            score = _score_activation_against_reference_decisions(
                reference_explanation=exp,
                ref_decisions=ref_decisions,
                my_decisions=my_decisions,
            )
            details.append(score)
    if enable_non_activation_score:
        assert client is not None
        for exp in reference_explanations:
            ref_decisions = non_activation_reference_decisions_raw.get(exp)
            if not use_reference_cache:
                ref_decisions = [
                    judge_should_activate(client, llm_model, exp, ctx) for ctx in non_activation_contexts
                ]
                non_activation_reference_decisions_raw[exp] = ref_decisions
            if not ref_decisions:
                raise ValueError(
                    f"Missing non-activation reference decisions for explanation: {exp}"
                )
            non_activation_score = _score_non_activation_against_reference_decisions(
                reference_explanation=exp,
                ref_decisions=ref_decisions,
                my_decisions=my_non_activation_decisions,
            )
            non_activation_details.append(non_activation_score)
    relative_quality_score: Optional[float] = None
    adherence: Optional[float] = None
    if enable_activation_score:
        relative_quality_values = [item.relative_quality for item in details if item.relative_quality is not None]
        adherence_values = [item.adherence for item in details]
        relative_quality_score = (
            sum(relative_quality_values) / len(relative_quality_values)
            if relative_quality_values
            else None
        )
        adherence = sum(adherence_values) / len(adherence_values) if adherence_values else None

    non_activation_relative_quality_score: Optional[float] = None
    non_activation_adherence: Optional[float] = None
    if enable_non_activation_score:
        non_activation_relative_quality_values = [
            item.relative_quality for item in non_activation_details if item.relative_quality is not None
        ]
        non_activation_adherence_values = [item.adherence for item in non_activation_details]
        non_activation_relative_quality_score = (
            sum(non_activation_relative_quality_values) / len(non_activation_relative_quality_values)
            if non_activation_relative_quality_values
            else None
        )
        non_activation_adherence = (
            sum(non_activation_adherence_values) / len(non_activation_adherence_values)
            if non_activation_adherence_values
            else None
        )

    boundary_reference_mean_non_activation_rate: Optional[float] = None
    boundary_relative_quality_score: Optional[float] = None
    if enable_boundary_score and boundary_reference_scores:
        boundary_reference_mean_non_activation_rate = (
            sum(item.non_activation_rate for item in boundary_reference_scores)
            / len(boundary_reference_scores)
        )
        if boundary_score is not None:
            boundary_relative_values = [
                boundary_score.non_activation_rate / item.non_activation_rate
                for item in boundary_reference_scores
                if item.non_activation_rate > 0
            ]
            boundary_relative_quality_score = (
                sum(boundary_relative_values) / len(boundary_relative_values)
                if boundary_relative_values
                else None
            )

    result = {
        "feature": {"model_id": model_id, "source": source, "index": index},
        "evaluations_enabled": {
            "activation": enable_activation_score,
            "non_activation": enable_non_activation_score,
            "boundary": enable_boundary_score,
        },
        "num_reference_explanations": len(reference_explanations),
        "num_contexts": len(contexts) if enable_activation_score else None,
        "relative_quality_score": relative_quality_score,
        "adherence": adherence,
        "details": [asdict(item) for item in details] if enable_activation_score else [],
        "num_non_activation_contexts": len(non_activation_contexts) if enable_non_activation_score else None,
        "non_activation_relative_quality_score": non_activation_relative_quality_score,
        "non_activation_adherence": non_activation_adherence,
        "non_activation_details": [asdict(item) for item in non_activation_details] if enable_non_activation_score else [],
        "num_boundary_contexts": len(boundary_contexts) if enable_boundary_score else None,
        "boundary_non_activation_rate": (
            boundary_score.non_activation_rate if enable_boundary_score and boundary_score is not None else None
        ),
        "boundary_activation_threshold": (
            boundary_score.activation_threshold if enable_boundary_score and boundary_score is not None else None
        ),
        "boundary_details": (
            [asdict(item) for item in boundary_score.details]
            if enable_boundary_score and boundary_score
            else []
        ),
        "boundary_reference_mean_non_activation_rate": (
            boundary_reference_mean_non_activation_rate if enable_boundary_score else None
        ),
        "boundary_relative_quality_score": (
            boundary_relative_quality_score if enable_boundary_score else None
        ),
        "boundary_reference_details": (
            [asdict(item) for item in boundary_reference_scores] if enable_boundary_score else []
        ),
        "output_paths": {
            "run_dir": str(output_paths["run_dir"]),
            "llm_io_log": str(output_paths["llm_io_log"]),
            "evaluation_record": str(output_paths["evaluation_record"]),
            "result_json": str(output_paths["result_json"]),
            "result_md": str(output_paths["result_md"]),
        },
        "neuronpedia_reference_cache": {
            "path": str(reference_cache_path),
            "hit": use_reference_cache,
            "signature_hash": cache_hash,
        },
        "boundary_warning": boundary_warning,
    }

    activation_case_results = [
        {
            "context": ctx,
            "expected": "ACTIVATE",
            "my_decision": decision,
            "is_correct": decision == "ACTIVATE",
        }
        for ctx, decision in zip(contexts, my_decisions)
    ]
    non_activation_case_results = [
        {
            "context": ctx,
            "expected": "DO_NOT_ACTIVATE",
            "my_decision": decision,
            "is_correct": decision == "DO_NOT_ACTIVATE",
        }
        for ctx, decision in zip(non_activation_contexts, my_non_activation_decisions)
    ]

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature": {"model_id": model_id, "source": source, "index": index},
        "hypothesis": my_explanation,
        "reference_explanations": reference_explanations,
        "evaluations_enabled": result["evaluations_enabled"],
        "test_cases": {
            "activation_cases": activation_case_results,
            "non_activation_cases": non_activation_case_results,
            "boundary_cases_my": (
                [asdict(item) for item in boundary_score.details]
                if enable_boundary_score and boundary_score
                else []
            ),
            "boundary_cases_reference": (
                [asdict(item) for item in boundary_reference_scores]
                if enable_boundary_score
                else []
            ),
        },
        "scores": {
            "activation_relative_quality_score": relative_quality_score,
            "activation_adherence": adherence,
            "non_activation_relative_quality_score": non_activation_relative_quality_score,
            "non_activation_adherence": non_activation_adherence,
            "boundary_non_activation_rate": (
                boundary_score.non_activation_rate if enable_boundary_score and boundary_score is not None else None
            ),
            "boundary_activation_threshold": (
                boundary_score.activation_threshold if enable_boundary_score and boundary_score is not None else None
            ),
            "boundary_reference_mean_non_activation_rate": (
                boundary_reference_mean_non_activation_rate if enable_boundary_score else None
            ),
            "boundary_relative_quality_score": (
                boundary_relative_quality_score if enable_boundary_score else None
            ),
        },
        "summary": {
            "num_activation_cases": len(activation_case_results),
            "num_non_activation_cases": len(non_activation_case_results),
            "num_boundary_cases_my": len(boundary_contexts) if enable_boundary_score else 0,
            "num_boundary_cases_reference": (
                sum(item.sample_count for item in boundary_reference_scores)
                if enable_boundary_score
                else 0
            ),
        },
        "neuronpedia_reference_cache": {
            "path": str(reference_cache_path),
            "hit": use_reference_cache,
            "signature_hash": cache_hash,
        },
        "boundary_warning": boundary_warning,
    }

    if not use_reference_cache:
        cache_payload = {
            "signature": cache_signature,
            "signature_hash": cache_hash,
            "cached_at": datetime.now().isoformat(timespec="seconds"),
            "feature": {"model_id": model_id, "source": source, "index": index},
            "reference_explanations": reference_explanations,
            "activation_contexts": contexts if enable_activation_score else [],
            "non_activation_contexts": (
                non_activation_contexts if enable_non_activation_score else []
            ),
            "activation_reference_decisions": activation_reference_decisions_raw,
            "non_activation_reference_decisions": non_activation_reference_decisions_raw,
            "boundary_reference_scores": (
                [asdict(item) for item in boundary_reference_scores] if enable_boundary_score else []
            ),
        }
        reference_cache_path.write_text(
            json.dumps(cache_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    output_paths["evaluation_record"].write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths["result_json"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_result_markdown(
        output_paths["result_md"],
        result=result,
        record=record,
    )
    return result


def _load_my_explanation(args: argparse.Namespace) -> str:
    if args.my_explanation and args.my_explanation_file:
        raise ValueError("Use either --my-explanation or --my-explanation-file, not both.")
    if args.my_explanation:
        return args.my_explanation.strip()
    if args.my_explanation_file:
        return Path(args.my_explanation_file).read_text(encoding="utf-8").strip()
    raise ValueError("Missing my explanation. Provide --my-explanation or --my-explanation-file.")


def _write_result_markdown(path: Path, *, result: Dict[str, Any], record: Dict[str, Any]) -> None:
    feature = result.get("feature", {})
    scores = record.get("scores", {})
    summary = record.get("summary", {})

    lines: List[str] = []
    lines.append("# Input-side Explanation Evaluation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- generated_at: {record.get('timestamp')}")
    lines.append(f"- model_id: {feature.get('model_id')}")
    lines.append(f"- source: {feature.get('source')}")
    lines.append(f"- feature_id: {feature.get('index')}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("```text")
    lines.append(str(record.get("hypothesis", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Scores")
    lines.append(f"- activation_relative_quality_score: {scores.get('activation_relative_quality_score')}")
    lines.append(f"- activation_adherence: {scores.get('activation_adherence')}")
    lines.append(f"- non_activation_relative_quality_score: {scores.get('non_activation_relative_quality_score')}")
    lines.append(f"- non_activation_adherence: {scores.get('non_activation_adherence')}")
    lines.append(f"- boundary_non_activation_rate: {scores.get('boundary_non_activation_rate')}")
    lines.append(f"- boundary_relative_quality_score: {scores.get('boundary_relative_quality_score')}")
    lines.append("")
    lines.append("## Case Counts")
    lines.append(f"- num_activation_cases: {summary.get('num_activation_cases')}")
    lines.append(f"- num_non_activation_cases: {summary.get('num_non_activation_cases')}")
    lines.append(f"- num_boundary_cases_my: {summary.get('num_boundary_cases_my')}")
    lines.append(f"- num_boundary_cases_reference: {summary.get('num_boundary_cases_reference')}")
    lines.append("")
    lines.append("## Reference Comparison")
    lines.append("| reference_explanation | activation_relative_quality | non_activation_relative_quality |")
    lines.append("| --- | ---: | ---: |")
    non_activation_map: Dict[str, Any] = {
        str(item.get("reference_explanation", "")): item.get("relative_quality")
        for item in result.get("non_activation_details", [])
        if isinstance(item, dict)
    }
    for item in result.get("details", []):
        if not isinstance(item, dict):
            continue
        reference_explanation = str(item.get("reference_explanation", ""))
        escaped_reference = reference_explanation.replace("|", "\\|")
        lines.append(
            f"| {escaped_reference} | {item.get('relative_quality')} | {non_activation_map.get(reference_explanation)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare your SAE explanation against Neuronpedia explanations "
            "using LLM-based activation judgments."
        )
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id, e.g. gemma-2-2b")
    parser.add_argument("--layer-id", required=True, help="Layer id used in Neuronpedia source, e.g. 0")
    parser.add_argument("--width", default="16k", help="Width in Neuronpedia source, e.g. 16k")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional full Neuronpedia source override. If omitted, uses {layer_id}-gemmascope-res-{width}.",
    )
    parser.add_argument(
        "--feature-id",
        "--index",
        dest="feature_id",
        required=True,
        help="Feature index",
    )
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens for LLM judge")
    parser.add_argument("--my-explanation", default=None, help="Your explanation string")
    parser.add_argument(
        "--my-explanation-file",
        default=None,
        help="Path to a UTF-8 text file containing your explanation",
    )
    parser.add_argument("--max-explanations", type=int, default=3, help="Top explanations to use")
    parser.add_argument(
        "--selection-method",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Activation sampling method (1/2/3), aligned with neuronpedia_feature_api copy.py",
    )
    parser.add_argument("--m", type=int, default=5, help="Method parameter m for activation sampling")
    parser.add_argument("--n", type=int, default=5, help="Method parameter n for activation sampling")
    parser.add_argument(
        "--non-activation-context-count",
        "--non-activation-samples",
        dest="non_activation_context_count",
        type=int,
        default=5,
        help="Count of tail activation contexts used as should-not-activate samples",
    )
    parser.add_argument(
        "--neuronpedia-api-key",
        default=None,
        help="Optional Neuronpedia API key (or set NEURONPEDIA_API_KEY)",
    )
    parser.add_argument("--neuronpedia-timeout", type=int, default=30, help="Neuronpedia timeout")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL, help="Judge model id")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM OpenAI-compatible base URL")
    parser.add_argument(
        "--api-key-file",
        default=DEFAULT_API_KEY_FILE,
        help="Path to the default LLM API key file",
    )
    parser.add_argument(
        "--disable-activation-score",
        action="store_true",
        help="Disable activation-score evaluation.",
    )
    parser.add_argument(
        "--disable-non-activation-score",
        action="store_true",
        help="Disable non-activation-score evaluation.",
    )
    parser.add_argument(
        "--disable-boundary-score",
        action="store_true",
        help="Disable boundary-case generation and SAE boundary scoring.",
    )
    parser.add_argument(
        "--boundary-case-count",
        type=int,
        default=5,
        help="Number of boundary contexts generated by LLM.",
    )
    parser.add_argument(
        "--boundary-max-tokens",
        type=int,
        default=10000,
        help="Max tokens for boundary-context generation call.",
    )
    parser.add_argument(
        "--boundary-llm-model",
        default=None,
        help="Optional separate model for boundary-case generation (defaults to --llm-model).",
    )
    parser.add_argument(
        "--boundary-activation-threshold",
        type=float,
        default=0.0,
        help="Activation threshold: activation <= threshold counts as non-activated.",
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help="Optional HuggingFace model name used by model_with_sae.py.",
    )
    parser.add_argument(
        "--sae-path",
        default=None,
        help="Optional SAE path/URI for model_with_sae.py. If omitted, inferred from model/source.",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=None,
        help="Optional SAE layer override for model_with_sae.py.",
    )
    parser.add_argument(
        "--sae-variant",
        default=None,
        help="Optional SAE variant name (e.g. average_l0_105) when SAE path is inferred.",
    )
    parser.add_argument(
        "--sae-name",
        default="gemmascope-res",
        help="SAE family name used to resolve default release and output path.",
    )
    parser.add_argument(
        "--sae-release",
        default=None,
        help="Optional explicit SAE release, e.g. gemma-scope-2b-pt-res.",
    )
    parser.add_argument(
        "--sae-average-l0",
        default=None,
        help="Optional average_l0 suffix number (e.g. 105). If omitted, resolve from canonical map.",
    )
    parser.add_argument(
        "--sae-canonical-map",
        default=str(PROJECT_ROOT / DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt used for default average_l0 resolution.",
    )
    parser.add_argument(
        "--sae-device",
        default="auto",
        help="Device for SAE scoring (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root directory. Results are saved under outputs/{sae}/layer-{id}/feature-{id}.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp subdirectory under feature path.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source or _build_source(layer_id=args.layer_id, width=args.width)
    my_explanation = _load_my_explanation(args)
    result = compare_with_neuronpedia_explanations(
        model_id=args.model_id,
        source=source,
        index=args.feature_id,
        my_explanation=my_explanation,
        max_explanations=args.max_explanations,
        selection_method=args.selection_method,
        m=args.m,
        n=args.n,
        non_activation_context_count=args.non_activation_context_count,
        neuronpedia_api_key=args.neuronpedia_api_key,
        neuronpedia_timeout=args.neuronpedia_timeout,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key_file=args.api_key_file,
        enable_activation_score=not args.disable_activation_score,
        enable_non_activation_score=not args.disable_non_activation_score,
        enable_boundary_score=not args.disable_boundary_score,
        boundary_case_count=args.boundary_case_count,
        boundary_max_tokens=args.boundary_max_tokens,
        boundary_llm_model=args.boundary_llm_model,
        boundary_activation_threshold=args.boundary_activation_threshold,
        hf_model_name=args.hf_model_name,
        sae_path=args.sae_path,
        sae_layer=args.sae_layer,
        sae_variant=args.sae_variant,
        sae_name=args.sae_name,
        sae_release=args.sae_release,
        sae_average_l0=args.sae_average_l0,
        sae_canonical_map=args.sae_canonical_map,
        sae_device=args.sae_device,
        output_root=Path(args.output_root),
        output_timestamp=args.timestamp,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
