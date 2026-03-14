from __future__ import annotations

import argparse
import re
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence

from openai import OpenAI

from neuronpedia_feature_api import extract_explanations, fetch_feature_json

Decision = Literal["ACTIVATE", "DO_NOT_ACTIVATE"]

DEFAULT_PPIO_BASE_URL = "https://api.ppio.com/openai"
DEFAULT_MODEL = "zai-org/glm-4.7"
DEFAULT_API_KEY_FILE = (
    "C:\\Users\\lzx\\Desktop\\\u7814\u4e00\u4e0b\\ppio_api_key.txt"
)
DEBUG_LLM_IO_PATH = Path("./outputs/llm_inout.md")


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


def _read_api_key(api_key_file: str) -> str:
    key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {api_key_file}")
    return key


def build_client(api_key_file: str, base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=_read_api_key(api_key_file))


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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_activation_contexts(
    feature_payload: Dict[str, Any],
    activation_ratio: float = 0.5,
    max_samples: int = 10,
) -> List[str]:
    if not (0 < activation_ratio <= 1):
        raise ValueError("activation_ratio must be in (0, 1].")
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    hist = feature_payload.get("freq_hist_data_bar_values") or []
    hist_max = _safe_float(hist[-1], default=0.0) if hist else 0.0

    activations = feature_payload.get("activations") or []
    if hist_max <= 0.0:
        derived_max = 0.0
        for item in activations:
            mv = item.get("maxValue")
            if mv is None:
                values = item.get("values") or []
                mv = max(values) if values else 0.0
            derived_max = max(derived_max, _safe_float(mv, default=0.0))
        hist_max = derived_max

    threshold = hist_max * activation_ratio

    contexts: List[str] = []
    for item in activations:
        mv = item.get("maxValue")
        if mv is None:
            values = item.get("values") or []
            mv = max(values) if values else 0.0
        max_value = _safe_float(mv, default=0.0)
        if max_value < threshold:
            continue

        tokens = item.get("tokens") or []
        context = restore_sentence(tokens)
        if context:
            contexts.append(context)
        if len(contexts) >= max_samples:
            break
    return contexts


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


def _activation_accuracy(decisions: Sequence[Decision]) -> float:
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d == "ACTIVATE") / len(decisions)


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


def compare_with_neuronpedia_explanations(
    model_id: str,
    source: str,
    index: str,
    my_explanation: str,
    *,
    max_explanations: int = 3,
    activation_ratio: float = 0.5,
    max_samples: int = 10,
    neuronpedia_api_key: Optional[str] = None,
    neuronpedia_timeout: int = 30,
    llm_model: str = DEFAULT_MODEL,
    ppio_base_url: str = DEFAULT_PPIO_BASE_URL,
    ppio_api_key_file: str = DEFAULT_API_KEY_FILE,
) -> Dict[str, Any]:
    print('in comparing:')
    payload = fetch_feature_json(
        model_id=model_id,
        source=source,
        index=index,
        api_key=neuronpedia_api_key,
        timeout=neuronpedia_timeout,
    )
    print('fetched')
    reference_explanations = extract_explanations(payload, limit=max_explanations)
    if not reference_explanations:
        raise ValueError("No explanation found in Neuronpedia response.")

    contexts = select_activation_contexts(
        payload,
        activation_ratio=activation_ratio,
        max_samples=max_samples,
    )
    if not contexts:
        raise ValueError("No contexts selected from activations.")

    print('context selected')
    client = build_client(ppio_api_key_file, ppio_base_url)
    my_decisions: List[Decision] = [
        judge_should_activate(client, llm_model, my_explanation, ctx) for ctx in contexts
    ]
    print('judged')

    details: List[ExplanationScore] = []
    for exp in reference_explanations:
        score = evaluate_against_reference(
            client=client,
            model=llm_model,
            contexts=contexts,
            reference_explanation=exp,
            my_explanation=my_explanation,
            my_decisions=my_decisions,
        )
        details.append(score)

    print('evaluated')
    relative_quality_values = [item.relative_quality for item in details if item.relative_quality is not None]
    adherence_values = [item.adherence for item in details]
    relative_quality_score = (
        sum(relative_quality_values) / len(relative_quality_values)
        if relative_quality_values
        else None
    )
    adherence = sum(adherence_values) / len(adherence_values) if adherence_values else None

    return {
        "feature": {"model_id": model_id, "source": source, "index": index},
        "num_reference_explanations": len(reference_explanations),
        "num_contexts": len(contexts),
        "relative_quality_score": relative_quality_score,
        "adherence": adherence,
        "details": [asdict(item) for item in details],
    }


def _load_my_explanation(args: argparse.Namespace) -> str:
    if args.my_explanation and args.my_explanation_file:
        raise ValueError("Use either --my-explanation or --my-explanation-file, not both.")
    if args.my_explanation:
        return args.my_explanation.strip()
    if args.my_explanation_file:
        return Path(args.my_explanation_file).read_text(encoding="utf-8").strip()
    raise ValueError("Missing my explanation. Provide --my-explanation or --my-explanation-file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare your SAE explanation against Neuronpedia explanations "
            "using LLM-based activation judgments."
        )
    )
    parser.add_argument("--model-id", default="gemma-2-2b", help="Neuronpedia model id, e.g. gemma-2-2b")
    parser.add_argument("--source", default="0-gemmascope-res-16k", help="Neuronpedia source id, e.g. 11-clt-hp")
    parser.add_argument("--index", required=True, help="Feature index")
    parser.add_argument("--my-explanation", default=None, help="Your explanation string")
    parser.add_argument(
        "--my-explanation-file",
        default=None,
        help="Path to a UTF-8 text file containing your explanation",
    )
    parser.add_argument("--max-explanations", type=int, default=3, help="Top explanations to use")
    parser.add_argument(
        "--activation-ratio",
        type=float,
        default=0.5,
        help="Threshold ratio of feature max activation for context sampling",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of activation contexts to evaluate",
    )
    parser.add_argument(
        "--neuronpedia-api-key",
        default=None,
        help="Optional Neuronpedia API key (or set NEURONPEDIA_API_KEY)",
    )
    parser.add_argument("--neuronpedia-timeout", type=int, default=30, help="Neuronpedia timeout")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL, help="Judge model id")
    parser.add_argument("--ppio-base-url", default=DEFAULT_PPIO_BASE_URL, help="PPIO OpenAI base URL")
    parser.add_argument(
        "--ppio-api-key-file",
        default=DEFAULT_API_KEY_FILE,
        help="Path to API key file used by llmapi.py",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    my_explanation = _load_my_explanation(args)
    result = compare_with_neuronpedia_explanations(
        model_id=args.model_id,
        source=args.source,
        index=args.index,
        my_explanation=my_explanation,
        max_explanations=args.max_explanations,
        activation_ratio=args.activation_ratio,
        max_samples=args.max_samples,
        neuronpedia_api_key=args.neuronpedia_api_key,
        neuronpedia_timeout=args.neuronpedia_timeout,
        llm_model=args.llm_model,
        ppio_base_url=args.ppio_base_url,
        ppio_api_key_file=args.ppio_api_key_file,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
