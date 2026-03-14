from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from openai import OpenAI


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_usage_counts(usage: Any) -> Dict[str, int]:
    """
    Extract prompt/completion/total token counts from OpenAI usage-like objects.
    """
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if isinstance(usage, Mapping):
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        total_tokens = _safe_int(usage.get("total_tokens"))
    else:
        prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
        total_tokens = _safe_int(getattr(usage, "total_tokens", 0))

    if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


@dataclass
class TokenUsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> Dict[str, int]:
        counts = extract_usage_counts(usage)
        self.prompt_tokens += counts["prompt_tokens"]
        self.completion_tokens += counts["completion_tokens"]
        self.total_tokens += counts["total_tokens"]
        return counts

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def read_api_key(api_key_file: str) -> str:
    key = Path(api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {api_key_file}")
    return key


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None

    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None

    try:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        return None
    return None


def call_llm_stream(
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Any]:
    response_stream = client.chat.completions.create(
        model=model,
        messages=list(messages),
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
    )

    output_parts = []
    usage_obj = None
    for chunk in response_stream:
        if getattr(chunk, "choices", None):
            delta = chunk.choices[0].delta.content
            if delta:
                output_parts.append(delta)
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

    return "".join(output_parts).strip(), usage_obj
