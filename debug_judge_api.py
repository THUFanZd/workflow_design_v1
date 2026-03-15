from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from function import read_api_key
from llm_api.llm_api_info import api_key_file as DEFAULT_API_KEY_FILE
from llm_api.llm_api_info import base_url as DEFAULT_BASE_URL
from llm_api.llm_api_info import model_name as DEFAULT_MODEL_NAME


def _safe_preview(obj: Any, limit: int = 1000) -> str:
    text = str(obj)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _extract_message_fields(response: Any) -> Dict[str, Any]:
    if not getattr(response, "choices", None):
        return {"content": None, "reasoning_content": None, "message_dict": None}
    message = response.choices[0].message
    message_dict = None
    if hasattr(message, "model_dump"):
        try:
            message_dict = message.model_dump()
        except Exception:
            message_dict = None
    return {
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "message_dict": message_dict,
    }


def _run_once(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_format_text: bool,
    extra_body_empty: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format_text:
        kwargs["response_format"] = {"type": "text"}
    if extra_body_empty:
        kwargs["extra_body"] = {}

    response = client.chat.completions.create(**kwargs)
    fields = _extract_message_fields(response)

    usage = None
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    response_dump = None
    if hasattr(response, "model_dump"):
        try:
            response_dump = response.model_dump()
        except Exception:
            response_dump = None

    return {
        "request_kwargs": kwargs,
        "usage": usage,
        "content_preview": _safe_preview(fields["content"]),
        "reasoning_content_preview": _safe_preview(fields["reasoning_content"]),
        "message_dict_preview": _safe_preview(fields["message_dict"]),
        "response_dump_preview": _safe_preview(response_dump, limit=3000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug judge API outputs using stored llm_calls messages.")
    parser.add_argument(
        "--execution-json",
        default="logs/0_4/20260314_225001/layer0-feature4-experiments-execution.json",
        help="Path to experiments execution json.",
    )
    parser.add_argument("--call-index", type=int, default=1, help="1-based llm call index from execution json.")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save structured debug result JSON.",
    )
    args = parser.parse_args()

    execution_path = Path(args.execution_json)
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    llm_calls = payload.get("llm_calls", [])
    if not isinstance(llm_calls, list) or not llm_calls:
        raise ValueError("No llm_calls found in execution json.")

    idx = int(args.call_index) - 1
    if idx < 0 or idx >= len(llm_calls):
        raise IndexError(f"call-index out of range: {args.call_index}, total calls: {len(llm_calls)}")

    call_obj = llm_calls[idx]
    messages = call_obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Selected llm_call has no valid messages list.")

    client = OpenAI(
        base_url=args.llm_base_url,
        api_key=read_api_key(args.llm_api_key_file),
    )

    tests = [
        {"name": "baseline_non_stream", "response_format_text": False, "extra_body_empty": False},
        {"name": "with_response_format_text", "response_format_text": True, "extra_body_empty": False},
        {"name": "with_response_format_text_and_extra_body", "response_format_text": True, "extra_body_empty": True},
    ]

    results = []
    for test in tests:
        one = _run_once(
            client=client,
            model=args.llm_model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            response_format_text=bool(test["response_format_text"]),
            extra_body_empty=bool(test["extra_body_empty"]),
        )
        one["name"] = test["name"]
        results.append(one)

    output = {
        "execution_json": str(execution_path),
        "call_index": args.call_index,
        "llm_model": args.llm_model,
        "base_url": args.llm_base_url,
        "results": results,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved debug output to: {out_path}")


if __name__ == "__main__":
    main()

