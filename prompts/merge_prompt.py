from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Sequence

SideType = Literal["input", "output"]


def _side_label(side: SideType) -> str:
    return "input-side" if side == "input" else "output-side"


def build_system_prompt(side: SideType) -> str:
    return (
        "You are merging semantically equivalent SAE hypotheses.\n"
        f"Target: {_side_label(side)} hypotheses.\n"
        "Rules:\n"
        "1) Merge ONLY when two hypotheses are semantically identical in all essential meaning.\n"
        "2) If there is any semantic difference, do NOT merge.\n"
        "3) Be conservative. Uncertain pairs must stay separate.\n"
        "4) Keep each merged hypothesis concise and faithful to the originals.\n"
        "5) Output strictly one JSON object and no extra text."
    )


def build_user_prompt(
    *,
    side: SideType,
    hypotheses: Sequence[str],
    reasons: Sequence[str],
) -> str:
    items: List[Dict[str, Any]] = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        reason = ""
        if index - 1 < len(reasons):
            reason = str(reasons[index - 1]).strip()
        items.append(
            {
                "index": index,
                "hypothesis": str(hypothesis).strip(),
                "reason": reason,
            }
        )

    return (
        "Task: Merge semantically equivalent hypotheses for the "
        f"{_side_label(side)} list below.\n"
        "Critical constraint: Only merge when meaning is fully the same. "
        "If there is any semantic difference, keep them separate.\n\n"
        "Input hypotheses:\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON exactly in this schema:\n"
        "{\n"
        '  "merged": [\n'
        "    {\n"
        '      "hypothesis": "merged hypothesis text",\n'
        '      "source_indices": [1, 3]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Requirements:\n"
        "- Every input index must appear exactly once across all source_indices.\n"
        "- Keep output order by the first source index in each merged group.\n"
        "- Do not output markdown."
    )
