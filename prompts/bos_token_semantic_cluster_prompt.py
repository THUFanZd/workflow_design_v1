from __future__ import annotations

import json
from typing import Sequence


def build_bos_token_semantic_cluster_system_prompt() -> str:
    return (
        "You are an expert linguistics and semantics clustering assistant.\n"
        "Your job is to merge pre-grouped token clusters into broader semantic groups.\n"
        "You must only use cluster IDs provided by the user.\n"
        "Do not split clusters, do not create new IDs, and do not drop IDs.\n"
        "Output JSON only."
    )


def build_bos_token_semantic_cluster_user_prompt(
    *,
    clusters: Sequence[dict],
    max_clusters: int,
) -> str:
    return (
        "Task:\n"
        "Given morphology-based token clusters, merge them into semantic clusters.\n"
        "Rules:\n"
        f"- Keep final cluster count <= {max_clusters} whenever possible.\n"
        "- Every source cluster ID must appear exactly once in the output.\n"
        "- You may only merge existing clusters; do not split any source cluster.\n"
        "- Do not invent new cluster IDs.\n\n"
        "Output format (JSON only):\n"
        "{\n"
        '  "merged_clusters": [\n'
        '    {"cluster_ids": [1, 3], "theme": "short semantic label"},\n'
        '    {"cluster_ids": [2], "theme": "short semantic label"}\n'
        "  ]\n"
        "}\n\n"
        "Input morphology clusters:\n"
        f"{json.dumps(list(clusters), ensure_ascii=False, indent=2)}"
    )
