from __future__ import annotations

import json
from typing import Any, Dict, Literal

SideType = Literal["input", "output"]
HistoryScope = Literal["same_hypothesis", "all_hypotheses"]


def _side_label(side: SideType) -> str:
    return "input-side activation" if side == "input" else "output-side intervention"


def _scoring_explanation(side: SideType) -> str:
    if side == "input":
        return (
            "Score definition:\n"
            "- score_non_zero_rate = non_zero_count / total_sentences.\n"
            "- Higher is better.\n"
            "- Input-side test sentences should semantically match the hypothesis and activate the SAE feature."
        )
    return (
        "Score definition:\n"
        "- score_blind_accuracy = blind_judge_successes / blind_judge_trials.\n"
        "- Higher is better.\n"
        "- Output-side hypothesis should explain the semantic direction caused by feature intervention."
    )


def build_system_prompt(side: SideType) -> str:
    return (
        "You are an expert mechanistic interpretability researcher for sparse autoencoder (SAE) features.\n"
        f"You are refining {_side_label(side)} hypotheses using memory and experiment evidence.\n"
        "You must reason from score first, then inspect failures and limitations in the previous reason/hypothesis.\n"
        "Return only valid JSON with no extra text."
    )


def build_user_prompt(
    *,
    side: SideType,
    hypothesis_index: int,
    current_hypothesis: str,
    current_reason: str,
    current_score_name: str,
    current_score: float,
    current_memory_evidence: Dict[str, Any],
    history_scope: HistoryScope,
    historical_evidence: Dict[str, Any],
    current_execution_evidence: Dict[str, Any],
) -> str:
    history_scope_text = (
        "same hypothesis across previous rounds"
        if history_scope == "same_hypothesis"
        else "all hypotheses on this side across previous rounds"
    )

    process_steps = (
        "Required process:\n"
        "1) Read the score and infer how strong/weak the current hypothesis is.\n"
        "2) Analyze mismatch patterns from failed examples.\n"
    )
    if history_scope == "same_hypothesis":
        process_steps += (
            "3) Use memory trends from previous rounds.\n"
            "4) Produce a sharper and more testable revised hypothesis.\n"
            "5) Explain why the revision should perform better.\n\n"
        )
    else:
        process_steps += (
            "3) Distinguish two memory groups: (a) your own hypothesis trajectory, "
            "(b) peer hypotheses from the same side.\n"
            "4) Mine transferable failure/success patterns from all hypotheses to improve your own.\n"
            "5) Produce a sharper and more testable revised hypothesis.\n"
            "6) Explain why the revision should perform better.\n\n"
        )

    return (
        "Background:\n"
        "You are improving one SAE feature hypothesis in an iterative workflow.\n"
        "The previous hypothesis and reason were generated in earlier rounds and then tested.\n"
        "Your job is to revise the hypothesis to improve future experiment performance.\n\n"
        f"Current side: {_side_label(side)}\n"
        f"{_scoring_explanation(side)}\n\n"
        f"{process_steps}"
        "Output constraints:\n"
        "- Keep hypothesis concise and testable (<= 30 words).\n"
        "- Keep reason concise and evidence-grounded.\n"
        "- Do not output markdown.\n"
        "- Output JSON only with this exact schema:\n"
        "{\n"
        '  "reason": "one concise improvement reason",\n'
        '  "hypothesis": "one revised concise hypothesis"\n'
        "}\n\n"
        "Current target hypothesis:\n"
        f"- hypothesis_index: {hypothesis_index}\n"
        f"- hypothesis: {current_hypothesis}\n"
        f"- reason: {current_reason}\n"
        f"- {current_score_name}: {current_score}\n\n"
        "Current-round memory evidence (compact):\n"
        f"{json.dumps(current_memory_evidence, ensure_ascii=False, indent=2)}\n\n"
        "Current-round execution evidence (compact):\n"
        f"{json.dumps(current_execution_evidence, ensure_ascii=False, indent=2)}\n\n"
        f"Historical memory evidence ({history_scope_text}):\n"
        f"{json.dumps(historical_evidence, ensure_ascii=False, indent=2)}"
    )
