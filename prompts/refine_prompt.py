from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

SideType = Literal["input", "output"]
HistoryScope = Literal["same_hypothesis", "all_hypotheses"]
InputRefinementMode = Literal["activation_repair", "activation_expand", "expansion_adjust"]


def _side_label(side: SideType) -> str:
    return "input-side activation" if side == "input" else "output-side intervention"


def _scoring_explanation(side: SideType) -> str:
    if side == "input":
        return (
            "Score definition:\n"
            "- score_non_zero_rate = non_zero_count / total_sentences.\n"
            "- Higher is better.\n"
            "- Input-side test sentences should semantically match the hypothesis and activate the SAE feature.\n"
            "- Current-round evidence also includes test_type: activation or expansion."
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
        "You must reason from score first, then inspect evidence patterns and limitations in the previous reason/hypothesis.\n"
        "Return only valid JSON with no extra text."
    )


def _input_mode_instructions(mode: InputRefinementMode) -> str:
    if mode == "activation_repair":
        return (
            "Current input-side mode: activation test.\n"
            "Goal:\n"
            "1) Keep successful activation patterns.\n"
            "2) Exclude semantics reflected by failed activation examples.\n"
            "3) Improve activation hit rate in the next round."
        )
    if mode == "activation_expand":
        return (
            "Current input-side mode: activation test with full score.\n"
            "Goal:\n"
            "1) Treat current hypothesis as a proven subset of the true feature set.\n"
            "2) Expand the hypothesis by abstracting one semantic level upward from the observed positives.\n"
            "3) Use successful examples as anchor cases, not as the full semantic boundary.\n"
            "4) Consider adjacent sibling cases that plausibly belong to the same feature.\n"
            "5) Use historical trajectory to avoid reusing earlier failed meanings.\n"
            "6) Keep the expanded hypothesis concise, precise, and testable.\n"
            "7) Do not lightly infer extra context restrictions from a small set of successful examples.\n"
            "8) Only add context-specific conditions if the evidence clearly suggests the feature depends on that context, rather than on the core semantics alone."
        )
    return (
        "Current input-side mode: expansion test.\n"
        "Goal:\n"
        "1) Include semantics of expansion examples that activated.\n"
        "2) Exclude semantics of expansion examples that did not activate.\n"
        # "3) Keep the revised hypothesis broader than the provided pre-expansion hypothesis."
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
    input_refinement_mode: Optional[InputRefinementMode] = None,
    input_pre_expansion_hypothesis: str = "",
    expand_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    history_scope_text = (
        "same hypothesis across previous rounds"
        if history_scope == "same_hypothesis"
        else "all hypotheses on this side across previous rounds"
    )

    input_mode_block = ""
    if side == "input":
        resolved_mode: InputRefinementMode = input_refinement_mode or "activation_repair"
        input_mode_block = f"{_input_mode_instructions(resolved_mode)}\n\n"
        if resolved_mode == "expansion_adjust":
            pre = input_pre_expansion_hypothesis.strip() or "(not provided)"
            input_mode_block += (
                "Pre-expansion hypothesis (must stay narrower than revised output):\n"
                f"{pre}\n\n"
            )
    else:
        resolved_mode = None

    if side == "input" and resolved_mode == "activation_expand":
        process_steps = (
            "Required process:\n"
            "1) Read the score and infer that the current hypothesis is a successful subset.\n"
            "2) Start from the provided expansion analysis instead of searching for the narrowest common denominator.\n"
            "3) Preserve the anchor cases covered by the current hypothesis.\n"
            "4) Extend toward the adjacent sibling cases identified in the analysis.\n"
            "5) Produce one broader revised hypothesis and one concise evidence-grounded reason.\n\n"
        )
    else:
        process_steps = "Required process:\n1) Read the score and infer current hypothesis quality.\n2) Analyze mismatch patterns from evidence.\n"
        if history_scope == "same_hypothesis":
            process_steps += "3) Use same-index history trajectory.\n4) Produce a more reliable revised hypothesis.\n5) Explain why it should perform better.\n\n"
        else:
            process_steps += (
                "3) Separate own trajectory and peer hypotheses in history.\n"
                "4) Reuse transferable success/failure patterns, but more concentrate on your own trajectory.\n"
                "5) Produce a more reliable revised hypothesis.\n"
                "6) Explain why it should perform better.\n\n"
            )

    task_statement = "Your job is to revise the hypothesis to improve future experiment performance."
    if side == "input" and resolved_mode == "activation_expand":
        task_statement = (
            "Your job is to expand the hypothesis to a broader latent semantic class while preserving the confirmed positive cases."
        )

    expand_analysis_block = ""
    if expand_analysis:
        expand_analysis_block = (
            "Expansion analysis to use:\n"
            f"{json.dumps(expand_analysis, ensure_ascii=False, indent=2)}\n\n"
        )

    return (
        "Background:\n"
        "You are improving one SAE feature hypothesis in an iterative workflow.\n"
        "The previous hypothesis and reason were generated in earlier rounds and then tested.\n"
        f"{task_statement}\n\n"
        f"Current side: {_side_label(side)}\n"
        f"{_scoring_explanation(side)}\n\n"
        f"{input_mode_block}"
        f"{process_steps}"
        "Output constraints:\n"
        "- Keep hypothesis concise and testable (<= 40 words).\n"
        "- Keep reason concise and evidence-grounded.\n"
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
        f"{expand_analysis_block}"
        "Current-round memory evidence (compact):\n"
        f"{json.dumps(current_memory_evidence, ensure_ascii=False, indent=2)}\n\n"
        "Current-round execution evidence (compact):\n"
        f"{json.dumps(current_execution_evidence, ensure_ascii=False, indent=2)}\n\n"
        f"Historical memory evidence ({history_scope_text}):\n"
        f"{json.dumps(historical_evidence, ensure_ascii=False, indent=2)}"
    )


def build_activation_expand_analysis_prompt(
    *,
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
    return (
        "Background:\n"
        "You are analyzing how to broaden one successful input-side SAE feature hypothesis.\n"
        "Do not search for the narrowest shared sense of the observed positives.\n"
        "Use the successful examples as anchor cases, then abstract one semantic level upward.\n\n"
        "Task:\n"
        "1) Extract the preserved core semantics that the current hypothesis already captures.\n"
        "2) Identify adjacent sibling cases that are not literally listed in the current hypothesis but plausibly belong to the same feature.\n"
        "3) Explain why the broader latent semantic class still covers the current positives.\n"
        "4) Treat incidental context overlap in the examples as weak evidence unless the feature clearly appears context-dependent.\n\n"
        "Output constraints:\n"
        "- preserved_core: 2-4 short bullet-like strings.\n"
        "- new_sibling_cases: 2-4 short bullet-like strings.\n"
        "- broader_scope_explanation: one concise paragraph.\n"
        "- Output JSON only with this exact schema:\n"
        "{\n"
        '  "preserved_core": ["item 1", "item 2"],\n'
        '  "new_sibling_cases": ["item 1", "item 2"],\n'
        '  "broader_scope_explanation": "one concise explanation"\n'
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
