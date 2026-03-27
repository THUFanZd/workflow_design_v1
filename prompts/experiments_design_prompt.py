from __future__ import annotations

from typing import Literal

SideType = Literal["input", "output"]
InputTestType = Literal["activation", "expansion"]


def _side_label(side: SideType) -> str:
    return "input-side" if side == "input" else "output-side"


def build_system_prompt(side: SideType, *, input_test_type: InputTestType = "activation") -> str:
    if side == "input":
        return build_input_system_prompt(test_type=input_test_type)
    return build_output_system_prompt()


def build_input_system_prompt(*, test_type: InputTestType) -> str:
    if test_type == "activation":
        task_desc = "activation test cases"
    else:
        task_desc = "expansion test cases"

    return (
        "You are an expert interpretability researcher for sparse autoencoder (SAE) features.\n"
        "You design validation experiments for hypotheses about one SAE feature.\n"
        f"The current task is to generate input-side {task_desc}.\n"
        "Follow the user's required JSON format exactly.\n"
        "Do not output chain-of-thought or extra commentary."
    )


def build_input_user_prompt(
    *,
    hypothesis: str,
    num_sentences: int,
    test_type: InputTestType,
    previous_hypothesis: str = "",
) -> str:
    if test_type == "activation":
        return (
            "Background:\n"
            "An SAE feature is hypothesized to activate for a specific semantic pattern.\n"
            "You need to generate test sentences that are likely to activate this feature.\n\n"
            "Task:\n"
            f"Given the hypothesis below, generate exactly {num_sentences} English sentences.\n"
            "Each sentence must directly reflect the literal meaning of the hypothesis and be likely "
            "to trigger the corresponding SAE feature activation.\n"
            "Keep each sentence natural, clear, and under 30 words.\n"
            "Avoid near-duplicates.\n\n"
            "Output format (JSON only):\n"
            "{\n"
            '  "sentences": ["sentence 1", "sentence 2"]\n'
            "}\n\n"
            f"Hypothesis:\n{hypothesis}"
        )

    previous = previous_hypothesis.strip() or "(none)"
    return (
        "Background:\n"
        "The hypothesis was refined from a previous version.\n"
        "You must design expansion test cases that focus on newly added semantics only.\n\n"
        "Task:\n"
        f"Generate exactly {num_sentences} English sentences for the semantic region:\n"
        "(current hypothesis semantics) minus (previous hypothesis semantics).\n"
        "Each sentence should target the new expanded meaning and be likely to activate the feature\n"
        "if the expansion is valid. Avoid sentences that are already clearly covered by the previous hypothesis.\n"
        "Keep each sentence natural, clear, and under 30 words.\n"
        "Avoid near-duplicates.\n\n"
        "Output format (JSON only):\n"
        "{\n"
        '  "sentences": ["sentence 1", "sentence 2"]\n'
        "}\n\n"
        "Previous hypothesis (before refinement):\n"
        f"{previous}\n\n"
        "Current hypothesis (after refinement):\n"
        f"{hypothesis}"
    )


def build_output_system_prompt() -> str:
    return (
        "You are preparing output-side intervention validation placeholders for an SAE feature.\n"
        "Follow the required JSON format exactly."
    )


def build_output_user_prompt(*, hypothesis: str) -> str:
    return (
        "Background:\n"
        "You are preparing output-side intervention validation placeholders for an SAE feature.\n\n"
        "Task:\n"
        "Return the following list exactly as JSON.\n\n"
        "Output format (JSON only):\n"
        '{\n  "sentences": ["The explanation is simple:", "I think", "We"]\n}\n\n'
        f"Hypothesis:\n{hypothesis}"
    )


def build_user_prompt(
    *,
    side: SideType,
    hypothesis: str,
    num_sentences: int,
    input_test_type: InputTestType = "activation",
    previous_hypothesis: str = "",
) -> str:
    if side == "input":
        return build_input_user_prompt(
            hypothesis=hypothesis,
            num_sentences=num_sentences,
            test_type=input_test_type,
            previous_hypothesis=previous_hypothesis,
        )
    return build_output_user_prompt(hypothesis=hypothesis)
