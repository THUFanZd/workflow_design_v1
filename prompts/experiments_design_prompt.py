from __future__ import annotations

from typing import Literal

SideType = Literal["input", "output"]


def _side_label(side: SideType) -> str:
    return "input-side activation" if side == "input" else "output-side intervention"


def _side_definition(side: SideType) -> str:
    if side == "input":
        return (
            "Definition: input-side means the hypothesis describes what kinds of input sentences, "
            "when fed into the model, activate the target SAE feature."
        )
    return (
        "Definition: output-side means the hypothesis describes how the model's output changes "
        "after the target SAE feature value is intervened on."
    )


def build_system_prompt(side: SideType) -> str:
    return (
        "You are an expert interpretability researcher for sparse autoencoder (SAE) features.\n"
        "You design validation experiments for hypotheses about one SAE feature.\n"
        f"The current task is for {_side_label(side)} hypotheses.\n"
        f"{_side_definition(side)}\n"
        "Follow the user's required output format exactly.\n"
        "Do not output extra commentary."
    )


def build_user_prompt(*, side: SideType, hypothesis: str, num_sentences: int) -> str:
    if side == "input":
        return (
            "Background:\n"
            "An SAE feature is hypothesized to activate for a specific semantic pattern.\n"
            "You need to generate test sentences that are likely to activate this feature.\n\n"
            "Task:\n"
            f"Given the hypothesis below, generate exactly {num_sentences} sentences.\n"
            "Each sentence must directly reflect the literal meaning of the hypothesis and be likely "
            "to trigger the corresponding SAE feature activation.\n"
            "While staying faithful to the hypothesis, vary the surrounding context as much as reasonably possible.\n"
            "Use diverse syntax, discourse settings, and nearby content instead of repeating one narrow scenario.\n"
            "Keep each sentence natural, clear, and under 60 words.\n"
            "Avoid near-duplicates.\n\n"
            "Output format (JSON only):\n"
            "{\n"
            '  "sentences": ["sentence 1", "sentence 2"]\n'
            "}\n\n"
            f"Hypothesis:\n{hypothesis}"
        )

    return (
        "Background:\n"
        "You are preparing output-side intervention validation placeholders for an SAE feature.\n\n"
        "Task:\n"
        "Return the following list exactly as JSON.\n\n"
        "Output format (JSON only):\n"
        '{\n  "sentences": ["The explanation is simple:", "I think", "We"]\n}\n\n'
        f"Hypothesis:\n{hypothesis}"
    )
