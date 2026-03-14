from __future__ import annotations

from typing import Sequence


def build_input_activation_prompt(*, hypothesis: str, designed_sentences: Sequence[str]) -> str:
    lines = []
    lines.append("Task background:")
    lines.append("You are validating an input-side SAE hypothesis.")
    lines.append("Each sentence should be semantically aligned with the hypothesis and likely to activate the target SAE feature.")
    lines.append("")
    lines.append("Hypothesis:")
    lines.append(hypothesis.strip())
    lines.append("")
    lines.append("Candidate activation sentences:")
    for index, sentence in enumerate(designed_sentences, start=1):
        lines.append(f"{index}. {sentence.strip()}")
    lines.append("")
    lines.append("Evaluation target:")
    lines.append("Measure feature activation for each sentence and compute non-zero activation rate.")
    return "\n".join(lines)


def build_output_judge_system_prompt(num_sets: int) -> str:
    if num_sets < 2:
        raise ValueError("num_sets must be at least 2.")

    return (
        "You are an expert evaluator for sparse autoencoder (SAE) feature interventions in language models.\n"
        "Task background:\n"
        "- A hypothesis describes what one SAE feature represents.\n"
        "- For each candidate set, model completions were produced after steering one SAE feature.\n"
        "- The prompts were designed to activate the target feature before intervention.\n"
        "- Exactly one candidate set is from the target feature intervention. The others are controls.\n\n"
        "Evaluation goal:\n"
        "Pick the set that is most consistent with the hypothesis semantics.\n"
        "Focus on lexical and semantic alignment with the hypothesis.\n"
        "Do not score by fluency or writing quality.\n"
        f"You must choose exactly one set number from 1 to {num_sets}.\n\n"
        "Output format requirements:\n"
        "- Line 1: one concise reason.\n"
        f"- Line 2: only one integer in [1, {num_sets}] with no extra text."
    )


def build_output_judge_user_prompt(*, explanation: str, option_sets: Sequence[str]) -> str:
    if not option_sets:
        raise ValueError("option_sets must not be empty.")

    lines = []
    lines.append("Hypothesis:")
    lines.append(explanation.strip())
    lines.append("")
    lines.append("Candidate completion sets:")
    for index, option_text in enumerate(option_sets, start=1):
        lines.append(f"# Set {index}")
        lines.append(option_text.strip())
        lines.append("")

    lines.append("Return exactly two lines:")
    lines.append("Line 1: one concise reason.")
    lines.append(f"Line 2: only one integer from 1 to {len(option_sets)}.")
    return "\n".join(lines)
