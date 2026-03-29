from __future__ import annotations

import json
from typing import Any, Dict, Literal, Sequence

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


def _observation_description(side: SideType) -> str:
    if side == "input":
        return (
            "Observation schema (input side):\n"
            "- Includes the number of activation sentences and their activation examples.\n"
            "- Each activation example includes sentence text, non-zero activation values with their corresponding tokens, and the maximum activation value with its corresponding token.\n"
        )
    elif side == "output":
        return (
            "Observation schema (output side):\n"
            "- The SAE decoder feature vector is projected to tokens through the LLM unembedding layer.\n"
            "- The observation lists tokens with the largest and smallest token logits under that projection.\n"
        )
    else:
        raise ValueError(f"Unknown side type: {side}")


def build_system_prompt(side: SideType) -> str:
    return (
        "You are an expert interpretability researcher for sparse autoencoder (SAE) features.\n"
        "Your task is to infer hypotheses about one SAE feature based on observations.\n"
        "The observation is not raw model internals; it is preprocessed evidence collected from this feature.\n"
        f"You are currently generating hypotheses for the {_side_label(side)} behavior.\n"
        f"{_side_definition(side)}\n"
        "A hypothesis must be concise, concrete, and testable, and not too complex.\n"
        "Each hypothesis must be concise(at most 30 words).\n"
        "Do not output any extra commentary."
    )


def build_single_call_user_prompt(
    side: SideType,
    observation: Dict[str, Any],
    num_hypothesis: int,
) -> str:
    return (
        "Background:\n"
        "You are analyzing one SAE feature in an LLM.\n"
        f"{_observation_description(side)}\n"
        "The observation below has already been parsed into a compact dict.\n\n"
        "Task:\n"
        f"Generate exactly {num_hypothesis} distinct hypotheses for the {_side_label(side)} explanation.\n"
        "Each hypothesis must be clear, specific, and no longer than 30 words.\n\n"
        "Output format (JSON only):\n"
        "{\n"
        '  "hypotheses": ["hypothesis 1", "hypothesis 2"]\n'
        "}\n\n"
        "Observation:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}"
    )


def build_iterative_user_prompt(
    side: SideType,
    observation: Dict[str, Any],
    existing_hypotheses: Sequence[str],
    current_index: int,
    total_count: int,
) -> str:
    if existing_hypotheses:
        previous = json.dumps(list(existing_hypotheses), ensure_ascii=False, indent=2)
    else:
        previous = "[]"

    return (
        "Background:\n"
        "You are analyzing one SAE feature in an LLM.\n"
        f"{_observation_description(side)}\n"
        "The observation below has already been parsed into a compact dict.\n\n"
        "Task:\n"
        f"Generate hypothesis {current_index}/{total_count} for the {_side_label(side)} explanation.\n"
        "Your new hypothesis must stay grounded in the observation and be meaningfully different from previous hypotheses.\n"
        "If you really can't find a meaningfully different hypothesis, just output 'None'.\n"
        "Keep it concrete and at most 30 words.\n\n"
        "Output format (JSON only):\n"
        "{\n"
        '  "hypothesis": "one concise hypothesis"\n'
        '  "reason": "brief reason from the observation to the hypothesis"\n'
        "}\n\n"
        "Previous hypotheses:\n"
        f"{previous}\n\n"
        "Observation:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}"
    )
