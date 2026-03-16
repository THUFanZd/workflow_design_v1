from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow importing project modules when this script is launched from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_design import OUTPUT_SIDE_PLACEHOLDER
from function import DEFAULT_CANONICAL_MAP_PATH, build_default_sae_path
from model_with_sae import ModelWithSAEModule

from intervention_blind_score import KL_DIV_VALUES

SAE_RELEASE_BY_NAME: Dict[str, str] = {
    "gemmascope-res": "gemma-scope-2b-pt-res",
}


def _resolve_sae(
    *,
    sae_name: str,
    sae_release: Optional[str],
    layer_id: str,
    width: str,
    average_l0: Optional[str],
    canonical_map_path: Path,
) -> Tuple[str, str, str]:
    release = (sae_release or SAE_RELEASE_BY_NAME.get(sae_name) or sae_name).strip()
    sae_uri, resolved_average_l0 = build_default_sae_path(
        layer_id=layer_id,
        width=width,
        release=release,
        average_l0=average_l0,
        canonical_map_path=canonical_map_path,
    )
    return sae_uri, release, resolved_average_l0


def _coerce_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _format_intervention_output_txt(
    *,
    prompts: Sequence[str],
    run_records: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    for record in run_records:
        kl = float(record["target_kl"])
        completions = list(record["completions"])
        if len(completions) != len(prompts):
            raise ValueError(
                f"KL={kl} completion size mismatch: got {len(completions)}, expected {len(prompts)}."
            )
        kl_tag = f"{kl:+g}"
        for idx, prompt in enumerate(prompts):
            completion = str(completions[idx]).replace("\n", "\\n").replace("\r", "\\r")
            lines.append(f"<{kl_tag}>'{prompt}': '{completion}'")
    return "\n".join(lines)


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


def _write_summary_markdown(
    *,
    path: Path,
    sae_name: str,
    sae_release: str,
    sae_uri: str,
    layer_id: str,
    width: str,
    feature_id: int,
    prompts: Sequence[str],
    run_records: Sequence[Dict[str, Any]],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Feature Intervention Outputs")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- generated_at: {now}")
    lines.append(f"- sae_name: {sae_name}")
    lines.append(f"- sae_release: {sae_release}")
    lines.append(f"- sae_uri: `{sae_uri}`")
    lines.append(f"- layer_id: {layer_id}")
    lines.append(f"- width: {width}")
    lines.append(f"- feature_id: {feature_id}")
    lines.append("")
    lines.append("## Input Prompts")
    for prompt in prompts:
        lines.append(f"- {prompt}")
    lines.append("")
    lines.append("## Intervention Details")
    lines.append("| target_kl (intervention strength) | actual_kl | input_prompt | output_completion |")
    lines.append("| ---: | ---: | --- | --- |")
    for record in run_records:
        target_kl = float(record["target_kl"])
        completions = list(record["completions"])
        actual_kl_values = list(record.get("actual_kl_values", []))
        for idx, prompt in enumerate(prompts):
            completion = _escape_md_cell(str(completions[idx]))
            actual_kl = ""
            if idx < len(actual_kl_values):
                actual_kl = f"{float(actual_kl_values[idx]):.6f}"
            lines.append(
                f"| {target_kl:+g} | {actual_kl} | {_escape_md_cell(prompt)} | {completion} |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run output-side SAE feature intervention and export results into md/txt files "
            "under explanation_quality_evaluation/output-side-evaluation/outputs/{SAE}/layer-{layer}/feature-{id}."
        )
    )
    parser.add_argument("--layer-id", required=True, help="SAE layer id")
    parser.add_argument("--width", default="16k", help="SAE width, e.g. 16k")
    parser.add_argument("--feature-id", required=True, type=int, help="Feature id to intervene")
    parser.add_argument(
        "--sae-name",
        default="gemmascope-res",
        help="SAE family name for output path and default release lookup",
    )
    parser.add_argument(
        "--sae-release",
        default=None,
        help="Optional explicit SAE release override, e.g. gemma-scope-2b-pt-res",
    )
    parser.add_argument(
        "--sae-average-l0",
        default=None,
        help="Optional average_l0 suffix override; if omitted, resolve from canonical_map.txt",
    )
    parser.add_argument(
        "--sae-canonical-map",
        default=str(PROJECT_ROOT / DEFAULT_CANONICAL_MAP_PATH),
        help="Path to canonical_map.txt for resolving canonical average_l0",
    )
    parser.add_argument(
        "--model-checkpoint-path",
        default="google/gemma-2-2b",
        help="Base model checkpoint path/name",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--kl-values",
        type=float,
        nargs="*",
        default=[float(x) for x in KL_DIV_VALUES],
        help="Target KL values used as intervention strengths",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "explanation_quality_evaluation" / "output-side-evaluation" / "outputs"),
        help="Root output directory",
    )
    parser.add_argument("--txt-filename", default="intervention_output.txt")
    parser.add_argument("--md-filename", default="intervention_summary.md")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    layer_id = str(args.layer_id)
    width = str(args.width)
    feature_id = int(args.feature_id)
    prompts = list(OUTPUT_SIDE_PLACEHOLDER)

    sae_uri, sae_release, resolved_average_l0 = _resolve_sae(
        sae_name=str(args.sae_name),
        sae_release=args.sae_release,
        layer_id=layer_id,
        width=width,
        average_l0=args.sae_average_l0,
        canonical_map_path=Path(args.sae_canonical_map),
    )

    module = ModelWithSAEModule(
        llm_name=str(args.model_checkpoint_path),
        sae_path=sae_uri,
        sae_layer=int(layer_id),
        feature_index=feature_id,
        device=str(args.device),
    )

    run_records: List[Dict[str, Any]] = []
    for target_kl_raw in args.kl_values:
        target_kl = float(target_kl_raw)
        output = module.generate_steered_completions(
            prompts=prompts,
            feature_index=feature_id,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            target_kl=target_kl,
        )
        completions = output.get("steered_completion")
        if not isinstance(completions, list):
            raise RuntimeError(f"Unexpected steered output format for KL={target_kl}: {output}")

        run_records.append(
            {
                "target_kl": target_kl,
                "completions": [str(x) for x in completions],
                "actual_kl_values": _coerce_float_list(output.get("kl_values")),
                "clamp_values": _coerce_float_list(output.get("clamp_values")),
            }
        )

    intervention_output_txt = _format_intervention_output_txt(
        prompts=prompts,
        run_records=run_records,
    )

    target_dir = (
        Path(args.output_root)
        / str(args.sae_name)
        / f"layer-{layer_id}"
        / f"feature-{feature_id}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    txt_path = target_dir / str(args.txt_filename)
    md_path = target_dir / str(args.md_filename)

    txt_path.write_text(intervention_output_txt + "\n", encoding="utf-8")
    _write_summary_markdown(
        path=md_path,
        sae_name=str(args.sae_name),
        sae_release=sae_release,
        sae_uri=sae_uri,
        layer_id=layer_id,
        width=width,
        feature_id=feature_id,
        prompts=prompts,
        run_records=run_records,
    )

    summary = {
        "output_dir": str(target_dir),
        "txt_file": str(txt_path),
        "md_file": str(md_path),
        "sae_name": str(args.sae_name),
        "sae_release": sae_release,
        "sae_uri": sae_uri,
        "resolved_average_l0": resolved_average_l0,
        "layer_id": layer_id,
        "width": width,
        "feature_id": feature_id,
        "prompts": prompts,
        "kl_values": [float(x["target_kl"]) for x in run_records],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
