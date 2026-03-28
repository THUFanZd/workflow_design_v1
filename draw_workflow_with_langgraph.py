from __future__ import annotations

from pathlib import Path
from typing import TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: langgraph\n"
        "Install with: pip install langgraph langchain-core"
    ) from exc


class WorkflowState(TypedDict, total=False):
    current_round: int
    converged: bool


def collect_observation(state: WorkflowState) -> WorkflowState:
    return state


def generate_initial_hypotheses(state: WorkflowState) -> WorkflowState:
    return state


def design_baseline_experiments(state: WorkflowState) -> WorkflowState:
    return state


def execute_baseline_experiments(state: WorkflowState) -> WorkflowState:
    return state


def build_baseline_memory(state: WorkflowState) -> WorkflowState:
    return state


def refine_hypotheses(state: WorkflowState) -> WorkflowState:
    return state


def merge_hypotheses(state: WorkflowState) -> WorkflowState:
    return state


def prepare_round_hypotheses(state: WorkflowState) -> WorkflowState:
    return state


def design_experiments(state: WorkflowState) -> WorkflowState:
    return state


def execute_experiments(state: WorkflowState) -> WorkflowState:
    return state


def build_memory(state: WorkflowState) -> WorkflowState:
    return state


def advance_round(state: WorkflowState) -> WorkflowState:
    return state


def finalize(state: WorkflowState) -> WorkflowState:
    return state


def route_after_baseline(state: WorkflowState) -> str:
    return "finalize" if state.get("converged", False) else "refine_hypotheses"


def route_after_memory(state: WorkflowState) -> str:
    return "finalize" if state.get("converged", False) else "advance_round"


def build_workflow_app():
    graph = StateGraph(WorkflowState)

    graph.add_node("collect_observation", collect_observation)
    graph.add_node("generate_initial_hypotheses", generate_initial_hypotheses)
    graph.add_node("design_baseline_experiments", design_baseline_experiments)
    graph.add_node("execute_baseline_experiments", execute_baseline_experiments)
    graph.add_node("build_baseline_memory", build_baseline_memory)
    graph.add_node("refine_hypotheses", refine_hypotheses)
    graph.add_node("merge_hypotheses", merge_hypotheses)
    graph.add_node("prepare_round_hypotheses", prepare_round_hypotheses)
    graph.add_node("design_experiments", design_experiments)
    graph.add_node("execute_experiments", execute_experiments)
    graph.add_node("build_memory", build_memory)
    graph.add_node("advance_round", advance_round)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "collect_observation")
    graph.add_edge("collect_observation", "generate_initial_hypotheses")
    graph.add_edge("generate_initial_hypotheses", "design_baseline_experiments")
    graph.add_edge("design_baseline_experiments", "execute_baseline_experiments")
    graph.add_edge("execute_baseline_experiments", "build_baseline_memory")
    graph.add_conditional_edges(
        "build_baseline_memory",
        route_after_baseline,
        {
            "refine_hypotheses": "refine_hypotheses",
            "finalize": "finalize",
        },
    )
    graph.add_edge("refine_hypotheses", "merge_hypotheses")
    graph.add_edge("merge_hypotheses", "prepare_round_hypotheses")
    graph.add_edge("prepare_round_hypotheses", "design_experiments")
    graph.add_edge("design_experiments", "execute_experiments")
    graph.add_edge("execute_experiments", "build_memory")
    graph.add_conditional_edges(
        "build_memory",
        route_after_memory,
        {
            "advance_round": "advance_round",
            "finalize": "finalize",
        },
    )
    graph.add_edge("advance_round", "refine_hypotheses")
    graph.add_edge("finalize", END)

    return graph.compile()


def export_diagram(output_dir: str = "logs") -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    app = build_workflow_app()
    drawable = app.get_graph()

    mermaid_text = drawable.draw_mermaid()
    mermaid_path = target_dir / "workflow_langgraph.mmd"
    mermaid_path.write_text(mermaid_text, encoding="utf-8")
    print(f"Saved Mermaid diagram: {mermaid_path}")

    png_path = target_dir / "workflow_langgraph.png"
    try:
        png_bytes = drawable.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"Saved PNG diagram: {png_path}")
    except Exception as exc:
        print(f"PNG export skipped: {exc}")


if __name__ == "__main__":
    export_diagram()
