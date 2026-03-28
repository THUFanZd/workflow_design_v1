from __future__ import annotations

from pathlib import Path

from workflow_runner import LANGGRAPH_AVAILABLE, _build_langgraph_app


def main() -> int:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "langgraph is not installed. Please install langgraph first, then rerun this script."
        )

    app = _build_langgraph_app()
    drawable = app.get_graph()

    mermaid_text = drawable.draw_mermaid()
    mermaid_path = logs_dir / "workflow_diagram.mmd"
    mermaid_path.write_text(mermaid_text, encoding="utf-8")
    print(f"Saved Mermaid diagram: {mermaid_path}")

    png_path = logs_dir / "workflow_diagram.png"
    try:
        png_bytes = drawable.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"Saved PNG diagram: {png_path}")
    except Exception as exc:
        print(f"PNG export skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
