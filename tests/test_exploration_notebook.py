import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "exploration.ipynb"


def test_exploration_notebook_has_required_sections_and_valid_code() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    markdown = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
    assert all(section in markdown for section in ["Data quality", "Feature distributions", "Missingness", "Baseline model"])

    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
