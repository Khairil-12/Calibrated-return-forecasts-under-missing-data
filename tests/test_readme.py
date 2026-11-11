from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_exists_and_contains_required_sections() -> None:
    readme_path = ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist in repository root"
    content = readme_path.read_text(encoding="utf-8")

    assert "# CalibMiss" in content
    for section in [
        "Research Questions",
        "Pipeline Architecture",
        "Setup and Installation",
        "Running the Pipeline",
        "Experimental Design",
        "Testing and Code Quality",
        "Limitations and Scope",
    ]:
        assert f"## {section}" in content

    for target in ["make all", "make data", "make features", "make experiment", "make figures"]:
        assert target in content

    assert "trading strategy" in content.lower()
