from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_documents_exist_with_required_sections() -> None:
    report = (ROOT / "reports" / "report.md").read_text(encoding="utf-8")
    summary = (ROOT / "reports" / "summary.md").read_text(encoding="utf-8")

    for section in [
        "Abstract",
        "Introduction",
        "Data",
        "Methods",
        "Experimental Design",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    ]:
        assert f"## {section}" in report
    assert "naive base rate" in report
    assert "naive base rate" in summary
