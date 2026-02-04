from pathlib import Path


def test_temporal_field_viewer_html_present() -> None:
    html = Path("temporal_field_viewer.html").read_text(encoding="utf-8")
    assert "temporal-field-canvas" in html
    assert "getContext(\"webgl\")" in html
