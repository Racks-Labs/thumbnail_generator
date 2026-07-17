import json
from pathlib import Path

import pytest

from racks_thumbnail_generator.spec import (
    Anchor,
    CanvasSpec,
    Position,
    SpecError,
    ThumbnailSpec,
    load_spec,
    resolve_position,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "thumbnail_config.json"


def test_example_config_round_trip():
    spec = load_spec(EXAMPLE)
    assert spec.version == 1
    assert spec.canvas.width == 1080
    assert len(spec.generation.references) == 2
    assert all(Path(r.path).exists() for r in spec.generation.references)
    assert spec.elements[0].width == 0.18
    assert spec.title.accent.style == "underline"
    assert spec.title.accent.words == ["pobres"]
    assert spec.branding is None


def test_minimal_config_defaults():
    spec = ThumbnailSpec.model_validate({"title": {"text": "HOLA MUNDO"}})
    assert spec.canvas.width == 1080 and spec.canvas.height == 1920
    assert spec.generation.prompt is None
    assert spec.title.size == "auto"
    assert spec.title.accent.style == "underline"
    assert spec.elements == []


def test_invalid_config_reports_field(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"title": {"size": "huge"}}))
    with pytest.raises(SpecError, match="title.size"):
        load_spec(cfg)


def test_missing_asset_reports_path(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "title": {"text": "X"},
        "elements": [{"path": "nope.png", "position": {"x": 0, "y": 0}}],
    }))
    with pytest.raises(SpecError, match="nope.png"):
        load_spec(cfg)


def test_relative_paths_resolved_against_config_dir(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"fake")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "title": {"text": "X"},
        "elements": [{"path": "logo.png"}],
    }))
    spec = load_spec(cfg)
    assert Path(spec.elements[0].path) == tmp_path / "logo.png"


@pytest.mark.parametrize("anchor,expected", [
    (Anchor.TOP_LEFT, (100, 200)),
    (Anchor.TOP_RIGHT, (60, 200)),
    (Anchor.BOTTOM_LEFT, (100, 180)),
    (Anchor.BOTTOM_RIGHT, (60, 180)),
    (Anchor.CENTER, (80, 190)),
    (Anchor.TOP_CENTER, (80, 200)),
    (Anchor.BOTTOM_CENTER, (80, 180)),
    (Anchor.CENTER_LEFT, (100, 190)),
    (Anchor.CENTER_RIGHT, (60, 190)),
])
def test_resolve_position_anchors(anchor, expected):
    # canvas 1000x1000, object 40x20, target point (100, 200) in px
    pos = Position(x=100, y=200, unit="px", anchor=anchor)
    assert resolve_position(pos, (1000, 1000), (40, 20)) == expected


def test_resolve_position_fraction():
    pos = Position(x=0.5, y=1.0, anchor=Anchor.BOTTOM_CENTER)
    assert resolve_position(pos, (1000, 2000), (100, 50)) == (450, 1950)


@pytest.mark.parametrize("w,h,expected", [
    (1080, 1920, "9:16"),
    (1920, 1080, "16:9"),
    (1000, 1000, "1:1"),
    (1080, 1350, "4:5"),
])
def test_canvas_aspect_ratio(w, h, expected):
    assert CanvasSpec(width=w, height=h).aspect_ratio() == expected


def test_json_schema_exports():
    schema = ThumbnailSpec.model_json_schema()
    assert "properties" in schema
    for key in ("canvas", "generation", "elements", "title", "branding"):
        assert key in schema["properties"]
