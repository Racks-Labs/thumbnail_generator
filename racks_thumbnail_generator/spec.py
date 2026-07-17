"""JSON config schema for thumbnail generation — the contract with the frontend.

A frontend app builds a JSON document matching `ThumbnailSpec` and this service
renders it: AI-generated scene (with reference images) + deterministic Pillow
overlays (graphic elements + title) at exact positions.

Export the machine-readable schema for the frontend with:
    python -c "import json; from racks_thumbnail_generator.spec import ThumbnailSpec; \\
               print(json.dumps(ThumbnailSpec.model_json_schema(), indent=2))"

Breaking changes to this schema must bump `ThumbnailSpec.version`.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class Anchor(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class Position(BaseModel):
    """A point on the canvas. `fraction` units are 0-1 relative to canvas size,
    so the same JSON works at any resolution. (x, y) is where the `anchor`
    of the object lands: e.g. anchor=bottom_left + x=0.05/y=0.95 pins the
    object's bottom-left corner near the canvas' bottom-left."""

    x: float = 0.5
    y: float = 0.5
    unit: Literal["fraction", "px"] = "fraction"
    anchor: Anchor = Anchor.TOP_LEFT


class CanvasSpec(BaseModel):
    width: int = Field(1080, gt=0)
    height: int = Field(1920, gt=0)

    def aspect_ratio(self) -> str:
        """Closest aspect ratio supported by the Gemini image API."""
        supported = {
            "1:1": 1.0, "2:3": 2 / 3, "3:2": 3 / 2, "3:4": 3 / 4, "4:3": 4 / 3,
            "4:5": 4 / 5, "5:4": 5 / 4, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9,
        }
        ratio = self.width / self.height
        return min(supported, key=lambda k: abs(supported[k] - ratio))


class ReferenceImageSpec(BaseModel):
    """An image passed to the AI as generation reference (background, person
    already cut out / on a simple background, product shot, ...). `role` is a
    free-text description injected into the prompt so the model knows how to
    use each image."""

    path: str
    role: str | None = None


class GenerationSpec(BaseModel):
    """How the AI scene is generated. If `prompt` is null the prompt is built
    from the video/script transcript via topic extraction (legacy flow)."""

    prompt: str | None = None
    template: str = "default"
    references: list[ReferenceImageSpec] = Field(default_factory=list)


class ElementSpec(BaseModel):
    """A graphic element (logo, badge, frame, ...) pasted deterministically with
    Pillow AFTER the AI image is generated — always at the exact position given.
    `width`/`height` are fractions of the canvas; give one to keep aspect ratio,
    both to stretch, none for the image's native pixel size."""

    path: str
    position: Position = Field(default_factory=Position)
    width: float | None = Field(None, gt=0)
    height: float | None = Field(None, gt=0)
    opacity: float = Field(1.0, ge=0.0, le=1.0)


class FontSpec(BaseModel):
    """Resolves to `<fonts_dir>/<family>-<weight>.ttf` (bundled Inter weights:
    Black, Bold, SemiBold, Medium, Regular, Light) unless `file` points to a
    custom .ttf, which takes precedence."""

    family: str = "Inter"
    weight: str = "Black"
    file: str | None = None


class AccentSpec(BaseModel):
    """Per-word emphasis in the title. `words` lists the words to emphasize
    (accent-insensitive match); empty list means no accent. `thickness` and
    `offset` are fractions of the font size."""

    style: Literal["underline", "highlight", "none"] = "underline"
    words: list[str] = Field(default_factory=list)
    color: str = "#BE190F"
    thickness: float = Field(0.08, gt=0)
    offset: float = 0.06
    text_color: str | None = None  # highlight style only; None → auto black/white


class ShadowSpec(BaseModel):
    """Soft drop shadow behind the title. Offsets/blur are fractions of the
    font size."""

    enabled: bool = True
    color: str = "#000000"
    opacity: float = Field(0.63, ge=0.0, le=1.0)
    blur: float = 0.06
    offset_x: float = 0.025
    offset_y: float = 0.04


class TextBoxSpec(BaseModel):
    """Region (fractions of the canvas) the title is laid out in. (x, y) is the
    `anchor` corner of the box."""

    x: float = 0.05
    y: float = 0.78
    width: float = Field(0.9, gt=0)
    height: float = Field(0.22, gt=0)
    anchor: Anchor = Anchor.BOTTOM_LEFT
    align: Literal["left", "center", "right"] = "left"


class TitleSpec(BaseModel):
    """Title overlay. If `text` is null the headline comes from the transcript
    (topic extraction), which then requires a video/script input."""

    text: str | None = None
    uppercase: bool = True
    box: TextBoxSpec = Field(default_factory=TextBoxSpec)
    font: FontSpec = Field(default_factory=FontSpec)
    size: int | Literal["auto"] = "auto"
    min_size: int = Field(22, gt=0)
    color: str = "#FFFFFF"
    line_spacing: float = 0.02
    accent: AccentSpec = Field(default_factory=AccentSpec)
    shadow: ShadowSpec = Field(default_factory=ShadowSpec)


class BrandingSpec(BaseModel):
    """Small branding text. Set the whole object to null to disable."""

    text: str = "RACKS"
    position: Position = Field(
        default_factory=lambda: Position(x=0.95, y=0.17, anchor=Anchor.TOP_RIGHT)
    )
    font: FontSpec = Field(default_factory=lambda: FontSpec(weight="Bold"))
    size: float = Field(0.022, gt=0)  # fraction of canvas height
    color: str = "#FFFFFF"


class ThumbnailSpec(BaseModel):
    version: Literal[1] = 1
    canvas: CanvasSpec = Field(default_factory=CanvasSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    elements: list[ElementSpec] = Field(default_factory=list)
    title: TitleSpec = Field(default_factory=TitleSpec)
    branding: BrandingSpec | None = None
    output: str | None = None


class SpecError(ValueError):
    """User-facing error loading/validating a thumbnail config JSON."""


def _resolve_asset(path_str: str, base_dir: Path, what: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise SpecError(f"{what} not found: {path}")
    return str(path)


def load_spec(config_path: Path) -> ThumbnailSpec:
    """Load + validate a JSON config. Asset paths (references, elements, custom
    fonts, output) are resolved relative to the JSON file's directory."""
    if not config_path.exists():
        raise SpecError(f"Config file not found: {config_path}")

    try:
        spec = ThumbnailSpec.model_validate_json(config_path.read_text(encoding="utf-8"))
    except ValidationError as e:
        lines = [f"Invalid thumbnail config: {config_path}"]
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            lines.append(f"  - {loc}: {err['msg']}")
        raise SpecError("\n".join(lines)) from e

    base_dir = config_path.resolve().parent

    for ref in spec.generation.references:
        ref.path = _resolve_asset(ref.path, base_dir, "Reference image")
    for el in spec.elements:
        el.path = _resolve_asset(el.path, base_dir, "Element image")
    for font in (spec.title.font, spec.branding.font if spec.branding else None):
        if font and font.file:
            font.file = _resolve_asset(font.file, base_dir, "Font file")
    if spec.output:
        out = Path(spec.output).expanduser()
        if not out.is_absolute():
            out = base_dir / out
        spec.output = str(out)

    return spec


def resolve_position(
    pos: Position,
    canvas_size: tuple[int, int],
    obj_size: tuple[int, int],
) -> tuple[int, int]:
    """Top-left paste coordinates for an object of `obj_size` so that its
    `pos.anchor` point lands on (pos.x, pos.y)."""
    cw, ch = canvas_size
    ow, oh = obj_size

    if pos.unit == "fraction":
        px, py = pos.x * cw, pos.y * ch
    else:
        px, py = pos.x, pos.y

    # anchor values are "<vertical>_<horizontal>" except plain "center"
    if pos.anchor is Anchor.CENTER:
        vert, horiz = "center", "center"
    else:
        vert, horiz = pos.anchor.value.split("_", 1)

    if horiz == "left":
        x = px
    elif horiz == "right":
        x = px - ow
    else:  # center
        x = px - ow / 2

    if vert == "top":
        y = py
    elif vert == "bottom":
        y = py - oh
    else:  # center
        y = py - oh / 2

    return int(round(x)), int(round(y))
