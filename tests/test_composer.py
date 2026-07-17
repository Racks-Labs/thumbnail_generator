from pathlib import Path

from PIL import Image

from racks_thumbnail_generator.compositor.composer import paste_elements
from racks_thumbnail_generator.compositor.text_overlay import overlay_title
from racks_thumbnail_generator.config import BUNDLED_FONTS
from racks_thumbnail_generator.spec import (
    AccentSpec,
    Anchor,
    BrandingSpec,
    ElementSpec,
    Position,
    ShadowSpec,
    TextBoxSpec,
    TitleSpec,
)

RED = (255, 0, 0)
TEAL = (53, 208, 186)


def _solid_png(path: Path, size, color, alpha=255):
    img = Image.new("RGBA", size, color + (alpha,))
    img.save(path)
    return path


def test_paste_elements_position_and_size(tmp_path):
    logo = _solid_png(tmp_path / "logo.png", (100, 50), RED)
    canvas = Image.new("RGB", (1000, 2000), (0, 0, 0))

    el = ElementSpec(
        path=str(logo),
        position=Position(x=0.05, y=0.95, anchor=Anchor.BOTTOM_LEFT),
        width=0.2,  # → 200px wide, 100px tall (aspect kept)
    )
    out = paste_elements(canvas, [el])

    # bottom-left anchor at (50, 1900) → element occupies x 50-250, y 1800-1900
    assert out.getpixel((51, 1801)) == RED
    assert out.getpixel((249, 1899)) == RED
    assert out.getpixel((251, 1850)) == (0, 0, 0)
    assert out.getpixel((51, 1799)) == (0, 0, 0)


def test_paste_elements_opacity(tmp_path):
    logo = _solid_png(tmp_path / "logo.png", (10, 10), (255, 255, 255))
    canvas = Image.new("RGB", (100, 100), (0, 0, 0))
    el = ElementSpec(path=str(logo), position=Position(x=0, y=0, unit="px"), opacity=0.5)
    out = paste_elements(canvas, [el])
    r, g, b = out.getpixel((5, 5))
    assert 120 <= r <= 135  # ~50% blend


def test_overlay_title_underline_present():
    canvas = Image.new("RGB", (1080, 1920), (0, 0, 0))
    title = TitleSpec(
        text="hola mundo",
        box=TextBoxSpec(x=0.05, y=0.1, width=0.9, height=0.3, anchor=Anchor.TOP_LEFT),
        accent=AccentSpec(style="underline", words=["mundo"], color="#35D0BA"),
        shadow=ShadowSpec(enabled=False),
    )
    out = overlay_title(canvas, title, BUNDLED_FONTS)

    colors = {c for _, c in out.convert("RGB").getcolors(maxcolors=100000)}
    assert TEAL in colors  # underline drawn
    assert (255, 255, 255) in colors  # text drawn


def test_overlay_title_no_accent_when_style_none():
    canvas = Image.new("RGB", (1080, 1920), (0, 0, 0))
    title = TitleSpec(
        text="hola mundo",
        accent=AccentSpec(style="none", words=["mundo"], color="#35D0BA"),
        shadow=ShadowSpec(enabled=False),
    )
    out = overlay_title(canvas, title, BUNDLED_FONTS)
    colors = {c for _, c in out.convert("RGB").getcolors(maxcolors=100000)}
    assert TEAL not in colors


def test_overlay_title_fixed_size_and_color():
    canvas = Image.new("RGB", (1080, 1920), (0, 0, 0))
    title = TitleSpec(
        text="ROJO",
        color="#FF0000",
        size=80,
        accent=AccentSpec(style="none"),
        shadow=ShadowSpec(enabled=False),
    )
    out = overlay_title(canvas, title, BUNDLED_FONTS)
    colors = {c for _, c in out.convert("RGB").getcolors(maxcolors=100000)}
    assert RED in colors


def test_overlay_title_highlight_style():
    canvas = Image.new("RGB", (1080, 1920), (0, 0, 0))
    title = TitleSpec(
        text="hola mundo",
        accent=AccentSpec(style="highlight", words=["mundo"], color="#BE190F"),
        shadow=ShadowSpec(enabled=False),
    )
    out = overlay_title(canvas, title, BUNDLED_FONTS)
    colors = {c for _, c in out.convert("RGB").getcolors(maxcolors=100000)}
    assert (190, 25, 15) in colors


def test_overlay_title_branding_only():
    canvas = Image.new("RGB", (1080, 1920), (0, 0, 0))
    title = TitleSpec(text=None, shadow=ShadowSpec(enabled=False))
    out = overlay_title(canvas, title, BUNDLED_FONTS, branding=BrandingSpec(color="#FF0000"))
    colors = {c for _, c in out.convert("RGB").getcolors(maxcolors=100000)}
    assert RED in colors
