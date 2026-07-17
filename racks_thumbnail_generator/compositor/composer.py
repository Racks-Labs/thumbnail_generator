"""Deterministic graphic-element composition (logos, badges, ...).

Runs AFTER the AI scene is generated: pure Pillow, exact positions from the
JSON spec — same guarantee as the text overlay.
"""

from pathlib import Path

from PIL import Image

from racks_thumbnail_generator.spec import ElementSpec, resolve_position


def _scaled_size(el: ElementSpec, img: Image.Image, canvas_size: tuple[int, int]) -> tuple[int, int]:
    cw, ch = canvas_size
    ow, oh = img.size

    if el.width is not None and el.height is not None:
        return max(1, int(el.width * cw)), max(1, int(el.height * ch))
    if el.width is not None:
        w = max(1, int(el.width * cw))
        return w, max(1, int(oh * w / ow))
    if el.height is not None:
        h = max(1, int(el.height * ch))
        return max(1, int(ow * h / oh)), h
    return ow, oh


def paste_elements(canvas: Image.Image, elements: list[ElementSpec]) -> Image.Image:
    """Alpha-paste each element onto the canvas at its spec position, in order."""
    if not elements:
        return canvas

    out = canvas.convert("RGB").copy()

    for el in elements:
        img = Image.open(Path(el.path)).convert("RGBA")
        size = _scaled_size(el, img, out.size)
        if size != img.size:
            img = img.resize(size, Image.LANCZOS)

        if el.opacity < 1.0:
            alpha = img.getchannel("A").point(lambda a: int(a * el.opacity))
            img.putalpha(alpha)

        pos = resolve_position(el.position, out.size, img.size)
        out.paste(img, pos, img)

    return out
