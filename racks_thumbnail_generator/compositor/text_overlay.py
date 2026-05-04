from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG relative luminance, 0..1."""
    def chan(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def _contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    l1 = _relative_luminance(rgb1)
    l2 = _relative_luminance(rgb2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _readable_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white — whichever has better contrast against bg.
    WCAG AA needs >= 4.5 for normal text. For large bold display text the
    threshold is more permissive (3.0) but we err on side of legibility."""
    white = (255, 255, 255)
    black = (0, 0, 0)
    return white if _contrast_ratio(white, bg_rgb) >= _contrast_ratio(black, bg_rgb) else black


def _wrap_words(words: list[str], font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[list[str]]:
    """Greedy word-wrap into lines that fit max_width."""
    lines: list[list[str]] = []
    current: list[str] = []

    for word in words:
        candidate = current + [word]
        text = " ".join(candidate)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = [word]

    if current:
        lines.append(current)
    return lines


LINE_GAP_RATIO = 0.02  # tighter line spacing


def _fit_font(
    text: str,
    font_path: Path,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int,
    draw: ImageDraw.ImageDraw,
) -> tuple[ImageFont.FreeTypeFont, list[list[str]], int]:
    """Find largest font size that fits text into max_width x max_height after wrapping."""
    words = text.split()
    for size in range(start_size, min_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        lines = _wrap_words(words, font, max_width, draw)
        line_h = font.getbbox("Ay")[3]
        gap = int(line_h * LINE_GAP_RATIO)
        total_h = len(lines) * line_h + (len(lines) - 1) * gap
        widths_ok = all(
            (draw.textbbox((0, 0), " ".join(l), font=font)[2]) <= max_width
            for l in lines
        )
        if total_h <= max_height and widths_ok:
            return font, lines, line_h
    # Fallback: smallest
    font = ImageFont.truetype(str(font_path), min_size)
    lines = _wrap_words(words, font, max_width, draw)
    return font, lines, font.getbbox("Ay")[3]


def overlay_text(
    image: Image.Image,
    headline: str,
    accent_word: str,
    accent_color_hex: str,
    fonts_dir: Path,
    branding: str = "RACKS",
) -> Image.Image:
    """Composite headline (with accent block on accent_word) and branding onto image."""
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    accent_rgb = _hex_to_rgb(accent_color_hex)

    # ---- Layout ----
    # Instagram Reels safe zones for 9:16 cover:
    #   - Feed grid (4:5 crop): top/bottom 14.8% cropped
    #   - Reel scroll view: bottom ~25-30% covered by caption/icons/profile UI
    # Combined safe zone for headline: between ~15% and ~75% from top.
    side_pad = int(W * 0.05)
    bottom_safe_pad = int(H * 0.22)  # distance from image bottom to bottom of headline
    max_text_width = W - 2 * side_pad
    # Reduced from 30% → 22% to match the safe zone available for the headline
    max_text_height = int(H * 0.22)

    headline_upper = headline.upper()
    accent_upper = accent_word.upper()

    # ---- Fit headline font ----
    # Reduced from 8% → 6.5% so text doesn't dominate within the safe zone
    start_size = max(int(H * 0.065), 50)
    min_size = max(int(H * 0.028), 22)

    font, lines, line_h = _fit_font(
        text=headline_upper,
        font_path=fonts_dir / "Inter-Black.ttf",
        max_width=max_text_width,
        max_height=max_text_height,
        start_size=start_size,
        min_size=min_size,
        draw=draw,
    )

    gap = int(line_h * LINE_GAP_RATIO)
    total_h = len(lines) * line_h + (len(lines) - 1) * gap

    text_bottom_y = H - bottom_safe_pad
    text_top_y = text_bottom_y - total_h

    # ---- Pre-compute element positions ----
    # Each entry: dict with type ('block' | 'text'), bbox/pos, color, etc.
    elements: list[dict] = []
    cursor_y = text_top_y
    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    accent_text_color = _readable_text_color(accent_rgb)

    for line_words in lines:
        bbox = draw.textbbox((0, 0), " ".join(line_words), font=font)
        ascent_offset = bbox[1]
        cursor_x = side_pad

        for i, word in enumerate(line_words):
            wbbox = draw.textbbox((0, 0), word, font=font)
            ww = wbbox[2] - wbbox[0]
            wh = wbbox[3] - wbbox[1]
            is_accent = word == accent_upper

            if is_accent:
                pad_x = int(font.size * 0.12)
                pad_y = int(font.size * 0.08)
                rect = (
                    cursor_x - pad_x,
                    cursor_y + ascent_offset - pad_y,
                    cursor_x + ww + pad_x,
                    cursor_y + ascent_offset + wh + pad_y,
                )
                elements.append({"kind": "block", "rect": rect, "color": accent_rgb})
                elements.append({"kind": "text", "pos": (cursor_x, cursor_y), "word": word, "color": accent_text_color, "shadowed": False})
            else:
                elements.append({"kind": "text", "pos": (cursor_x, cursor_y), "word": word, "color": (255, 255, 255), "shadowed": True})

            cursor_x += ww
            if i < len(line_words) - 1:
                cursor_x += space_w

        cursor_y += line_h + gap

    # ---- Branding position ----
    brand_size = max(int(H * 0.022), 18)
    brand_font = ImageFont.truetype(str(fonts_dir / "Inter-Bold.ttf"), brand_size)
    bbbox = draw.textbbox((0, 0), branding, font=brand_font)
    bw = bbbox[2] - bbbox[0]
    brand_pos = (W - side_pad - bw, int(H * 0.17))

    # ---- Pass 1: blurred drop shadow layer ----
    # Soft shadow for ALL headline elements (white text + accent block + branding).
    # Goal: legibility on bright scene zones without looking like a hard outline.
    shadow_offset_x = max(2, int(font.size * 0.025))
    shadow_offset_y = max(3, int(font.size * 0.04))
    blur_radius = max(4, int(font.size * 0.06))
    shadow_alpha = 160

    shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_layer)

    for el in elements:
        if el["kind"] == "block":
            x0, y0, x1, y1 = el["rect"]
            sdraw.rectangle(
                [(x0 + shadow_offset_x, y0 + shadow_offset_y),
                 (x1 + shadow_offset_x, y1 + shadow_offset_y)],
                fill=(0, 0, 0, shadow_alpha),
            )
        elif el["kind"] == "text" and el["shadowed"]:
            sdraw.text(
                (el["pos"][0] + shadow_offset_x, el["pos"][1] + shadow_offset_y),
                el["word"],
                font=font,
                fill=(0, 0, 0, shadow_alpha),
            )

    # Branding shadow
    sdraw.text(
        (brand_pos[0] + max(1, shadow_offset_x // 2), brand_pos[1] + max(2, shadow_offset_y // 2)),
        branding,
        font=brand_font,
        fill=(0, 0, 0, shadow_alpha),
    )

    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img.paste(shadow_layer, (0, 0), shadow_layer)

    # ---- Pass 2: content (blocks + text on top of shadow) ----
    for el in elements:
        if el["kind"] == "block":
            x0, y0, x1, y1 = el["rect"]
            draw.rectangle([(x0, y0), (x1, y1)], fill=el["color"] + (255,))
        else:
            draw.text(el["pos"], el["word"], font=font, fill=el["color"] + (255,))

    draw.text(brand_pos, branding, font=brand_font, fill=(255, 255, 255, 255))

    return img
