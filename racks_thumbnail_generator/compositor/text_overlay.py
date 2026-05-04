from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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
        gap = int(line_h * 0.05)
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
    side_pad = int(W * 0.05)
    bottom_pad = int(H * 0.05)
    max_text_width = W - 2 * side_pad
    max_text_height = int(H * 0.30)

    headline_upper = headline.upper()
    accent_upper = accent_word.upper()

    # ---- Fit headline font ----
    start_size = max(int(H * 0.08), 60)
    min_size = max(int(H * 0.03), 24)

    font, lines, line_h = _fit_font(
        text=headline_upper,
        font_path=fonts_dir / "Inter-Black.ttf",
        max_width=max_text_width,
        max_height=max_text_height,
        start_size=start_size,
        min_size=min_size,
        draw=draw,
    )

    gap = int(line_h * 0.05)
    total_h = len(lines) * line_h + (len(lines) - 1) * gap

    # ---- Subtle dark gradient at bottom for legibility ----
    grad_h = total_h + bottom_pad * 2
    overlay = Image.new("RGBA", (W, grad_h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for i in range(grad_h):
        alpha = int(180 * (i / grad_h) ** 1.5)
        odraw.rectangle([(0, i), (W, i + 1)], fill=(0, 0, 0, alpha))
    img.paste(overlay, (0, H - grad_h), overlay)

    # ---- Render headline lines ----
    cursor_y = H - bottom_pad - total_h

    for line_words in lines:
        line_text = " ".join(line_words)
        bbox = draw.textbbox((0, 0), line_text, font=font)
        line_w = bbox[2] - bbox[0]
        x = side_pad
        ascent_offset = bbox[1]

        # Render word by word to inject accent block
        cursor_x = x
        space_w = draw.textbbox((0, 0), " ", font=font)[2]

        for i, word in enumerate(line_words):
            wbbox = draw.textbbox((0, 0), word, font=font)
            ww = wbbox[2] - wbbox[0]
            wh = wbbox[3] - wbbox[1]

            is_accent = word == accent_upper

            if is_accent:
                pad_x = int(font.size * 0.12)
                pad_y = int(font.size * 0.08)
                rect_x0 = cursor_x - pad_x
                rect_y0 = cursor_y + ascent_offset - pad_y
                rect_x1 = cursor_x + ww + pad_x
                rect_y1 = cursor_y + ascent_offset + wh + pad_y
                draw.rectangle(
                    [(rect_x0, rect_y0), (rect_x1, rect_y1)],
                    fill=accent_rgb + (255,),
                )

            draw.text(
                (cursor_x, cursor_y),
                word,
                font=font,
                fill=(255, 255, 255, 255),
            )

            cursor_x += ww
            if i < len(line_words) - 1:
                cursor_x += space_w

        cursor_y += line_h + gap

    # ---- Branding (top right) ----
    brand_size = max(int(H * 0.018), 16)
    brand_font = ImageFont.truetype(str(fonts_dir / "Inter-Bold.ttf"), brand_size)
    bbbox = draw.textbbox((0, 0), branding, font=brand_font)
    bw = bbbox[2] - bbbox[0]
    brand_x = W - side_pad - bw
    brand_y = int(H * 0.025)
    draw.text(
        (brand_x, brand_y),
        branding,
        font=brand_font,
        fill=(255, 255, 255, 255),
    )

    return img
