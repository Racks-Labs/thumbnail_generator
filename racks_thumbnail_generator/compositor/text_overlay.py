import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from racks_thumbnail_generator.spec import (
    BrandingSpec,
    Position,
    TitleSpec,
    resolve_position,
)


def _normalize(s: str) -> str:
    """Uppercase + strip accents + drop non-alphanumerics. For fuzzy word matching."""
    s = s.upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # drop combining marks
    return "".join(c for c in s if c.isalnum())


def _resolve_accent_indices(headline_words: list[str], accent_word: str) -> set[int]:
    """Return the indices of headline_words that should get the accent block.
    Tries exact, then normalized, then substring, then 'longest word' fallback."""
    if not accent_word.strip():
        return set()

    norm_words = [_normalize(w) for w in headline_words]
    norm_accent = _normalize(accent_word)
    if not norm_accent:
        return set()

    # 1) exact normalized match (handles case + punctuation + accents)
    matches = {i for i, w in enumerate(norm_words) if w == norm_accent}
    if matches:
        return matches

    # 2) accent_word may be multi-token ("LA MITAD") → match each subtoken
    accent_tokens = [_normalize(t) for t in accent_word.split() if _normalize(t)]
    if len(accent_tokens) > 1:
        matches = {i for i, w in enumerate(norm_words) if w in accent_tokens}
        if matches:
            return matches

    # 3) substring either way (handles morphological variation: DISEÑA vs DISEÑAR)
    matches = {
        i for i, w in enumerate(norm_words)
        if w and (w in norm_accent or norm_accent in w)
    }
    if matches:
        return matches

    # 4) fallback: pick the single longest headline word (skip very short ones)
    candidates = [(i, w) for i, w in enumerate(norm_words) if len(w) >= 4]
    if candidates:
        candidates.sort(key=lambda iw: -len(iw[1]))
        return {candidates[0][0]}
    return set()


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
    line_gap_ratio: float = LINE_GAP_RATIO,
) -> tuple[ImageFont.FreeTypeFont, list[list[str]], int]:
    """Find largest font size that fits text into max_width x max_height after wrapping."""
    words = text.split()
    for size in range(start_size, min_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        lines = _wrap_words(words, font, max_width, draw)
        line_h = font.getbbox("Ay")[3]
        gap = int(line_h * line_gap_ratio)
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

    # Resolve accent indices over the FLAT word list of the headline (post-wrap)
    flat_words = [w for line in lines for w in line]
    accent_indices_flat = _resolve_accent_indices(flat_words, accent_word)

    # ---- Pre-compute element positions ----
    # Each entry: dict with type ('block' | 'text'), bbox/pos, color, etc.
    elements: list[dict] = []
    cursor_y = text_top_y
    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    accent_text_color = _readable_text_color(accent_rgb)

    flat_idx = -1
    for line_words in lines:
        bbox = draw.textbbox((0, 0), " ".join(line_words), font=font)
        ascent_offset = bbox[1]
        cursor_x = side_pad

        for i, word in enumerate(line_words):
            flat_idx += 1
            wbbox = draw.textbbox((0, 0), word, font=font)
            ww = wbbox[2] - wbbox[0]
            wh = wbbox[3] - wbbox[1]
            is_accent = flat_idx in accent_indices_flat

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


# ---------------------------------------------------------------------------
# Spec-driven title overlay (JSON config mode)
# ---------------------------------------------------------------------------


def _resolve_font_path(font_spec, fonts_dir: Path) -> Path:
    """`font.file` wins; otherwise `<fonts_dir>/<family>-<weight>.ttf`."""
    if font_spec.file:
        path = Path(font_spec.file)
        if not path.exists():
            raise FileNotFoundError(f"Font file not found: {path}")
        return path
    path = fonts_dir / f"{font_spec.family}-{font_spec.weight}.ttf"
    if not path.exists():
        available = sorted(p.stem for p in fonts_dir.glob("*.ttf"))
        raise FileNotFoundError(
            f"Font not found: {path}. Available in {fonts_dir}: {available}"
        )
    return path


def _match_accent_indices(words: list[str], accent_words: list[str]) -> set[int]:
    """Indices of `words` matching any of `accent_words` (accent/case/punct
    insensitive, with substring fallback per listed word). Unlike
    `_resolve_accent_indices`, an explicit list that matches nothing highlights
    nothing — no longest-word fallback."""
    norm_words = [_normalize(w) for w in words]
    matched: set[int] = set()
    for accent in accent_words:
        norm_accent = _normalize(accent)
        if not norm_accent:
            continue
        exact = {i for i, w in enumerate(norm_words) if w == norm_accent}
        if exact:
            matched |= exact
            continue
        matched |= {
            i for i, w in enumerate(norm_words)
            if w and (w in norm_accent or norm_accent in w)
        }
    return matched


def overlay_title(
    image: Image.Image,
    title: TitleSpec,
    fonts_dir: Path,
    branding: BrandingSpec | None = None,
) -> Image.Image:
    """Composite a JSON-spec-driven title (and optional branding) onto image.

    Fully configurable: text box (position/anchor/align), font, size (fixed or
    auto-fit), color, per-word accent (underline / highlight / none), shadow.
    """
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    text = (title.text or "").strip()
    if title.uppercase:
        text = text.upper()

    elements: list[dict] = []
    font = None

    if text:
        # ---- Text box rect (fractions of canvas + anchor) ----
        box_w = int(title.box.width * W)
        box_h = int(title.box.height * H)
        box_x, box_y = resolve_position(
            Position(x=title.box.x, y=title.box.y, anchor=title.box.anchor),
            (W, H),
            (box_w, box_h),
        )

        font_path = _resolve_font_path(title.font, fonts_dir)

        # ---- Font: fixed size or auto-fit into the box ----
        if title.size == "auto":
            start_size = max(int(box_h * 0.55), title.min_size)
            font, lines, line_h = _fit_font(
                text=text,
                font_path=font_path,
                max_width=box_w,
                max_height=box_h,
                start_size=start_size,
                min_size=title.min_size,
                draw=draw,
                line_gap_ratio=title.line_spacing,
            )
        else:
            font = ImageFont.truetype(str(font_path), int(title.size))
            lines = _wrap_words(text.split(), font, box_w, draw)
            line_h = font.getbbox("Ay")[3]

        gap = int(line_h * title.line_spacing)
        total_h = len(lines) * line_h + (len(lines) - 1) * gap

        # Vertical placement inside the box follows the box anchor's vertical part
        vert = "center" if title.box.anchor.value == "center" else title.box.anchor.value.split("_", 1)[0]
        if vert == "top":
            cursor_y = box_y
        elif vert == "bottom":
            cursor_y = box_y + box_h - total_h
        else:
            cursor_y = box_y + (box_h - total_h) // 2

        # ---- Accent word matching ----
        flat_words = [w for line in lines for w in line]
        accent_indices = (
            _match_accent_indices(flat_words, title.accent.words)
            if title.accent.style != "none"
            else set()
        )

        accent_rgb = _hex_to_rgb(title.accent.color)
        text_rgb = _hex_to_rgb(title.color)
        highlight_text_rgb = (
            _hex_to_rgb(title.accent.text_color)
            if title.accent.text_color
            else _readable_text_color(accent_rgb)
        )
        ascent = font.getmetrics()[0]
        underline_h = max(2, int(font.size * title.accent.thickness))
        underline_off = int(font.size * title.accent.offset)

        # ---- Layout: per-word positions ----
        space_w = draw.textbbox((0, 0), " ", font=font)[2]
        flat_idx = -1
        for line_words in lines:
            lbbox = draw.textbbox((0, 0), " ".join(line_words), font=font)
            line_w = lbbox[2] - lbbox[0]
            ascent_offset = lbbox[1]

            if title.box.align == "center":
                cursor_x = box_x + (box_w - line_w) // 2
            elif title.box.align == "right":
                cursor_x = box_x + box_w - line_w
            else:
                cursor_x = box_x

            for i, word in enumerate(line_words):
                flat_idx += 1
                wbbox = draw.textbbox((0, 0), word, font=font)
                ww = wbbox[2] - wbbox[0]
                wh = wbbox[3] - wbbox[1]
                is_accent = flat_idx in accent_indices

                if is_accent and title.accent.style == "highlight":
                    pad_x = int(font.size * 0.12)
                    pad_y = int(font.size * 0.08)
                    rect = (
                        cursor_x - pad_x,
                        cursor_y + ascent_offset - pad_y,
                        cursor_x + ww + pad_x,
                        cursor_y + ascent_offset + wh + pad_y,
                    )
                    elements.append({"kind": "block", "rect": rect, "color": accent_rgb})
                    elements.append({"kind": "text", "pos": (cursor_x, cursor_y), "word": word, "color": highlight_text_rgb, "shadowed": False})
                else:
                    if is_accent and title.accent.style == "underline":
                        uy = cursor_y + ascent + underline_off
                        rect = (cursor_x, uy, cursor_x + ww, uy + underline_h)
                        elements.append({"kind": "block", "rect": rect, "color": accent_rgb})
                    elements.append({"kind": "text", "pos": (cursor_x, cursor_y), "word": word, "color": text_rgb, "shadowed": True})

                cursor_x += ww
                if i < len(line_words) - 1:
                    cursor_x += space_w

            cursor_y += line_h + gap

    # ---- Branding ----
    brand = None
    if branding and branding.text.strip():
        brand_font = ImageFont.truetype(
            str(_resolve_font_path(branding.font, fonts_dir)),
            max(int(branding.size * H), 12),
        )
        bbbox = draw.textbbox((0, 0), branding.text, font=brand_font)
        bsize = (bbbox[2] - bbbox[0], bbbox[3] - bbbox[1])
        brand = {
            "text": branding.text,
            "font": brand_font,
            "pos": resolve_position(branding.position, (W, H), bsize),
            "color": _hex_to_rgb(branding.color),
        }

    if not elements and not brand:
        return img

    # ---- Pass 1: blurred drop shadow ----
    shadow = title.shadow
    if shadow.enabled:
        ref_size = font.size if font else int(H * 0.03)
        off_x = max(1, int(ref_size * shadow.offset_x))
        off_y = max(1, int(ref_size * shadow.offset_y))
        blur_radius = max(1, int(ref_size * shadow.blur))
        shadow_fill = _hex_to_rgb(shadow.color) + (int(shadow.opacity * 255),)

        shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_layer)

        for el in elements:
            if el["kind"] == "block":
                x0, y0, x1, y1 = el["rect"]
                sdraw.rectangle(
                    [(x0 + off_x, y0 + off_y), (x1 + off_x, y1 + off_y)],
                    fill=shadow_fill,
                )
            elif el["shadowed"]:
                sdraw.text(
                    (el["pos"][0] + off_x, el["pos"][1] + off_y),
                    el["word"],
                    font=font,
                    fill=shadow_fill,
                )

        if brand:
            sdraw.text(
                (brand["pos"][0] + max(1, off_x // 2), brand["pos"][1] + max(1, off_y // 2)),
                brand["text"],
                font=brand["font"],
                fill=shadow_fill,
            )

        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img.paste(shadow_layer, (0, 0), shadow_layer)

    # ---- Pass 2: content ----
    for el in elements:
        if el["kind"] == "block":
            x0, y0, x1, y1 = el["rect"]
            draw.rectangle([(x0, y0), (x1, y1)], fill=el["color"] + (255,))
        else:
            draw.text(el["pos"], el["word"], font=font, fill=el["color"] + (255,))

    if brand:
        draw.text(brand["pos"], brand["text"], font=brand["font"], fill=brand["color"] + (255,))

    return img
