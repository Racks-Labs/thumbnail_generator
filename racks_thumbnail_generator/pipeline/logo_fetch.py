"""Fetch brand logos from Wikipedia for use as Nano Banana reference images.

Strategy:
1. opensearch: query → canonical Wikipedia page title
2. parse: list image filenames on that page, filter for logo-like names
3. query: resolve File: title → upload URL
4. SVG → PNG via Wikipedia's auto-generated /thumb/.../<W>px-...png pattern
5. download bytes to a local cache file
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

UA = "racks-thumbnail/0.7 (https://github.com/Racks-Labs/thumbnail_generator)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
TIMEOUT = 10


def _http_get(url: str, params: dict | None = None) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def _wiki_page_title(query: str) -> str | None:
    try:
        raw = _http_get(WIKI_API, {
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "format": "json",
        })
        data = json.loads(raw)
        if isinstance(data, list) and len(data) >= 2 and data[1]:
            return data[1][0]
    except Exception:
        pass
    return None


def _list_logo_filenames(title: str) -> list[str]:
    try:
        raw = _http_get(WIKI_API, {
            "action": "parse",
            "page": title,
            "prop": "images",
            "format": "json",
            "redirects": "1",
        })
        data = json.loads(raw)
    except Exception:
        return []

    images = data.get("parse", {}).get("images", [])
    scored: list[tuple[int, str]] = []
    for f in images:
        low = f.lower()
        # skip obviously non-logo files
        if any(skip in low for skip in ("commons-logo", "wikidata", "edit-icon", "ambox")):
            continue
        if not (low.endswith(".svg") or low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg")):
            continue
        score = 0
        if "logo" in low:
            score += 3
        if "wordmark" in low:
            score += 2
        if "brand" in low:
            score += 1
        if "icon" in low:
            score -= 1
        if "favicon" in low:
            score -= 2
        if "headquarters" in low or "building" in low or "office" in low:
            score -= 5  # photos of HQ are not logos
        if "ceo" in low or "founder" in low:
            score -= 5
        if score >= 1:
            scored.append((score, f))
    scored.sort(key=lambda sf: -sf[0])
    return [f for _, f in scored]


def _resolve_file_url(filename: str) -> str | None:
    try:
        raw = _http_get(WIKI_API, {
            "action": "query",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        })
        data = json.loads(raw)
    except Exception:
        return None

    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        info_list = p.get("imageinfo") or []
        if info_list and "url" in info_list[0]:
            return info_list[0]["url"]
    return None


_SVG_URL_RX = re.compile(
    r"^(https?://upload\.wikimedia\.org/wikipedia/[a-z]+)/([0-9a-f])/([0-9a-f]+)/(.+\.svg)$"
)


def _svg_to_png_url(svg_url: str, width: int = 512) -> str:
    m = _SVG_URL_RX.match(svg_url)
    if not m:
        return svg_url
    base, h1, h2, fname = m.groups()
    return f"{base}/thumb/{h1}/{h2}/{fname}/{width}px-{fname}.png"


def _safe_name(query: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", query.lower()).strip("_")[:40] or "brand"


def fetch_brand_logo(query: str, out_dir: Path, width: int = 512) -> Path | None:
    """Fetch a single brand logo PNG. Returns local Path on success, None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    title = _wiki_page_title(query)
    if not title:
        return None
    candidates = _list_logo_filenames(title)
    if not candidates:
        return None

    for fname in candidates[:5]:
        url = _resolve_file_url(fname)
        if not url:
            continue
        if url.lower().endswith(".svg"):
            url = _svg_to_png_url(url, width)
        try:
            data = _http_get(url)
        except Exception:
            continue
        if len(data) < 200:
            continue
        out_path = out_dir / f"logo_{_safe_name(query)}.png"
        out_path.write_bytes(data)
        return out_path
    return None


def fetch_brand_logos(queries: list[str], out_dir: Path, width: int = 512) -> list[Path]:
    """Fetch multiple brand logos. Returns list of successful paths (failures skipped silently)."""
    paths: list[Path] = []
    for q in queries:
        if not q.strip():
            continue
        p = fetch_brand_logo(q.strip(), out_dir, width=width)
        if p:
            paths.append(p)
    return paths
