# racks-thumbnail-generator

Automatic thumbnail generator for Racks reels. Pipeline: video → audio → transcription → topic extraction → AI scene (Nanobanana) → text overlay (Pillow + Inter + exact accent color).

## Stack
- Python 3.12 via uv
- Gemini API (google-genai) — transcription, topic extraction, image generation (Nanobanana 2 / `gemini-2.5-flash-image`)
- OpenAI API (optional) for Whisper transcription
- ffmpeg for audio extraction
- Pillow for text composition (Inter font + pixel-exact colors)

## Architecture
- `racks_thumbnail_generator/` — installable package
  - `spec.py` — pydantic schema for the JSON config mode. **This is the contract
    with external frontends** — bump `ThumbnailSpec.version` on breaking changes.
  - `pipeline/` — audio, transcription, topic extraction, `spec_runner.py`
    (JSON-config orchestration: AI scene w/ reference images → deterministic overlays)
  - `generator/` — AI image generation + prompt builder (incl. reference-role block)
  - `compositor/` — Pillow overlays (post-AI): `text_overlay.py` (legacy
    `overlay_text` + spec-driven `overlay_title` with per-word underline/highlight),
    `composer.py` (graphic elements at fixed positions)
  - `templates/default.yaml` — scene + style template
  - `assets/fonts/Inter/` — bundled Inter TTFs
- `examples/thumbnail_config.json` — working JSON config example (+ placeholder assets)
- `pyproject.toml` — entry point: `racks-thumbnail = "racks_thumbnail_generator.main:app"`

## Dev commands
- `uv run racks-thumbnail generate <video>` — full pipeline
- `uv run racks-thumbnail generate-from-config <config.json>` — JSON config mode
- `uv run racks-thumbnail config-schema` — JSON Schema for frontend integration
- `uv run racks-thumbnail test-transcribe <audio>` — only transcription + topic
- `uv run racks-thumbnail config show` — view stored config
- `uv run pytest` — test suite (spec validation, positioning, overlays)
- `uv build` — build wheel for distribution
- Bump `pyproject.toml` version on changes (Hatchling caches by version)

## Config
Global config at `~/.config/racks-thumbnail/.env` (chmod 600). Precedence: CLI flag > env var > local `.env` > global config.

## Style
- Brand: Racks. Colors: black `#000000`, white `#FFFFFF`, red `#BE190F` (configurable)
- Font: Inter (bundled — Black, Bold, SemiBold, Medium, Regular, Light)
- Reference: Guia_estilos_RU.pdf
