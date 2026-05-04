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
  - `pipeline/` — audio, transcription, topic extraction
  - `generator/` — AI image generation + prompt builder
  - `compositor/` — Pillow text overlay (post-AI)
  - `templates/default.yaml` — scene + style template
  - `assets/fonts/Inter/` — bundled Inter TTFs
- `pyproject.toml` — entry point: `racks-thumbnail = "racks_thumbnail_generator.main:app"`

## Dev commands
- `uv run racks-thumbnail generate <video>` — full pipeline
- `uv run racks-thumbnail test-transcribe <audio>` — only transcription + topic
- `uv run racks-thumbnail config show` — view stored config
- `uv build` — build wheel for distribution
- Bump `pyproject.toml` version on changes (Hatchling caches by version)

## Config
Global config at `~/.config/racks-thumbnail/.env` (chmod 600). Precedence: CLI flag > env var > local `.env` > global config.

## Style
- Brand: Racks. Colors: black `#000000`, white `#FFFFFF`, red `#BE190F` (configurable)
- Font: Inter (bundled — Black, Bold, SemiBold, Medium, Regular, Light)
- Reference: Guia_estilos_RU.pdf
