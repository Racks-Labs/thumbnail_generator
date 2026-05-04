---
name: racks-thumbnail
description: Generate Instagram reel thumbnails for Racks brand from a video file. Pipeline extracts audio, transcribes, identifies the value proposition, generates a cinematic AI scene with Nanobanana, and composites the headline with Inter font and exact brand colors.
---

# Racks Thumbnail Generator

Use this skill when the user asks to create, generate, make, or design a thumbnail / miniatura for a Racks reel from a video file.

## Prerequisites check

Before invoking, verify the user has the tool installed and configured:

```bash
racks-thumbnail --help
```

If command not found, instruct the user to install:
```bash
uv tool install git+https://github.com/Racks-Labs/thumbnail_generator
```

If `GOOGLE_API_KEY` not configured, instruct:
```bash
racks-thumbnail config set google-api-key <key>
```
(They can get a key at https://aistudio.google.com/apikey)

## Basic usage

```bash
racks-thumbnail generate <path/to/video.mp4>
```

Output is written to `./output/thumbnail_<timestamp>.png` (and a `_bg_<timestamp>.png` intermediate scene).

## Common flags

- `--output <dir>` — change output directory (default `./output`)
- `--accent-color "#FF6B00"` — override the accent color block (default `#BE190F` Racks red)
- `--transcriber whisper` — use OpenAI Whisper instead of Gemini (requires `OPENAI_API_KEY`)
- `--model gemini-3.1-flash-image-preview` — try a newer image model

## Auxiliary commands

- `racks-thumbnail test-transcribe <audio.mp3>` — only run transcription + topic extraction without generating an image (faster, useful for tuning)
- `racks-thumbnail config show` — display current config (API keys masked)
- `racks-thumbnail config set <key> <value>` — update a config key
- `racks-thumbnail config path` — print path to config file

## Behavior notes

- The tool reads the video, transcribes the audio in Spanish, and uses Gemini to identify the core value proposition of the reel — the headline is built around what the viewer GAINS by watching, not generic slogans.
- The visual scene is generated cinematically (A24 film-still aesthetic) with high-contrast chiaroscuro lighting. One main subject + one real recognizable secondary object.
- Text is composited with Pillow using bundled Inter font and pixel-exact accent color — never AI-rendered (which would be inaccurate).
- Default canvas: 9:16 vertical (768×1344 from Nanobanana 2.5).

## What NOT to do

- Don't pass video files larger than ~500MB without warning the user (transcription cost scales)
- Don't bypass the tool by trying to call Gemini directly — the tool already handles prompt engineering, fallbacks, and brand consistency
- Don't modify generated thumbnails by re-running with the same input expecting determinism — outputs vary per generation (each run gives a different scene)

## Example end-to-end flow

```bash
# Once
uv tool install git+https://github.com/Racks-Labs/thumbnail_generator
racks-thumbnail config set google-api-key AIzaSy...

# Per video
cd ~/Desktop
racks-thumbnail generate ~/Downloads/reel-15.mp4
open output/thumbnail_*.png
```

Output gets placed in the current working directory's `./output/`. Choose your `cd` accordingly.
