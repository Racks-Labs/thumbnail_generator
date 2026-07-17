import os
import sys
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from racks_thumbnail_generator.config import (
    get_settings,
    TranscriberBackend,
    global_config_dir,
    global_config_path,
    read_global_config,
    write_global_config,
    set_global_config_value,
    CONFIG_KEYS,
)
from racks_thumbnail_generator.pipeline.audio import extract_audio
from racks_thumbnail_generator.pipeline.transcriber import get_transcriber
from racks_thumbnail_generator.pipeline.topic_extractor import extract_topic
from racks_thumbnail_generator.pipeline.logo_fetch import fetch_brand_logos
from racks_thumbnail_generator.generator.prompt_builder import build_prompt
from racks_thumbnail_generator.generator.image_gen import generate_thumbnail, DEFAULT_MODEL
from racks_thumbnail_generator.compositor.text_overlay import overlay_text
from racks_thumbnail_generator.pipeline.spec_runner import run_from_spec
from racks_thumbnail_generator.spec import SpecError, load_spec

APP_HELP = """\
Generate cinematic thumbnails for Racks reels from a video file or script.

Pipeline: video → audio → transcription → topic → AI scene (Nanobanana) → text overlay (Pillow + Inter + brand color).
Or, skip the first two steps: script → topic → AI scene → text overlay.

[bold]Quick start[/bold]
  [cyan]racks-thumbnail config set google-api-key TU_KEY[/cyan]    First-time setup
  [cyan]racks-thumbnail generate video.mp4[/cyan]                  Generate from video (full pipeline)
  [cyan]racks-thumbnail generate-from-script script.txt[/cyan]     Skip transcription, use existing script
  [cyan]racks-thumbnail generate-from-config config.json[/cyan]    JSON config: reference images + fixed overlays + styled title
  [cyan]racks-thumbnail generate video.mp4 --config config.json[/cyan]      Video pipeline + JSON composition
  [cyan]racks-thumbnail generate video.mp4 --accent-color "#FF6B00"[/cyan]   Custom color
  [cyan]racks-thumbnail test-transcribe audio.mp3[/cyan]           Test transcription only
  [cyan]racks-thumbnail config show[/cyan]                         View saved config

[bold]Tab completion[/bold]
  Auto-installed on first run. Open a new shell to enable.
  Manual: [cyan]racks-thumbnail completion --install[/cyan]   /   uninstall: [cyan]racks-thumbnail completion --uninstall[/cyan]

[bold]Docs[/bold]
  https://github.com/Racks-Labs/thumbnail_generator
"""

app = typer.Typer(
    name="racks-thumbnail",
    help=APP_HELP,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
config_app = typer.Typer(
    name="config",
    help="Manage API keys and settings (stored at ~/.config/racks-thumbnail/.env)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(config_app, name="config")
console = Console()


def _safe_logo_name(query: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", query.lower()).strip("_")[:40] or "brand"


def _mask(value: str) -> str:
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _ensure_api_key(settings) -> None:
    """If GOOGLE_API_KEY missing, prompt user and persist to global config."""
    if settings.google_api_key:
        return
    console.print("[yellow]No GOOGLE_API_KEY found.[/yellow]")
    console.print(f"Will be saved to {global_config_path()} (chmod 600).")
    key = Prompt.ask("Paste your Google API key", password=True).strip()
    if not key:
        console.print("[red]No key provided. Aborting.[/red]")
        raise typer.Exit(1)
    set_global_config_value("GOOGLE_API_KEY", key)
    settings.google_api_key = key
    console.print("[green]Saved.[/green]\n")


def _resolve_template_path(settings, template: str) -> Path:
    template_path = settings.templates_dir / f"{template}.yaml"
    if not template_path.exists():
        console.print(f"[red]Template not found: {template_path}[/red]")
        raise typer.Exit(1)
    return template_path


def _run_thumbnail_from_transcript(
    transcript: str,
    settings,
    template_path: Path,
    model: str,
    show_prompt: bool,
    step_offset: int = 0,
    total_steps: int = 2,
) -> None:
    """Steps after transcription: topic extraction → logos → image gen → text overlay."""
    topic_step = 1 + step_offset
    fetch_step = 2 + step_offset

    console.print(f"[bold]{topic_step}/{total_steps}[/bold] Extracting topic + visual concept...")
    content = extract_topic(transcript, settings.google_api_key)
    from racks_thumbnail_generator.compositor.text_overlay import _resolve_accent_indices
    headline_words = content.headline.split()
    matched = _resolve_accent_indices(headline_words, content.headline_accent_word)
    matched_words = [headline_words[i] for i in sorted(matched)]
    accent_display = (
        f"[bold]{content.headline_accent_word}[/bold] → matched: {', '.join(matched_words)}"
        if matched else
        f"[bold]{content.headline_accent_word}[/bold] [yellow](no exact match — fallback used)[/yellow]"
    )
    console.print(f"  Headline: [bold red]{content.headline}[/bold red]  ({len(headline_words)} words, {len(content.headline)} chars)")
    console.print(f"  Accent word: {accent_display}")
    console.print(f"  Tono: {content.tono}")
    console.print(f"  Scene focus: [bold cyan]{content.subject_focus}[/bold cyan]")
    console.print(Panel(content.concepto_visual, title="Concepto visual"))

    references = []
    if content.brand_queries:
        console.print(f"[bold]{fetch_step}/{total_steps}[/bold] Fetching {len(content.brand_queries)} brand logo(s)...")
        logo_dir = settings.output_dir / "_logos"
        references = fetch_brand_logos(content.brand_queries, logo_dir)
        for q in content.brand_queries:
            ok = any(_safe_logo_name(q) in r.name for r in references)
            status = "[green]✓[/green]" if ok else "[yellow]✗ not found[/yellow]"
            console.print(f"  {status} {q}")

    user_refs = Path.cwd() / "references"
    if user_refs.exists():
        references += [
            r for r in user_refs.glob("*")
            if r.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

    prompt = build_prompt(template_path, content, settings.accent_color)

    if show_prompt:
        console.print(Panel(prompt, title="Final Nano Banana prompt", border_style="cyan"))
        if references:
            console.print(f"[dim]Reference images that would be passed: {[str(r) for r in references]}[/dim]")
        console.print("[yellow]--show-prompt: skipping image generation.[/yellow]")
        return

    console.print("Generating background scene with Nanobanana...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_path = settings.output_dir / f"_bg_{timestamp}.png"

    background = generate_thumbnail(
        prompt=prompt,
        api_key=settings.google_api_key,
        model=model,
        reference_images=references or None,
        output_path=background_path,
    )

    console.print("  Compositing text overlay (Inter + exact accent color)...")
    final = overlay_text(
        image=background,
        headline=content.headline,
        accent_word=content.headline_accent_word,
        accent_color_hex=settings.accent_color,
        fonts_dir=settings.fonts_dir,
        branding="RACKS",
    )

    output_path = settings.output_dir / f"thumbnail_{timestamp}.png"
    final.save(output_path, quality=95)

    console.print(f"\n[bold green]Thumbnail saved → {output_path}[/bold green]")
    console.print(f"  Size: {final.size[0]}x{final.size[1]}")


def _load_spec_or_exit(config: Path):
    try:
        return load_spec(config)
    except SpecError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def generate(
    video: Path = typer.Argument(..., help="Path to video file (.mp4)"),
    template: str = typer.Option("default", help="Template name from templates/"),
    output: Path = typer.Option(None, help="Output directory"),
    accent_color: str = typer.Option(None, help="Accent color hex (default: #BE190F)"),
    transcriber: str = typer.Option(None, help="Transcription backend: gemini or whisper"),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemini image model"),
    config: Path = typer.Option(None, "--config", help="JSON thumbnail config (reference images + overlays + title style). See examples/thumbnail_config.json"),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the final Nano Banana prompt and exit (no image generated)"),
):
    """Full pipeline: video → audio → transcription → thumbnail."""
    settings = get_settings()
    _ensure_api_key(settings)

    if accent_color:
        settings.accent_color = accent_color
    if transcriber:
        settings.transcriber = TranscriberBackend(transcriber)
    if output:
        settings.output_dir = output
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    if not video.exists():
        console.print(f"[red]Video not found: {video}[/red]")
        raise typer.Exit(1)

    spec = _load_spec_or_exit(config) if config else None
    template_path = _resolve_template_path(settings, template)

    console.print("[bold]1/4[/bold] Extracting audio...")
    audio_path = extract_audio(video)

    console.print(f"[bold]2/4[/bold] Transcribing with {settings.transcriber.value}...")
    transcriber_impl = get_transcriber(
        backend=settings.transcriber.value,
        google_api_key=settings.google_api_key,
        openai_api_key=settings.openai_api_key,
    )
    transcript = transcriber_impl.transcribe(audio_path)
    console.print(Panel(transcript[:300] + ("..." if len(transcript) > 300 else ""), title="Transcripción"))

    try:
        if spec:
            try:
                run_from_spec(
                    spec=spec,
                    settings=settings,
                    transcript=transcript,
                    model=model,
                    show_prompt=show_prompt,
                )
            except SpecError as e:
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1)
        else:
            _run_thumbnail_from_transcript(
                transcript=transcript,
                settings=settings,
                template_path=template_path,
                model=model,
                show_prompt=show_prompt,
                step_offset=2,
                total_steps=4,
            )
    finally:
        audio_path.unlink(missing_ok=True)


@app.command(name="generate-from-config")
def generate_from_config(
    config: Path = typer.Argument(..., help="JSON thumbnail config. See examples/thumbnail_config.json"),
    script: Path = typer.Option(None, "--script", help="Script/transcript file ('-' for stdin) — only needed when the config omits title.text or generation.prompt"),
    output: Path = typer.Option(None, help="Output directory (overrides default; config `output` wins)"),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemini image model"),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the final Nano Banana prompt and exit (no image generated)"),
):
    """Generate a thumbnail from a JSON config: AI scene from reference images
    + deterministic overlays (elements at fixed positions + styled title)."""
    settings = get_settings()

    spec = _load_spec_or_exit(config)

    transcript = None
    if script is not None:
        if str(script) == "-":
            transcript = sys.stdin.read().strip()
        else:
            if not script.exists():
                console.print(f"[red]Script not found: {script}[/red]")
                raise typer.Exit(1)
            transcript = script.read_text(encoding="utf-8").strip()

    # API key needed to generate (and for topic extraction); --show-prompt with a
    # fully specified config works without one.
    fully_specified = spec.title.text is not None and spec.generation.prompt is not None
    if not (show_prompt and fully_specified):
        _ensure_api_key(settings)

    if output:
        settings.output_dir = output
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_from_spec(
            spec=spec,
            settings=settings,
            transcript=transcript,
            model=model,
            show_prompt=show_prompt,
        )
    except SpecError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command(name="config-schema")
def config_schema():
    """Print the JSON Schema of the thumbnail config (for frontend integration)."""
    import json

    from racks_thumbnail_generator.spec import ThumbnailSpec

    print(json.dumps(ThumbnailSpec.model_json_schema(), indent=2, ensure_ascii=False))


@app.command(name="generate-from-script")
def generate_from_script(
    script: Path = typer.Argument(..., help="Path to script/transcript file (plain text). Use '-' to read from stdin."),
    template: str = typer.Option("default", help="Template name from templates/"),
    output: Path = typer.Option(None, help="Output directory"),
    accent_color: str = typer.Option(None, help="Accent color hex (default: #BE190F)"),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemini image model"),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the final Nano Banana prompt and exit (no image generated)"),
):
    """Generate from an existing script — skips audio extraction + transcription."""
    settings = get_settings()
    _ensure_api_key(settings)

    if accent_color:
        settings.accent_color = accent_color
    if output:
        settings.output_dir = output
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    if str(script) == "-":
        transcript = sys.stdin.read()
    else:
        if not script.exists():
            console.print(f"[red]Script not found: {script}[/red]")
            raise typer.Exit(1)
        transcript = script.read_text(encoding="utf-8")

    transcript = transcript.strip()
    if not transcript:
        console.print("[red]Script is empty.[/red]")
        raise typer.Exit(1)

    template_path = _resolve_template_path(settings, template)

    console.print(Panel(transcript[:300] + ("..." if len(transcript) > 300 else ""), title="Script"))

    _run_thumbnail_from_transcript(
        transcript=transcript,
        settings=settings,
        template_path=template_path,
        model=model,
        show_prompt=show_prompt,
        step_offset=0,
        total_steps=2,
    )


@app.command(name="test-transcribe")
def test_transcribe(
    audio: Path = typer.Argument(..., help="Path to audio file (.mp3, .wav)"),
    transcriber: str = typer.Option(None, help="Backend: gemini or whisper"),
):
    """Test transcription without full pipeline."""
    settings = get_settings()
    _ensure_api_key(settings)
    if transcriber:
        settings.transcriber = TranscriberBackend(transcriber)

    transcriber_impl = get_transcriber(
        backend=settings.transcriber.value,
        google_api_key=settings.google_api_key,
        openai_api_key=settings.openai_api_key,
    )
    transcript = transcriber_impl.transcribe(audio)
    console.print(Panel(transcript, title="Transcripción"))

    content = extract_topic(transcript, settings.google_api_key)
    console.print(f"\nHeadline: [bold red]{content.headline}[/bold red]")
    console.print(f"Accent word: [bold]{content.headline_accent_word}[/bold]")
    console.print(f"Subtexto: {content.subtexto}")
    console.print(f"Keywords: {', '.join(content.keywords)}")
    console.print(f"Tono: {content.tono}")
    console.print(Panel(content.concepto_visual, title="Concepto visual"))


# ---- Config subcommands ----

CONFIG_KEY_HELP = {
    "GOOGLE_API_KEY": "Google AI Studio API key (required for Gemini transcription + image gen)",
    "OPENAI_API_KEY": "OpenAI API key (required only for --transcriber whisper)",
    "TRANSCRIBER":    "Default transcription backend: gemini | whisper",
    "ACCENT_COLOR":   "Default accent color hex (e.g. #BE190F)",
}

CONFIG_KEYS_HINT = (
    "Available keys:\n"
    + "\n".join(f"  • {k.lower().replace('_', '-')}  —  {desc}" for k, desc in CONFIG_KEY_HELP.items())
)


def _complete_config_key(incomplete: str) -> list[str]:
    """Tab completion candidates for config keys."""
    candidates = [k.lower().replace("_", "-") for k in CONFIG_KEYS]
    return [c for c in candidates if c.startswith(incomplete.lower())]


def _show_keys_and_exit():
    console.print("[yellow]Missing key argument.[/yellow]\n")
    console.print(CONFIG_KEYS_HINT)
    console.print(
        f"\n[bold]Examples:[/bold]\n"
        f"  [cyan]racks-thumbnail config set google-api-key AIza...[/cyan]\n"
        f"  [cyan]racks-thumbnail config set accent-color \"#BE190F\"[/cyan]\n"
        f"  [cyan]racks-thumbnail config set transcriber whisper[/cyan]"
    )
    raise typer.Exit(1)


@config_app.command(
    "set",
    help=(
        "Set a config value (saved to ~/.config/racks-thumbnail/.env).\n\n"
        + CONFIG_KEYS_HINT
    ),
)
def config_set(
    key: str = typer.Argument(
        None,
        help="Config key (e.g. google-api-key)",
        autocompletion=_complete_config_key,
    ),
    value: str = typer.Argument(None, help="Value to store"),
):
    if key is None:
        _show_keys_and_exit()
    if value is None:
        console.print(f"[yellow]Missing value for {key}.[/yellow]")
        console.print(f"\nUsage: [cyan]racks-thumbnail config set {key} <value>[/cyan]")
        raise typer.Exit(1)
    try:
        path = set_global_config_value(key, value)
    except ValueError as e:
        console.print(f"[red]{e}[/red]\n")
        console.print(CONFIG_KEYS_HINT)
        raise typer.Exit(1)
    console.print(f"[green]Saved {key.upper().replace('-', '_')} → {path}[/green]")


@config_app.command(
    "get",
    help=(
        "Read a config value.\n\n"
        + CONFIG_KEYS_HINT
    ),
)
def config_get(
    key: str = typer.Argument(
        ...,
        help="Config key to read",
        autocompletion=_complete_config_key,
    ),
):
    cfg = read_global_config()
    value = cfg.get(key.upper().replace("-", "_"), "")
    console.print(_mask(value) if "KEY" in key.upper() else value or "(unset)")


@config_app.command("show")
def config_show():
    """Show all config (API keys masked)."""
    cfg = read_global_config()
    console.print(f"[bold]Config file:[/bold] {global_config_path()}")
    console.print()
    if not cfg:
        console.print("[yellow](empty)[/yellow]")
        return
    for k, v in cfg.items():
        display = _mask(v) if "KEY" in k else v
        console.print(f"  [cyan]{k}[/cyan] = {display}")


@config_app.command("path")
def config_path():
    """Print the config file path."""
    console.print(str(global_config_path()))


@config_app.command(
    "unset",
    help="Remove a config value.\n\n" + CONFIG_KEYS_HINT,
)
def config_unset(
    key: str = typer.Argument(
        ...,
        help="Config key to remove",
        autocompletion=_complete_config_key,
    ),
):
    cfg = read_global_config()
    key = key.upper().replace("-", "_")
    if key in cfg:
        del cfg[key]
        write_global_config(cfg)
        console.print(f"[green]Removed {key}[/green]")
    else:
        console.print(f"[yellow]{key} not set[/yellow]")


# ---- Shell completion ----

PROG_NAME = "racks-thumbnail"
COMPLETION_VAR = f"_{PROG_NAME.replace('-', '_').upper()}_COMPLETE"
COMPLETION_MARKER_BEGIN = "# >>> racks-thumbnail completion >>>"
COMPLETION_MARKER_END = "# <<< racks-thumbnail completion <<<"

SHELL_RC = {
    "zsh": "~/.zshrc",
    "bash": "~/.bashrc",
    "fish": "~/.config/fish/completions/racks-thumbnail.fish",
}


def _detect_shell() -> str | None:
    sh = os.environ.get("SHELL", "")
    name = Path(sh).name.lower()
    return name if name in SHELL_RC else None


def _get_typer_script(shell: str) -> str:
    from typer._completion_shared import get_completion_script
    return get_completion_script(prog_name=PROG_NAME, complete_var=COMPLETION_VAR, shell=shell)


def _is_persistent_install() -> bool:
    """Auto-install completion only for permanent installs (uv tool / pip --user).
    Skip for ephemeral envs: uvx cache, dev .venv, system temp.
    Uses sys.prefix (venv root) — does not follow symlinks."""
    s = sys.prefix.lower()
    # Persistent: uv tool installs land here
    if "/.local/share/uv/tools/" in s:
        return True
    if "/.local/" in s and "/site-packages" not in s:
        # pip install --user style
        return True
    # Ephemeral: skip
    return False


def _install_completion_to_rc(shell: str) -> tuple[bool, Path]:
    """Append completion block to rc file. Returns (newly_installed, rc_path)."""
    rc_path = Path(os.path.expanduser(SHELL_RC[shell]))
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = rc_path.read_text() if rc_path.exists() else ""
    if COMPLETION_MARKER_BEGIN in existing or COMPLETION_VAR in existing:
        return False, rc_path
    script = _get_typer_script(shell)
    block = f"\n{COMPLETION_MARKER_BEGIN}\n{script}\n{COMPLETION_MARKER_END}\n"
    with rc_path.open("a") as f:
        f.write(block)
    return True, rc_path


def _uninstall_completion_from_rc(shell: str) -> tuple[bool, Path]:
    """Remove completion block from rc file. Returns (was_present, rc_path)."""
    rc_path = Path(os.path.expanduser(SHELL_RC[shell]))
    if not rc_path.exists():
        return False, rc_path
    text = rc_path.read_text()
    if COMPLETION_MARKER_BEGIN not in text:
        return False, rc_path
    lines = text.splitlines()
    out, skip = [], False
    for ln in lines:
        if ln.strip() == COMPLETION_MARKER_BEGIN:
            skip = True
            continue
        if ln.strip() == COMPLETION_MARKER_END:
            skip = False
            continue
        if not skip:
            out.append(ln)
    rc_path.write_text("\n".join(out).rstrip() + "\n")
    return True, rc_path


def _maybe_auto_install_completion() -> None:
    """First-run silent auto-install. No-op if already installed, ephemeral env, or unsupported shell."""
    marker = global_config_dir() / ".completion-installed"
    if marker.exists():
        return
    if not _is_persistent_install():
        return
    shell = _detect_shell()
    if not shell:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return
    try:
        installed, rc_path = _install_completion_to_rc(shell)
        if installed:
            console.print(f"[dim]✓ Tab completion installed → {rc_path}. Open a new shell to use.[/dim]")
    except Exception:
        pass
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


@app.callback()
def _root_callback():
    """Runs before any subcommand — triggers first-run auto-install."""
    _maybe_auto_install_completion()


@app.command("completion")
def completion(
    shell: str = typer.Option(None, help="Shell: zsh, bash, fish (auto-detected from $SHELL)"),
    install: bool = typer.Option(False, "--install", help="Manually (re)install completion"),
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove completion from your rc file"),
    show_script: bool = typer.Option(False, "--show-script", help="Print the completion script"),
):
    """Manage tab completion (auto-installed on first run)."""
    shell = shell or _detect_shell()
    if not shell:
        console.print("[red]Could not detect shell from $SHELL.[/red] Pass --shell zsh|bash|fish.")
        raise typer.Exit(1)
    if shell not in SHELL_RC:
        console.print(f"[red]Unsupported shell: {shell}.[/red] Use zsh, bash, or fish.")
        raise typer.Exit(1)

    if show_script:
        console.print(_get_typer_script(shell))
        return

    if uninstall:
        was_present, rc_path = _uninstall_completion_from_rc(shell)
        marker = global_config_dir() / ".completion-installed"
        if marker.exists():
            marker.unlink()
        if was_present:
            console.print(f"[green]Removed completion from {rc_path}[/green]")
        else:
            console.print(f"[yellow]No completion block found in {rc_path}[/yellow]")
        return

    if install:
        installed, rc_path = _install_completion_to_rc(shell)
        marker = global_config_dir() / ".completion-installed"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        if installed:
            console.print(f"[green]Installed completion → {rc_path}[/green]")
        else:
            console.print(f"[yellow]Already installed in {rc_path}[/yellow]")
        console.print(f"\nReload shell:  [cyan]source {rc_path}[/cyan]  (or open a new terminal)")
        return

    # Status
    rc_path = Path(os.path.expanduser(SHELL_RC[shell]))
    existing = rc_path.read_text() if rc_path.exists() else ""
    is_installed = COMPLETION_MARKER_BEGIN in existing
    console.print(f"Shell: [bold]{shell}[/bold]")
    console.print(f"RC file: [bold]{rc_path}[/bold]")
    console.print(f"Status: {'[green]installed[/green]' if is_installed else '[yellow]not installed[/yellow]'}\n")
    console.print(f"  [cyan]racks-thumbnail completion --install[/cyan]    Install / reinstall")
    console.print(f"  [cyan]racks-thumbnail completion --uninstall[/cyan]  Remove")
    console.print(f"  [cyan]racks-thumbnail completion --show-script[/cyan] Print raw script")


if __name__ == "__main__":
    app()
