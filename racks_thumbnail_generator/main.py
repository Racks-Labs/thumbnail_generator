import os
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from racks_thumbnail_generator.config import (
    get_settings,
    TranscriberBackend,
    global_config_path,
    read_global_config,
    write_global_config,
    set_global_config_value,
    CONFIG_KEYS,
)
from racks_thumbnail_generator.pipeline.audio import extract_audio
from racks_thumbnail_generator.pipeline.transcriber import get_transcriber
from racks_thumbnail_generator.pipeline.topic_extractor import extract_topic
from racks_thumbnail_generator.generator.prompt_builder import build_prompt
from racks_thumbnail_generator.generator.image_gen import generate_thumbnail, DEFAULT_MODEL
from racks_thumbnail_generator.compositor.text_overlay import overlay_text

APP_HELP = """\
Generate cinematic thumbnails for Racks reels from a video file.

Pipeline: video → audio → transcription → topic → AI scene (Nanobanana) → text overlay (Pillow + Inter + brand color).

[bold]Quick start[/bold]
  [cyan]racks-thumbnail config set google-api-key TU_KEY[/cyan]    First-time setup
  [cyan]racks-thumbnail generate video.mp4[/cyan]                  Generate thumbnail (output → ./output/)
  [cyan]racks-thumbnail generate video.mp4 --accent-color "#FF6B00"[/cyan]   Custom color
  [cyan]racks-thumbnail test-transcribe audio.mp3[/cyan]           Test transcription only
  [cyan]racks-thumbnail config show[/cyan]                         View saved config

[bold]Tab completion[/bold]
  [cyan]racks-thumbnail completion[/cyan]                          Show snippet for your shell
  [cyan]racks-thumbnail completion --install[/cyan]                Auto-add to ~/.zshrc / ~/.bashrc

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


@app.command()
def generate(
    video: Path = typer.Argument(..., help="Path to video file (.mp4)"),
    template: str = typer.Option("default", help="Template name from templates/"),
    output: Path = typer.Option(None, help="Output directory"),
    accent_color: str = typer.Option(None, help="Accent color hex (default: #BE190F)"),
    transcriber: str = typer.Option(None, help="Transcription backend: gemini or whisper"),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemini image model"),
):
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

    template_path = settings.templates_dir / f"{template}.yaml"
    if not template_path.exists():
        console.print(f"[red]Template not found: {template_path}[/red]")
        raise typer.Exit(1)

    # Step 1: Extract audio
    console.print("[bold]1/4[/bold] Extracting audio...")
    audio_path = extract_audio(video)

    # Step 2: Transcribe
    console.print(f"[bold]2/4[/bold] Transcribing with {settings.transcriber.value}...")
    transcriber_impl = get_transcriber(
        backend=settings.transcriber.value,
        google_api_key=settings.google_api_key,
        openai_api_key=settings.openai_api_key,
    )
    transcript = transcriber_impl.transcribe(audio_path)
    console.print(Panel(transcript[:300] + ("..." if len(transcript) > 300 else ""), title="Transcripción"))

    # Step 3: Extract topic + visual concept
    console.print("[bold]3/4[/bold] Extracting topic + visual concept...")
    content = extract_topic(transcript, settings.google_api_key)
    console.print(f"  Headline: [bold red]{content.headline}[/bold red]")
    console.print(f"  Accent word: [bold]{content.headline_accent_word}[/bold]")
    console.print(f"  Tono: {content.tono}")
    console.print(Panel(content.concepto_visual, title="Concepto visual"))

    # Step 4a: Generate background scene with Nanobanana
    console.print("[bold]4/4[/bold] Generating background scene with Nanobanana...")
    prompt = build_prompt(template_path, content, settings.accent_color)

    references = []
    user_refs = Path.cwd() / "references"
    if user_refs.exists():
        references = [
            r for r in user_refs.glob("*")
            if r.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_path = settings.output_dir / f"_bg_{timestamp}.png"

    background = generate_thumbnail(
        prompt=prompt,
        api_key=settings.google_api_key,
        model=model,
        reference_images=references or None,
        output_path=background_path,
    )

    # Step 4b: Composite text overlay with Pillow (exact font + exact color)
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

    audio_path.unlink(missing_ok=True)


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

@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help=f"One of: {sorted(CONFIG_KEYS)}"),
    value: str = typer.Argument(..., help="Value to store"),
):
    """Set a config value (saved to ~/.config/thumbnail-generator/.env)."""
    try:
        path = set_global_config_value(key, value)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Saved {key.upper()} → {path}[/green]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Config key to read")):
    """Read a config value."""
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


@config_app.command("unset")
def config_unset(key: str = typer.Argument(..., help="Config key to remove")):
    """Remove a config value."""
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


@app.command("completion")
def completion(
    shell: str = typer.Option(None, help="Shell: zsh, bash, fish (auto-detected from $SHELL)"),
    install: bool = typer.Option(False, "--install", help="Auto-append to your shell rc file"),
    show_script: bool = typer.Option(False, "--show-script", help="Print the full completion script"),
):
    """Setup tab completion for racks-thumbnail."""
    shell = shell or _detect_shell()
    if not shell:
        console.print("[red]Could not detect shell from $SHELL.[/red] Pass --shell zsh|bash|fish.")
        raise typer.Exit(1)
    if shell not in SHELL_RC:
        console.print(f"[red]Unsupported shell: {shell}.[/red] Use zsh, bash, or fish.")
        raise typer.Exit(1)

    script = _get_typer_script(shell)

    if show_script:
        console.print(script)
        return

    rc_path = Path(os.path.expanduser(SHELL_RC[shell]))

    if install:
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        existing = rc_path.read_text() if rc_path.exists() else ""
        marker = f"# racks-thumbnail completion ({shell})"
        if marker in existing or COMPLETION_VAR in existing:
            console.print(f"[yellow]Completion already installed in {rc_path}[/yellow]")
        else:
            with rc_path.open("a") as f:
                f.write(f"\n{marker}\n{script}\n")
            console.print(f"[green]Installed completion → {rc_path}[/green]")
        console.print(f"\nReload shell:  [cyan]source {rc_path}[/cyan]  (or open a new terminal)")
    else:
        console.print(f"Detected shell: [bold]{shell}[/bold]")
        console.print(f"Will install to: [bold]{SHELL_RC[shell]}[/bold]\n")
        console.print(f"  [cyan]racks-thumbnail completion --install[/cyan]              Auto-add to rc file")
        console.print(f"  [cyan]racks-thumbnail completion --show-script[/cyan]          Print the script")
        console.print(f"\nThen open a new shell to enable.")


if __name__ == "__main__":
    app()
