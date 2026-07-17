"""Orchestration for JSON-config-driven thumbnail generation.

Flow: resolve title/prompt (from the spec, falling back to transcript topic
extraction) → AI scene with reference images (Nanobanana) → deterministic
overlays (graphic elements + title) with Pillow.
"""

from datetime import datetime
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.panel import Panel

from racks_thumbnail_generator.compositor.composer import paste_elements
from racks_thumbnail_generator.compositor.text_overlay import overlay_title
from racks_thumbnail_generator.generator.image_gen import DEFAULT_MODEL, generate_thumbnail
from racks_thumbnail_generator.generator.prompt_builder import build_prompt, describe_references
from racks_thumbnail_generator.spec import SpecError, ThumbnailSpec, TitleSpec

console = Console()


def run_from_spec(
    spec: ThumbnailSpec,
    settings,
    transcript: str | None = None,
    model: str = DEFAULT_MODEL,
    show_prompt: bool = False,
) -> Path | None:
    """Render a thumbnail from a validated ThumbnailSpec.

    `transcript` is only required when the spec omits `title.text` and/or
    `generation.prompt` (those then come from topic extraction).
    Returns the output path, or None when `show_prompt` short-circuits.
    """
    title: TitleSpec = spec.title.model_copy(deep=True)

    # ---- Resolve content that may come from the transcript ----
    needs_topic = title.text is None or spec.generation.prompt is None
    content = None
    if needs_topic:
        if not transcript:
            missing = []
            if title.text is None:
                missing.append("title.text")
            if spec.generation.prompt is None:
                missing.append("generation.prompt")
            raise SpecError(
                f"The config omits {' and '.join(missing)}, which requires a video/script "
                "transcript to auto-generate. Provide them in the JSON or run with a "
                "video (generate --config) / script (--script)."
            )
        from racks_thumbnail_generator.pipeline.topic_extractor import extract_topic

        console.print("Extracting topic + visual concept from transcript...")
        content = extract_topic(transcript, settings.google_api_key)
        if title.text is None:
            title.text = content.headline
            if not title.accent.words and content.headline_accent_word:
                title.accent.words = [content.headline_accent_word]
        console.print(f"  Headline: [bold]{title.text}[/bold]")

    # ---- Build the generation prompt ----
    if spec.generation.prompt is not None:
        prompt = spec.generation.prompt
    else:
        template_path = settings.templates_dir / f"{spec.generation.template}.yaml"
        if not template_path.exists():
            raise SpecError(f"Template not found: {template_path}")
        prompt = build_prompt(template_path, content, title.accent.color)

    refs_block = describe_references(spec.generation.references)
    if refs_block:
        prompt = f"{prompt}\n\n{refs_block}"

    reference_paths = [Path(r.path) for r in spec.generation.references]

    if show_prompt:
        console.print(Panel(prompt, title="Final Nano Banana prompt", border_style="cyan"))
        if reference_paths:
            console.print(f"[dim]Reference images: {[str(p) for p in reference_paths]}[/dim]")
        console.print("[yellow]--show-prompt: skipping image generation.[/yellow]")
        return None

    # ---- AI scene generation ----
    console.print("Generating scene with Nanobanana...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_path = settings.output_dir / f"_bg_{timestamp}.png"

    image = generate_thumbnail(
        prompt=prompt,
        api_key=settings.google_api_key,
        model=model,
        aspect_ratio=spec.canvas.aspect_ratio(),
        reference_images=reference_paths or None,
        output_path=background_path,
    )

    if image.size != (spec.canvas.width, spec.canvas.height):
        image = image.resize((spec.canvas.width, spec.canvas.height), Image.LANCZOS)

    # ---- Deterministic overlays: elements first, then title ----
    if spec.elements:
        console.print(f"  Compositing {len(spec.elements)} graphic element(s)...")
        image = paste_elements(image, spec.elements)

    console.print("  Compositing title overlay...")
    final = overlay_title(
        image=image,
        title=title,
        fonts_dir=settings.fonts_dir,
        branding=spec.branding,
    )

    output_path = (
        Path(spec.output) if spec.output
        else settings.output_dir / f"thumbnail_{timestamp}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path, quality=95)

    console.print(f"\n[bold green]Thumbnail saved → {output_path}[/bold green]")
    console.print(f"  Size: {final.size[0]}x{final.size[1]}")
    return output_path
