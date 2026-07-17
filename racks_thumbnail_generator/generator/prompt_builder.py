from pathlib import Path

import yaml

from racks_thumbnail_generator.pipeline.topic_extractor import ThumbnailContent


def load_template(template_path: Path) -> dict:
    with open(template_path) as f:
        return yaml.safe_load(f)


def _substitute(obj, variables: dict):
    if isinstance(obj, str):
        for key, val in variables.items():
            obj = obj.replace(f"{{{{{key}}}}}", str(val))
        return obj
    if isinstance(obj, dict):
        return {k: _substitute(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(item, variables) for item in obj]
    return obj


def build_prompt(
    template_path: Path,
    content: ThumbnailContent,
    accent_color: str = "#BE190F",
) -> str:
    template = load_template(template_path)

    variables = {
        "concepto_visual": content.concepto_visual,
        "accent_color": accent_color,
    }
    template = _substitute(template, variables)

    fmt = template["format"]
    style = template["style"]
    scene = template["scene"]
    rules = template.get("rules", [])

    parts = [
        f"Generate a thumbnail background image, {fmt['orientation']} orientation, {fmt['width']}x{fmt['height']} pixels.",
        "",
        "=== STYLE ===",
        f"Aesthetic: {style['aesthetic']}",
        "",
        "=== SCENE ===",
        scene["description"],
        "",
        scene["guidance"],
        "",
        "=== RULES (CRITICAL) ===",
    ]

    for rule in rules:
        parts.append(f"- {rule}")

    return "\n".join(parts)


def describe_references(references) -> str:
    """Prompt block describing each reference image and its role, in the same
    order the images are attached to the request. `references` is a list of
    `spec.ReferenceImageSpec`."""
    if not references:
        return ""
    lines = [
        "=== REFERENCE IMAGES ===",
        "The images attached before this prompt are references, in this order:",
    ]
    for i, ref in enumerate(references, start=1):
        role = ref.role or "visual reference — use it as provided"
        lines.append(f"- Image {i}: {role}")
    lines.append(
        "Use each reference exactly for its stated role. Keep identities, shapes "
        "and colors faithful to the references. Do not invent replacements for them."
    )
    return "\n".join(lines)
