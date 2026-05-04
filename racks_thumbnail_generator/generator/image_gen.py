import io
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types as genai_types


DEFAULT_MODEL = "gemini-2.5-flash-image"


def generate_thumbnail(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "9:16",
    reference_images: list[Path] | None = None,
    output_path: Path | None = None,
) -> Image.Image:
    client = genai.Client(api_key=api_key)

    parts: list[genai_types.Part] = []

    if reference_images:
        for ref_path in reference_images:
            mime = "image/png" if ref_path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(
                genai_types.Part.from_bytes(data=ref_path.read_bytes(), mime_type=mime)
            )

    parts.append(genai_types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=model,
        contents=[genai_types.Content(parts=parts)],
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=genai_types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img = Image.open(io.BytesIO(part.inline_data.data))
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, quality=95)
            return img

    raise RuntimeError("No image returned from Gemini. Response: " + (response.text or ""))
