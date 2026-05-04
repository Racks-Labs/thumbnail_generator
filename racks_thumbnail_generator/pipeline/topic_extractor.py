from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types


class ThumbnailContent(BaseModel):
    headline: str
    headline_accent_word: str
    subtexto: str
    keywords: list[str]
    tono: str
    concepto_visual: str


EXTRACTION_PROMPT = """\
Analyze this transcript from a Spanish-language reel and design a thumbnail concept.

Transcript:
{transcript}

Return a JSON object with:
- headline: A short impactful phrase in Spanish (3-6 words MAX) that communicates the SPECIFIC VALUE the viewer gets by watching this reel. Why should someone stop scrolling? Answer that in the headline.

  CRITICAL — the headline must:
  * Be CONCRETE — name the specific benefit, tool, or outcome (not vague motivation)
  * Reference WHAT the viewer learns / saves / gets — money, time, a tool name, a method
  * NOT be a vague slogan ("Domina tu IA", "Cambia tu vida") — those are useless
  * Sound like a Spanish-speaking creator's editorial title, not corporate
  * NO punctuation, NO colons, NO subtitles inside

  Good examples (concrete value, specific tool/outcome):
  - "CLAUDE RESPONDE COMO UN CAVERNICOLA"
  - "AHORRA 70% EN TOKENS DE IA"
  - "PAGA LA MITAD POR CLAUDE"
  - "EL HACK GRATIS PARA CLAUDE"

  Bad examples (vague, no value):
  - "DOMINA TU IA AHORA" (vague — what does that even mean?)
  - "DEJA DE PAGAR DE MAS" (no specific tool/method)
  - "LA SOLUCION DEFINITIVA" (says nothing)

- headline_accent_word: The single most important / impactful word from the headline that should be highlighted with a red accent block behind it. Must be a word that appears EXACTLY in the headline string. Pick the word that carries the value (the brand, the number, the verb).
- subtexto: Short complementary phrase (max 8 words) in Spanish. Optional context.
- keywords: List of 3-5 main topics/themes.
- tono: One of: provocador, informativo, motivacional, educativo, controversial.
- concepto_visual: Cinematic scene description in English for an AI image generator. EXPLICIT but SUBTLE visual metaphor that reads in 1 second. Premium editorial photography with HIGH CONTRAST lighting.

  HARD CONSTRAINTS — banned elements:
  * NO holograms, NO neon, NO glowing floating screens, NO futuristic interfaces, NO blue/purple sci-fi lighting, NO server rooms with rocks
  * NO fictional / made-up / symbolic objects (no "AI token coins", no "energy orbs", no abstract symbols, no metaphysical floating elements)
  * NO weird stuff in hands or floating around — every object must be something a normal person would instantly recognize from real life
  * NO clutter — the scene must have ONE main subject + AT MOST ONE secondary real object. No third elements competing for attention.

  STRUCTURE (instant readability):
  * MAIN subject: clear character with a distinctive recognizable look. If the message references "caveman" / "primitive" — render an actual caveman: shaggy beard, fur tunic, raw look. Doing a modern action. The contrast IS the point.
  * SECONDARY element (single real object, instantly readable): pick ONE that screams the message — a fistful of dollar bills, an open laptop, a coffee mug, a clock, a calendar, a printed receipt, a credit card, a book. NO invented or symbolic objects. The viewer must recognize it in 0.5 seconds.
  * Setting: plausible real environment, MINIMAL. Dark studio, dark wood desk, plain wall, simple kitchen. Strip out background props. Empty negative space is good — it adds drama.
  * Lighting: HIGH CONTRAST CHIAROSCURO. Single hard key light from one side. Deep dark shadows on the opposite side. The subject must POP from a darker background. Think Caravaggio meets editorial portrait. Light TEMPERATURE adapts to theme — warm tungsten for cozy/intimate topics, cool daylight for serious/clinical topics, neutral for editorial.
  * Color palette: dark, rich, restrained as a BASE (mostly black/deep shadows). The dominant color temperature ADAPTS to the theme — earth tones for organic/lifestyle, cool greys/blues for tech/finance/serious, desaturated for melancholic, warm for human/intimate. The ONE color POP must come from the secondary object (green bills, red book, blue notebook, yellow legal pad, etc.) — that color pop is the only saturation in the frame. Avoid muddy mid-tones.
  * Style: photorealistic, cinematic, shallow depth of field, A24 film still, high contrast editorial portrait.

  Good examples (explicit + subtle + high contrast — note palette adapts to theme):
  - WARM example (intimate/human topic): "A scruffy caveman with shaggy hair and beard wearing a torn animal-fur tunic sits at a dark wooden desk against a near-black wall, holding a thick stack of crumpled green dollar bills in one hand. A single hard warm tungsten key light from the left carves out his face and the cash; the right side of the frame is in deep shadow. Dark moody palette with the green of the bills as the only color pop."
  - COOL example (serious/tech topic): "A man in a sharp dark suit sits at a clean concrete desk in a near-black studio, sliding a single bright red passport across the table toward the camera. Cool blue daylight from the upper left, hard shadows. Desaturated cool grey palette with the red passport as the only color pop. Photorealistic editorial portrait, high contrast chiaroscuro."
  - NEUTRAL example (editorial/clinical): "A woman in a plain black turtleneck holds an open yellow legal notepad covered in handwritten notes, against a flat dark grey wall. Single neutral key light from above. Mostly black and grey palette, the yellow notepad as the only color pop. Photorealistic editorial portrait, A24 aesthetic."

  Bad examples (DO NOT do this):
  - "...holds a small symbolic stack of metallic AI token coins" (FICTIONAL OBJECT — banned)
  - "...glowing AI interface" / "holographic display" (sci-fi cliché — banned)
  - "...sits in a brightly lit modern office with several monitors and plants" (cluttered, no contrast, no drama)
  - "A muscular primal caveman dominates over..." (theatrical, vague)
  - "A bearded man in a tan shirt sits calmly at a desk" (lost the caveman, no metaphor)
"""


def extract_topic(transcript: str, api_key: str) -> ThumbnailContent:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=EXTRACTION_PROMPT.format(transcript=transcript),
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ThumbnailContent,
        ),
    )
    return ThumbnailContent.model_validate_json(response.text)
