from typing import Literal

from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types


SubjectFocus = Literal["brand_product", "person", "object"]


class ThumbnailContent(BaseModel):
    headline: str
    headline_accent_word: str
    subtexto: str
    keywords: list[str]
    tono: str
    subject_focus: SubjectFocus
    concepto_visual: str


EXTRACTION_PROMPT = """\
Analyze this transcript from a Spanish-language reel and design a thumbnail concept.

Transcript:
{transcript}

Return a JSON object with:

- headline: A short impactful phrase in Spanish (3-6 words MAX) that communicates the SPECIFIC VALUE the viewer gets by watching this reel.

  CRITICAL — the headline must:
  * Be CONCRETE — name the specific benefit, tool, brand, or outcome
  * Reference WHAT the viewer learns / saves / gets — money, time, a tool name, a method
  * NOT be a vague slogan ("Domina tu IA") — useless
  * Sound like a Spanish-speaking creator's editorial title, not corporate
  * NO punctuation, NO colons, NO subtitles inside

  Good examples:
  - "7 NUEVAS IAS DE GOOGLE"
  - "CLAUDE RESPONDE COMO CAVERNICOLA"
  - "AHORRA 70% EN TOKENS DE IA"
  - "PAGA LA MITAD POR CLAUDE"

- headline_accent_word: The single most important word from the headline that gets a colored accent block behind it. Must appear EXACTLY in the headline. Pick the word that carries the value (the brand name, the number, a key noun).

- subtexto: Short complementary phrase in Spanish (max 8 words). Optional context.

- keywords: 3-5 main topics/themes.

- tono: provocador | informativo | motivacional | educativo | controversial.

- subject_focus: Decide what the SCENE is centered around. Pick ONE:
  * "brand_product" — when the reel is ABOUT a specific named brand, product, app, model, or tool (e.g. Google, Claude, ChatGPT, Notion, MacBook, Tesla, iPhone, Sora). The scene must hero THE BRAND/PRODUCT itself.
  * "person" — when the reel is about a human topic, mindset, advice, lifestyle, money habits, how-tos with no specific tool as the hero. The scene must hero a person doing something representative.
  * "object" — when the reel is about a concept that's best symbolized by a single non-brand real object (a book, money, a clock). Rare — use only if neither person nor brand fits.

- concepto_visual: Cinematic scene description in English for an AI image generator. The composition depends on subject_focus.

  COMMON HARD CONSTRAINTS — banned in EVERY scene type:
  * NO holograms, NO neon, NO glowing floating screens, NO futuristic interfaces, NO blue/purple sci-fi lighting, NO server rooms with rocks
  * NO fictional / made-up / symbolic objects (no "AI token coins", no "energy orbs", no abstract symbols)
  * NO clutter — strip out background props
  * Photorealistic, cinematic, shallow depth of field, premium editorial / commercial photography quality
  * HIGH CONTRAST CHIAROSCURO lighting — single hard key light, deep dark shadows
  * NO TEXT in the image (text gets composited later)

  ============================================================
  IF subject_focus == "brand_product":
  ============================================================
  Hero the BRAND or PRODUCT itself. NO person in the frame. Use creative product / hero photography.

  STRUCTURE:
  * MAIN subject: the brand's recognizable mark or product, rendered in a real physical / 3D form. NOT a flat 2D logo on a screen — physical mark sitting in the scene. Examples:
    - Google: a tangible 3D Google "G" letter (full color: red/yellow/green/blue) as a physical sculpture — emerging from a kraft cardboard box, on a black pedestal, on a wooden table, etc.
    - Claude: a tangible "Claude" wordmark sculpted in matte material, or the official orange/cream Claude visual identity rendered as a physical object
    - ChatGPT: the OpenAI swirl logo as a 3D rendered metal/glass object
    - Apple: a real MacBook or iPhone shown front-and-center, beautifully lit
    - Notion: a 3D Notion "N" sculpture
    - Tesla: an actual Tesla car or the T logo
  * SECONDARY element (optional, only if it adds to the message): a real physical container or context — a kraft cardboard box (often with a stamp like "GRATIS", "NEW", "FREE" if relevant — but the stamp text will be added separately, just describe the box as having a printable stamp area), gift wrapping, a wooden crate, a museum-style pedestal, a Polaroid-style frame.
  * Setting: dark studio backdrop OR softly out-of-focus modern office bokeh OR clean dark wooden surface. Empty negative space around the hero. The hero MUST dominate the frame.
  * Lighting: dramatic single key light highlighting the brand mark, with light leaks / colored rim light if it matches the brand colors (e.g. Google = subtle multi-color rainbow light leak in the background; Apple = clean white-on-white; Claude = warm cream/orange tones).
  * Color palette: dominated by the BRAND'S OWN color identity (Google = full RGB color pop; Claude = warm cream/orange; OpenAI = teal/black; Apple = silver/white). The rest of the frame stays dark/neutral.
  * Style: premium product photography, commercial advertising still, 3D render quality, hyperrealistic.

  Good "brand_product" examples:
  - GOOGLE: "A tangible 3D rendered Google 'G' letter (full color: red, yellow, green, blue) emerges dramatically from an open kraft cardboard box sitting on a dark wooden table. The box has a prominent flat blank area on its front face suitable for a printed stamp. Soft polystyrene packing peanuts spill around the G. Subtle multicolor rainbow light leak from the right side, deep dark out-of-focus office bokeh in the background. Dramatic key light from above-right, deep shadows on the box. Premium commercial product photography, hyperrealistic, cinematic, shallow depth of field."
  - CLAUDE: "A 3D rendered matte ceramic 'Claude' wordmark sculpture in warm cream and burnt orange tones, sitting on a polished dark wooden table against a near-black blurred studio background. A single warm tungsten key light from the upper left rakes across the lettering creating soft warm shadows. Premium commercial 3D product still, hyperrealistic, cinematic, editorial."
  - APPLE: "A pristine open MacBook Pro centered on a clean dark slate desk in a darkened minimalist studio. The screen glows softly with a clean abstract gradient. Hard key light from the upper left, deep shadows on the right. No other objects. Premium commercial product photography, hyperrealistic, A24 film still."

  ============================================================
  IF subject_focus == "person":
  ============================================================
  Hero a clear character + ONE real recognizable secondary object. (Current default.)

  STRUCTURE:
  * MAIN subject: clear character with distinctive look. If the message references "caveman" / "primitive" → render an actual caveman with shaggy beard and fur tunic. Doing a modern action — the contrast is the point.
  * SECONDARY element: ONE real instantly-readable object — fistful of dollar bills, open laptop with a visible brand wordmark on screen, coffee mug, clock, calendar, receipt, credit card, book.
  * Setting: minimal real environment, dark studio, dark wood desk, plain wall.
  * Lighting: single hard key from one side, deep shadows on the other.
  * Color palette: dark/restrained base, ONE color pop from the secondary object.

  Good "person" examples:
  - WARM (intimate/human): "A scruffy caveman with shaggy hair and beard wearing a torn animal-fur tunic sits at a dark wooden desk against a near-black wall, holding a thick stack of crumpled green dollar bills in one hand. A hard warm tungsten key light from the left carves out his face and the cash; right side in deep shadow. Dark moody palette with the green of the bills as the only color pop."
  - COOL (serious/tech): "A man in a sharp dark suit sits at a clean concrete desk in a near-black studio, sliding a single bright red passport across the table toward the camera. Cool blue daylight from upper left, hard shadows. Desaturated cool grey palette with the red passport as the only color pop."

  ============================================================
  IF subject_focus == "object":
  ============================================================
  Hero a single iconic object. No person.

  STRUCTURE:
  * MAIN subject: ONE iconic real-world object centered in frame, presented as a still life — an open antique book, a vintage clock, a stack of cash, a brass key.
  * Setting: dark studio, polished dark wooden surface, plain backdrop.
  * Lighting: dramatic single key, deep shadows.
  * Color palette: dark/restrained, the natural color of the object as the only pop.

  Good "object" example: "A single weathered brass key lies flat on a polished dark walnut surface against a near-black backdrop. A single hard warm key light from the upper left catches the worn metal, creating a long deep shadow trailing right. No other objects. Cinematic still life, hyperrealistic, A24 aesthetic."

  ============================================================
  Bad examples (DO NOT do this regardless of focus):
  ============================================================
  - "...holds a small symbolic stack of metallic AI token coins" (fictional object)
  - "...glowing AI interface" / "holographic display" (sci-fi cliché)
  - "...sits in a brightly lit modern office with several monitors and plants" (cluttered)
  - "A muscular primal caveman dominates over..." (theatrical)
  - For Google reel: "A man in a suit holds a black folder" (lost the brand entirely — should hero the Google G logo)
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
