from typing import Literal

from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types


SubjectFocus = Literal["brand_product", "theme_artifact", "person", "object"]


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

- subject_focus: Decide what the SCENE is centered around. Pick ONE.

  =====================================================================
  DECISION TREE — apply IN THIS ORDER, stop at first match:
  =====================================================================

  STEP 1 — Famous brand test.
    Does the reel center on a brand the average non-tech Spanish Instagram
    user instantly recognizes from its logo alone (no text), in <0.5s?
    Examples that PASS: Google, Apple, iPhone, MacBook, ChatGPT, OpenAI,
    YouTube, Instagram, TikTok, WhatsApp, Tesla, Spotify, Netflix,
    Microsoft, Windows, Amazon, Meta, Facebook, X / Twitter, Uber, PayPal, Visa.
    → If YES → "brand_product".
    → If NO (Claude, Cursor, Notion, Linear, Sora, Suno, Runway, n8n,
      Zapier, Replit, v0, any niche/new/beta tool) → continue to STEP 2.

  STEP 2 — Theme artifact test.
    Does the reel's TOPIC have iconic visual artifacts that VISUALLY
    represent the topic better than any human face or single still-life
    object would? An "artifact" = the tangible OUTPUT or TOOL of the topic
    itself, instantly readable as the subject matter.

    Examples by topic → artifact:
      - UI / UX design / interfaces      → mockup screens, wireframes,
                                            Figma-style layouts, app screens
      - Code / programming / dev tools   → code editor close-up with
                                            colored syntax (no readable text),
                                            terminal close-up with ASCII shapes
      - Music / audio / production       → studio monitors, mixing console,
                                            synthesizer keys, audio waveform
                                            on a real screen, analog tape reel
      - Finance / stocks / trading       → printed candlestick chart on
                                            paper, ledger pages with
                                            handwritten figures, stock
                                            tickers on a vintage screen
      - Cooking / food / recipes         → plated dish overhead, raw
                                            ingredients arranged, knife on
                                            cutting board
      - Fashion / style                  → garment hanging, shoe close-up,
                                            fabric folds, accessory on stand
      - Architecture / interior design   → architectural blueprints, scale
                                            model, miniature room set
      - Photography / video editing      → camera body close-up, film strips,
                                            color grading swatches
      - Writing / content                → typewriter, manuscript with red
                                            edits, opened journal
      - Fitness / workout                → kettlebell, gym bench, jump rope
      - Beauty / skincare                → product bottles, brushes, palette
      - Crypto / blockchain              → physical metal coin (Bitcoin),
                                            paper wallet print
      - 3D / motion / VFX                → 3D render of a wireframe object,
                                            grease-pencil storyboard frames
      - Productivity / planning          → analog planner, pen, sticky notes
      - Marketing / ads                  → printed billboard miniature,
                                            tear-sheet magazine ad

    Trigger phrase test: if the headline or transcript implies the topic
    PRODUCES something visual ("la IA diseña interfaces" → interfaces;
    "código limpio" → code; "edita videos así" → film/timeline), the
    artifact wins over a person.
    → If YES → "theme_artifact".
    → If NO (topic is purely human: mindset, advice, money habits,
      relationships, motivation, biographical, opinion takes) → continue
      to STEP 3.

  STEP 3 — Person test.
    Topic is fundamentally about a HUMAN doing/feeling something (advice,
    mindset, lifestyle, identity, story). The substance is the human angle.
    → "person".
    The secondary object MUST come from the substance of the reel and be
    HIGHLY topic-specific. Forbidden default fallbacks: generic open
    laptop, generic coffee mug, generic notebook (use these ONLY if
    they're literally what the reel is about).

  STEP 4 — Pure object fallback.
    None of the above fit and the topic is best shown as a single iconic
    still-life (a key for "freedom", a hourglass for "time").
    → "object". Rare.

  =====================================================================
  When in doubt between brand_product and theme_artifact → theme_artifact.
  When in doubt between person and theme_artifact → theme_artifact (a
  topic-specific artifact almost always beats a generic person scene).
  =====================================================================

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
  IF subject_focus == "theme_artifact":
  ============================================================
  Hero the VISUAL OUTPUT / TOOL of the topic itself. NO person in frame.
  The artifact must instantly communicate the subject matter — viewer
  reads the topic from the object alone in <1s.

  STRUCTURE:
  * MAIN subject: the topic's tangible artifact, rendered as a real
    physical / on-screen object. Pick the artifact from STEP 2's list
    (UI mockups, code editor, mixing console, candlestick chart on
    paper, plated dish, blueprints, camera body, etc.).
  * Multiple instances OK: 2-4 artifacts arranged compositionally are
    fine (e.g. 3 phone mockups stacked, several blueprint sheets fanned,
    multiple cooking ingredients), as long as they all reinforce the
    SAME topic. Avoid clutter from unrelated objects.
  * Setting: dark studio, polished dark surface, plain dark backdrop.
    Empty negative space lets the artifact dominate.
  * Lighting: dramatic key light from one side, deep shadows. Cool
    daylight for tech/design topics, warm tungsten for craft/lifestyle
    topics. The artifact's natural colors are the only saturation.
  * NO person, NO faces, NO body parts unless the artifact is literally
    held by a single hand entering frame (e.g. a hand holding a paint
    brush over a palette — but no full body or face).
  * Style: premium editorial product / design photography, hyperrealistic.

  Good "theme_artifact" examples:
  - UI DESIGN: "Three pristine smartphone-sized mockup screens float
    overlapping in a dark studio, displaying clean modern app UI layouts
    with abstract colored shapes, cards, and buttons (no readable text,
    just shapes). The screens have crisp white/colored interface
    elements on dark UI. Cool daylight key from upper left, deep shadows
    on the right, dark backdrop. Premium editorial design photography,
    hyperrealistic, cinematic, shallow depth of field."
  - CODE: "An ultra-close macro shot of a laptop screen filled with
    abstract colored syntax-highlighted code blocks (no readable
    letters — just the visual texture of indented colored shapes
    suggesting code). Dark IDE background. Single cool key light raking
    across the screen, deep shadows around. Minimal, premium editorial
    tech photography, hyperrealistic."
  - FINANCE: "A printed candlestick stock chart on cream paper lies on
    a dark wooden desk, the chart trending upward. A vintage fountain
    pen rests beside it. Hard warm key light from upper left, deep
    shadow on the right side. Dark moody palette, the green/red
    candlesticks as the only color pop. Editorial financial photography,
    hyperrealistic."
  - COOKING: "A perfectly plated minimalist dish — seared scallops with
    micro herbs and a citrus reduction — centered on a matte black
    ceramic plate against a deep black backdrop. Dramatic overhead key
    light, deep shadows. Earthy palette with the bright citrus as the
    only color pop. Premium food editorial photography, hyperrealistic."
  - ARCHITECTURE: "A detailed white architectural scale model of a
    modern house sits on a polished dark concrete surface in a darkened
    studio. Cool daylight from the upper left, long sharp shadows on
    the right. Minimalist, no other objects. Premium editorial
    architecture photography, hyperrealistic."

  ============================================================
  IF subject_focus == "person":
  ============================================================
  Hero a clear character + ONE highly TOPIC-SPECIFIC secondary object.
  Use ONLY when the reel is fundamentally about a human angle (mindset,
  advice, lifestyle, identity, story).

  STRUCTURE:
  * MAIN subject: clear character with distinctive look. If the message
    references "caveman" / "primitive" → render an actual caveman with
    shaggy beard and fur tunic doing a modern action. The contrast IS
    the point.
  * SECONDARY element: ONE real, instantly-readable object that comes
    DIRECTLY from the substance of the reel. NOT a generic default.
    Forbidden generic fallbacks (use only if THE reel is literally about
    them): plain open laptop with no visible content, plain coffee mug,
    plain notebook, plain wall calendar. If you find yourself reaching
    for one of these, STOP — reconsider whether theme_artifact is the
    right archetype instead.
  * Setting: minimal real environment, dark studio, dark wood desk.
  * Lighting: single hard key from one side, deep shadows on the other.
  * Color palette: dark/restrained base, ONE color pop from the
    secondary object.

  Good "person" examples (object always tied to substance):
  - MONEY MINDSET: "A scruffy caveman with shaggy hair and beard
    wearing a torn animal-fur tunic sits at a dark wooden desk against
    a near-black wall, holding a thick stack of crumpled green dollar
    bills in one hand. Hard warm tungsten key light from the left;
    right side in deep shadow. Dark moody palette with the green of
    the bills as the only color pop."
  - TRAVEL/IDENTITY: "A man in a sharp dark suit sits at a clean
    concrete desk in a near-black studio, sliding a single bright red
    passport across the table toward the camera. Cool blue daylight
    from upper left, hard shadows. Desaturated cool grey palette with
    the red passport as the only color pop."

  ============================================================
  IF subject_focus == "object":
  ============================================================
  Hero a single iconic still-life object representing an abstract
  concept. Use only when no theme artifact, brand, or person fits.

  STRUCTURE:
  * MAIN subject: ONE iconic real-world object, still life, centered.
  * Setting: dark studio, polished dark surface, plain backdrop.
  * Lighting: dramatic single key, deep shadows.
  * Color palette: dark/restrained, the object's natural color as
    the only pop.

  Good "object" example: "A single weathered brass key lies flat on a
  polished dark walnut surface against a near-black backdrop. Hard
  warm key light from upper left catches the worn metal, casting a
  long deep shadow trailing right. No other objects. Cinematic still
  life, hyperrealistic, A24 aesthetic."

  ============================================================
  Bad examples (DO NOT do this regardless of focus):
  ============================================================
  - "...holds a small symbolic stack of metallic AI token coins" (fictional object)
  - "...glowing AI interface" / "holographic display" (sci-fi cliché)
  - "...sits in a brightly lit modern office with several monitors and plants" (cluttered)
  - "A muscular primal caveman dominates over..." (theatrical)
  - For Google reel (famous brand): "A man in a suit holds a black folder" (wrong archetype — should hero the Google G logo)
  - For Claude/Cursor/Caveman reel (NOT famous): "A 3D Claude wordmark on a pedestal alone" (wrong archetype — viewer doesn't recognize the logo; integrate the wordmark via person archetype OR use theme_artifact if the topic has visual outputs)
  - For UI design reel: "A man at a laptop in a dark room" (wrong archetype — generic person scene loses the topic; should hero phone mockups / wireframes via theme_artifact)
  - For code reel: "A developer staring at a screen" (generic — should hero the code itself via theme_artifact)
  - For cooking reel: "A chef holding a knife" (lazy — should hero the plated dish or the ingredients via theme_artifact)
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
