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
    brand_queries: list[str]


EXTRACTION_PROMPT = """\
Analyze this transcript from a Spanish-language reel and design a thumbnail concept.

Transcript:
{transcript}

Return a JSON object with:

- headline: A short impactful phrase in Spanish that communicates the SPECIFIC VALUE the viewer gets by watching this reel.

  HARD LIMITS (the renderer will look bad if you exceed these):
  * 3-5 words MAX. Never 6+. If the idea needs more, cut it.
  * 30 characters MAX (counting spaces). Shorter = punchier.
  * NO punctuation: no commas, periods, colons, semicolons, dashes,
    quotes, parentheses, exclamation marks, question marks. Plain words
    separated by single spaces only.
  * NO subtitles inside (no "X: Y" structure — pick one or the other).

  CONTENT RULES:
  * Be CONCRETE — name the specific benefit, tool, brand, or outcome.
  * Reference WHAT the viewer learns / saves / gets — money, time,
    a tool name, a method, a number.
  * NOT a vague slogan ("Domina tu IA", "Cambia tu vida") — useless.
  * Sound like a Spanish-speaking creator's editorial title, not corporate.

  Good examples (count: 3-5 words, ≤30 chars):
  - "7 NUEVAS IAS DE GOOGLE"        (5 words, 21 chars)
  - "CLAUDE COMO CAVERNICOLA"        (3 words, 23 chars)
  - "AHORRA 70% EN TOKENS"           (4 words, 19 chars)
  - "PAGA LA MITAD POR CLAUDE"       (5 words, 24 chars)
  - "7 IAS QUE DISEÑAN UI"           (5 words, 20 chars)

  Bad examples (TOO LONG / WRONG):
  - "DOMINA EL DISEÑO DE INTERFACES CON IA EN MINUTOS" (10 words — too long)
  - "AWESOME DESIGN: DISEÑA INTERFACES PERFECTAS"      (colon banned, too long)
  - "DEJA DE PAGAR DE MAS POR CLAUDE!"                 (! banned)

- headline_accent_word: ONE single word that appears LITERALLY (character-for-character)
  in the headline. The renderer will try fuzzy matching but EXACT is best.

  RULES:
  * Pick ONE word from the headline string. Just one. Single token, no spaces.
  * Must be a verbatim copy of the word as it appears in the headline
    (same accents, same casing of letters — match the exact spelling).
  * Pick the word that carries the value: the brand (CLAUDE, GOOGLE),
    the number (7, 70%), the key verb (AHORRA, DOMINA), the key noun (UI, IAS).
  * Avoid filler words ("DE", "LA", "POR", "CON", "EN", "QUE") unless
    they're the ONLY meaningful tokens (rare).
  * If the headline has no obvious standout, pick the longest word.

  Examples:
  - headline "7 NUEVAS IAS DE GOOGLE"   → accent "GOOGLE" or "7"
  - headline "AHORRA 70% EN TOKENS"     → accent "70%" or "AHORRA"
  - headline "CLAUDE COMO CAVERNICOLA"  → accent "CAVERNICOLA" or "CLAUDE"
  - headline "7 IAS QUE DISEÑAN UI"     → accent "DISEÑAN" or "UI"

- subtexto: Short complementary phrase in Spanish (max 8 words). Optional context.

- keywords: 3-5 main topics/themes.

- tono: provocador | informativo | motivacional | educativo | controversial.

- subject_focus: Decide what the SCENE is centered around. Pick ONE.

  =====================================================================
  DECISION TREE — apply IN THIS ORDER, stop at first match:
  =====================================================================

  CORE QUESTION before anything else:
    What is the VISUAL VALUE this reel teaches? What does the viewer
    walk away SEEING? Be specific:
      - "How to upscale image resolution" → transformation: blurry → sharp
      - "Edit videos with AI in 30s" → before/after of an edit
      - "Adobe Photoshop hidden tool" → the tool's effect on an image
      - "What is Anthropic" → brand identity itself
      - "Sony A7iv vs Canon R6" → two cameras side-by-side
      - "Stop wasting money on tokens" → savings concept (money + brand)
      - "Best 7 AI design apps" → grid of apps / mockups
      - "My morning routine" → ritual / lifestyle still

    The VALUE drives the archetype. The brand mentioned is often
    INCIDENTAL — it's the tool the viewer uses to achieve the value,
    not the value itself. Ask: would the thumbnail still communicate
    the value if the brand logo were removed? If yes, the brand is
    incidental → favor theme_artifact. If no (the brand IS the topic),
    favor brand_product.

  STEP 1 — Is the BRAND itself the topic?
    Triggers (pick brand_product):
    - Reel is a review / first look / hot take ON the brand
    - Reel asks "what is X" / "X is dead" / "X vs Y" (brand-on-brand)
    - Reel is brand identity / launch / history / comparison
    - The headline you wrote literally names the brand as the main noun

    NOT triggers (do NOT pick brand_product, even if brand mentioned):
    - Reel teaches a TECHNIQUE that happens to use the brand as a tool
      ("how to upscale resolution with Freepik" → the technique is the
      hero, not Freepik)
    - Reel shows an EFFECT or TRANSFORMATION the brand produces
      ("Photoshop AI fill before/after" → the transformation is hero)
    - Reel demos an OUTCOME ("designs you can make with Figma" → the
      designs are hero)
    - Reel uses brand as one of MANY tools mentioned in passing

    → Brand IS the topic → "brand_product".
    → Brand is incidental → continue to STEP 2.

    For brand_product to apply, you should also be confident you know
    the brand's actual logo shape and primary colors. If genuinely
    unknown → STEP 2.

  STEP 2 — Theme artifact / concept test.
    Does the reel teach a TECHNIQUE, EFFECT, TRANSFORMATION, COMPARISON,
    OUTCOME or visible concept? If yes → "theme_artifact" with a scene
    composition tailored to the concept.

    SCENE PATTERN — pick the one best suited to the reel. ROTATE — do
    NOT default to product photography of a single object. Patterns:

    a) BEFORE / AFTER VERTICAL SPLIT — for transformations: low-res →
       high-res, messy → clean, old → new, rough → polished, drab →
       vibrant. Example: "Vertical split: left half a blurry pixelated
       portrait, right half the same portrait crisp and sharp."

    b) BEFORE / AFTER HORIZONTAL SPLIT — same as above but top/bottom.

    c) DIAGONAL TRANSITION / WIPE — transformation mid-process across a
       diagonal line cutting the frame.

    d) BEFORE → ARROW → AFTER — two states with a visible arrow,
       flow line, or stacked triptych showing progression.

    e) SIDE-BY-SIDE COMPARISON — A vs B (two cameras, two phones, two
       plates, two charts). Equal weight on both halves.

    f) GRID / SHEET ARRANGEMENT — N items laid out as a sample sheet
       (e.g. "Best 7 AI apps" → 7 phone mockups arranged 2x4; "5 color
       palettes" → 5 color swatches; "10 fonts" → typographic specimen).

    g) MACRO REVEAL / MAGNIFIER — close-up of detail being revealed
       (e.g. magnifying glass over fabric, lens revealing pixels at
       extreme zoom).

    h) HAND IN ACTION — a single hand performing the technique on the
       artifact (hand sharpening a knife, hand holding a polaroid mid-
       development, hand swiping a tablet). Brings the technique to life.

    i) INPUT → OUTPUT — show the raw input and the finished output in
       the same composition (raw ingredients next to plated dish; sketch
       next to finished render; tangled wires next to clean setup).

    j) ARTIFACT HERO — single iconic visual output filling the frame
       (UI mockups, code editor close-up, stock chart, plated dish,
       blueprints, camera body, mixing console, sheet music, paint
       palette, garment, scale model, etc.). Use only when the artifact
       alone tells the story; otherwise pick a multi-state pattern.

    Topic → artifact lookup (fallback if pattern unclear):
      - UI / UX / interfaces → phone or screen mockups
      - Code / dev → code editor close-up with colored syntax
      - Music / audio → mixing console, synthesizer, waveform
      - Finance / trading → candlestick chart on paper, ledger
      - Cooking / food → plated dish overhead, ingredients
      - Photography / video editing → camera body, film strips
      - 3D / motion / VFX → 3D model render, storyboard frames
      - Writing / content → typewriter, edited manuscript
      - Fitness → kettlebell, gym bench
      - Beauty → product bottles, brushes
      - Crypto → physical coin (Bitcoin)
      - Productivity → planner, sticky notes

    Brand integration in theme_artifact: when brand_queries is non-empty
    AND scene archetype is theme_artifact, the brand logo should appear
    NATURALLY in the scene as a label / tag / corner watermark / device
    edge / app icon on a phone mockup — never as the dominant hero
    (that's brand_product's job). The TRANSFORMATION or CONCEPT is the
    hero. The logo is the credit.

    → Concept fits a pattern → "theme_artifact".
    → No clear visual concept and no brand → continue to STEP 3.

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
  TIE-BREAKERS:
  - If the reel teaches a TECHNIQUE, EFFECT, TRANSFORMATION, OUTCOME or
    COMPARISON → "theme_artifact" wins, even when a brand is mentioned
    (the brand becomes a small credit/tag in the corner, not the hero).
  - If the reel is ABOUT the brand itself (review, identity, launch,
    brand-on-brand showdown) → "brand_product".
  - When unsure between brand_product and theme_artifact: ask "could
    this thumbnail communicate the value WITHOUT the logo?" If yes →
    theme_artifact. If no → brand_product.
  - "person" is reserved for genuinely human topics (mindset, advice,
    relationships, biography) with no transformation / artifact / brand
    identity at the core.
  =====================================================================

- brand_queries: List of search queries (one per brand) that the pipeline
  will use to download official logo PNGs from Wikipedia and pass them as
  reference images to the image generator. The image generator will then
  composite the REAL logo into the scene (not an invented version).

  Rules:
  * Include EVERY named brand / app / tool / company / product the reel
    talks about, even if subject_focus is not "brand_product". Useful for
    person scenes where a brand wordmark appears on a laptop / mug, and
    for theme_artifact scenes where a brand logo appears on the artifact.
  * Each query string must be the canonical name that maximizes the
    chance of hitting the right Wikipedia page. Prefer "Adobe Inc.",
    "Sony Group Corporation", "Blender (software)", "Autodesk Maya",
    "Anthropic" over ambiguous short names. Append disambiguation in
    parentheses when a plain name is too generic ("Apple Inc." not "Apple").
  * Order them by importance (the main brand of the reel first).
  * Empty list [] when the reel has no specific named brand.

  Examples:
  - Reel about Adobe Photoshop → ["Adobe Photoshop"]
  - Reel about Sony cameras → ["Sony Group Corporation", "Sony Alpha"]
  - Reel about Claude tool by Anthropic → ["Anthropic"]  (Anthropic page has Claude logo)
  - Reel about Blender vs Maya → ["Blender (software)", "Autodesk Maya"]
  - Reel about an unbranded mindset topic → []

- concepto_visual: Cinematic scene description in English for an AI image generator. The composition depends on subject_focus.

  COMMON HARD CONSTRAINTS — banned in EVERY scene type:
  * NO holograms, NO neon, NO glowing floating screens, NO futuristic interfaces, NO blue/purple sci-fi lighting, NO server rooms with rocks
  * NO fictional / made-up / symbolic objects (no "AI token coins", no "energy orbs", no abstract symbols)
  * NO ABSTRACT HEROES — the main subject must be a LITERAL, instantly
    recognizable real-world thing (a real product, a real logo, a real
    object). NEVER abstract wireframes, abstract digital art, abstract
    gears floating in space, abstract gradients as the hero, generic
    geometric shapes as the hero. Viewers must name what they see in
    under 1 second.
  * HERO SATURATION — the protagonist (logo, product, or artifact) must
    be HIGHLY SATURATED and visually striking. Render brand logos in
    their FULL official colors at full vibrancy, not desaturated, not
    muted, not greyscale, not "moody monochrome". The rest of the frame
    stays dark and restrained — the hero is the only saturated thing,
    and it pops hard.
  * NO clutter — strip out background props
  * Photorealistic, cinematic, shallow depth of field, premium editorial / commercial photography quality
  * HIGH CONTRAST CHIAROSCURO lighting — single hard key light, deep dark shadows
  * NO TEXT in the image (text gets composited later)
  * READABILITY TEST — a viewer scrolling Instagram for 0.5 seconds must
    INSTANTLY identify what the hero is. If your scene needs explanation,
    it fails. Pick obvious over clever.
  * CREATIVITY / NO TEMPLATES — never default to one go-to composition.
    Each reel deserves a scene custom-fit to its specific VALUE. If you
    notice yourself reaching for "kraft cardboard box with logo on
    front" or "single mug on a dark desk" yet again, STOP. Re-read
    the reel's value statement (the headline) and pick a different
    composition that better illustrates THAT specific message —
    transformation split, comparison, magnifier reveal, hand-in-action,
    grid, before-arrow-after, etc. Variety is mandatory.

  ============================================================
  IF subject_focus == "brand_product":
  ============================================================
  Hero the BRAND or PRODUCT itself. NO person in the frame. Use creative product / hero photography.

  STRUCTURE:
  * MAIN subject: the brand's actual logo or hero product, integrated
    into a creative real-world scene. The pipeline will pass the
    actual logo PNG as a reference image — your concepto_visual just
    needs to describe the SCENE that incorporates that logo. Don't
    describe the logo's exact colors / pixel-perfect shape — the
    real logo file does that for you. Reference it as "the provided
    <BRAND> logo".

  * LOGO SIZE — CRITICAL. The logo (or the product carrying the logo)
    is the HERO of the frame. It must dominate, not be a tiny detail
    on a distant object. Required size:
    - The logo / product MUST occupy at least 50% of the frame width
      AND at least 40% of the frame height in the upper-middle region.
    - If the logo sits on an object (mug, box, hoodie, billboard, etc.)
      that object must fill most of the frame so that the printed/
      embroidered logo on it reads instantly large. Frame the object
      tightly — close-up macro / hero shot, not a wide context shot.
    - Camera should be CLOSE to the hero, not far away. Background
      bokeh stays small in the frame; the hero is BIG.
    - Bad: "a small mug on a large desk with bookshelves and plants
      behind it" → logo ends up tiny.
    - Good: "a tight hero close-up of a single ceramic mug filling
      most of the frame, the provided logo printed boldly across its
      front, dark out-of-focus backdrop" → logo dominates.

  * VARIETY IS REQUIRED. Do NOT default to "3D logo on pedestal" every
    time. Pick a different real-world placement for the logo each time.
    Examples of placements (rotate between these / invent similar ones):
    - Logo on a kraft cardboard box (gift wrap / packaging)
    - Logo printed on a hoodie / t-shirt being held up
    - Logo on a coffee mug sitting next to a planner
    - Logo on a glass storefront window with reflection
    - Logo painted as a mural on a brick wall
    - Logo on a vintage neon sign hanging over a bar / studio (real
      neon, not glowing sci-fi)
    - Logo embroidered on a baseball cap on a dark wood surface
    - Logo on a sticker stuck to a laptop lid
    - Logo on a billboard above a city street at dusk
    - Logo on a polished silver pin badge held between fingers
    - Logo as an enamel paperweight on a desk
    - Logo printed on a tote bag hanging on a chair
    - Logo as a giant inflatable balloon at a trade show
    - Logo on a tablet screen as wallpaper, the tablet propped on a stand
    - Logo etched into wood / marble / brushed metal
    - Logo displayed inside a clear acrylic / glass cube paperweight
    - Logo on a vinyl record sleeve
    - Logo on a magazine cover
    - Logo on a hardback book spine
    - Logo woven into fabric of a folded blanket
    - Logo as the badge on the front of a sneaker / cap / car
    - 3D logo sculpture on a pedestal (FINE OCCASIONALLY, but not the default)
  * If the brand has a specific iconic product (Tesla car, MacBook,
    iPhone, AirPods), the product itself can be the hero with the logo
    naturally visible on it.

  * Multi-brand reels: when brand_queries contains 2+ entries, the
    scene should integrate ALL the logos in a believable composition
    (e.g. two product boxes side by side, two mugs on a desk, multiple
    storefront signs, head-to-head comparison setup). Don't just pick
    one and ignore the others.

  * The provided reference logo image(s) are the source of truth for
    visual identity. Use them. Do NOT re-imagine or restyle the logo.
  * SECONDARY element (optional, only if it adds to the message): a real physical container or context — a kraft cardboard box (often with a stamp like "GRATIS", "NEW", "FREE" if relevant — but the stamp text will be added separately, just describe the box as having a printable stamp area), gift wrapping, a wooden crate, a museum-style pedestal, a Polaroid-style frame.
  * Setting: dark studio backdrop OR softly out-of-focus modern office bokeh OR clean dark wooden surface. Empty negative space around the hero. The hero MUST dominate the frame.
  * Lighting: dramatic single key light highlighting the brand mark, with light leaks / colored rim light if it matches the brand colors (e.g. Google = subtle multi-color rainbow light leak in the background; Apple = clean white-on-white; Claude = warm cream/orange tones).
  * Color palette: dominated by the BRAND'S OWN color identity (Google = full RGB color pop; Claude = warm cream/orange; OpenAI = teal/black; Apple = silver/white). The rest of the frame stays dark/neutral.
  * Style: premium product photography, commercial advertising still, 3D render quality, hyperrealistic.

  CRITICAL — DO NOT COPY ANY EXAMPLES LITERALLY:
  Every reel deserves its OWN composition. The placements listed above
  (mug, box, hoodie, billboard, etc.) are starting points to break out
  of templates — pick whichever feels least obvious for THIS particular
  brand and value, or invent a completely different placement that fits
  this brand's specific identity better. If you've seen yourself reach
  for "cardboard box" or "ceramic mug" in your last few outputs,
  exclude those. Every generation must look freshly imagined for the
  specific brand at hand. Variety across runs is mandatory.

  ============================================================
  IF subject_focus == "theme_artifact":
  ============================================================
  Hero the VISUAL CONCEPT of the reel — the transformation, the
  comparison, the technique, the outcome, or the artifact itself.
  Pick a SCENE PATTERN (see STEP 2 a-j) that best matches the value
  the reel teaches. Vary the pattern across generations — do NOT
  always default to pattern (j) "artifact hero".

  STRUCTURE:
  * MAIN subject: the topic's tangible artifact, rendered as a real
    physical / on-screen object. Pick the artifact from STEP 2's list
    (UI mockups, code editor, mixing console, candlestick chart on
    paper, plated dish, blueprints, camera body, etc.).
  * The artifact must be LITERAL and INSTANTLY READABLE. NEVER abstract
    wireframes, NEVER abstract floating gears, NEVER abstract digital
    art compositions, NEVER vague "creative output" symbols. If a
    viewer can't say "that's a UI mockup" / "that's a code editor" /
    "that's a stock chart" in <1 second, the scene fails.
  * Multiple instances OK: 2-4 artifacts arranged compositionally are
    fine (e.g. 3 phone mockups stacked, several blueprint sheets fanned,
    multiple cooking ingredients), as long as they all reinforce the
    SAME topic. Avoid clutter from unrelated objects.
  * Setting: dark studio, polished dark surface, plain dark backdrop.
    Empty negative space lets the artifact dominate.
  * Lighting: dramatic key light from one side, deep shadows. Cool
    daylight for tech/design topics, warm tungsten for craft/lifestyle
    topics.
  * SATURATION: the artifact must be in vivid, recognizable colors at
    high saturation (real UI in colorful screens, real food in real
    plate colors, real sheet music in clean black on cream). The rest
    of the frame stays dark and restrained — the artifact is the
    saturated hero.
  * NO person, NO faces, NO body parts unless the artifact is literally
    held by a single hand entering frame (e.g. a hand holding a paint
    brush over a palette — but no full body or face).
  * Style: premium editorial product / design photography, hyperrealistic.

  NO CONCRETE EXAMPLES PROVIDED ON PURPOSE.

  Reason: every time examples are listed, they become anchors and the
  same composition gets repeated across unrelated reels. The
  scene-pattern menu in STEP 2 (a-j) is the ONLY anchoring you get.
  Pick the pattern that fits the reel's specific value, then INVENT
  a concrete composition fresh for THIS reel — drawing on:
    - the specific topic / value of the reel
    - the brand colors and identity (if any logos provided)
    - real-world settings appropriate to the topic's domain
    - the chosen scene pattern's structural logic

  Required diversity check: do NOT use the same scene formula across
  consecutive reels. If the previous reel was a vertical before/after
  split, this one should NOT be a vertical before/after split unless
  the topic is genuinely identical. Pick a different pattern.

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
  - For an "upscale image resolution with Freepik" reel: "Hero close-up of a cardboard box with the Freepik logo printed on its front" — WRONG. The VALUE is the upscale transformation, not Freepik's identity. Use theme_artifact pattern (a) before/after split showing the blurry→sharp transformation. Freepik logo, if shown, is a small credit corner.
  - For an Adobe Photoshop AI fill reel: "3D Adobe logo on pedestal" — WRONG. The value is the fill effect. Use theme_artifact before/after split. Adobe logo as small credit only.
  - For a Sony vs Canon reel: "Sony logo hero close-up alone" — WRONG. Use side-by-side comparison with BOTH cameras / logos.
  - Defaulting to "cardboard box with logo on front" / "single ceramic mug with logo" repeatedly for different reels — WRONG. Each reel deserves its own composition based on its specific value. Rotate scene patterns aggressively.
  - For UI design reel: "A man at a laptop in a dark room" (wrong archetype — generic person scene loses the topic; should hero phone mockups / wireframes via theme_artifact)
  - For code reel: "A developer staring at a screen" (generic — should hero the code itself via theme_artifact)
  - For cooking reel: "A chef holding a knife" (lazy — should hero the plated dish or the ingredients via theme_artifact)
  - For Adobe / Photoshop / Illustrator / Figma reel (FAMOUS BRAND): "Floating wireframe gears next to abstract gradient digital art" (WRONG archetype + abstract — Adobe is famous, the scene must hero the bright red Adobe 'A' logo or the Ps/Ai/Pr/Figma badge in vivid official colors via brand_product, never abstract shapes)
  - "Abstract wireframe / floating gears / abstract digital art / vague creative output" as the hero (BANNED — must be a real recognizable product or logo, not abstract)
  - "...rendered in cool grey tones" / "...desaturated palette including the hero" (WRONG — the hero must be vividly saturated, only the background is dark/restrained)
"""


_PUNCT_TO_STRIP = set('.,;:!?¡¿"\'()[]{}—–-')


def _sanitize_headline(text: str) -> str:
    """Strip banned punctuation, collapse whitespace, uppercase, cap word count."""
    cleaned = "".join((" " if c in _PUNCT_TO_STRIP else c) for c in text)
    words = [w for w in cleaned.split() if w]
    if len(words) > 5:
        words = words[:5]
    return " ".join(words).upper()


def extract_topic(transcript: str, api_key: str) -> ThumbnailContent:
    """Extract thumbnail content from transcript.

    High temperature + per-call random anchor to fight pattern collapse:
    Gemini at default temperature locks onto the same composition formula
    across calls (e.g. always 'cardboard box with logo'). We push
    temperature high and inject a random seed phrase so each call gets
    distinct sampling.
    """
    import random
    seed = random.randint(1_000_000, 99_999_999)

    diversity_anchor = (
        f"\n\n=== GENERATION SEED #{seed} ===\n"
        f"Variation directive for this specific call: deliberately diverge from\n"
        f"the most obvious / first-thought composition. If your initial concept\n"
        f"would be 'logo on a cardboard box' or 'vertical before/after split' or\n"
        f"'hero close-up of a single object', force yourself to consider a\n"
        f"different scene pattern from STEP 2 (a-j) that ALSO fits this reel.\n"
        f"Use the seed number above as a creative jolt to break out of templates.\n"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=EXTRACTION_PROMPT.format(transcript=transcript) + diversity_anchor,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ThumbnailContent,
            temperature=1.5,
            top_p=0.95,
        ),
    )
    content = ThumbnailContent.model_validate_json(response.text)
    # Hard-enforce headline limits even if LLM ignored them
    content.headline = _sanitize_headline(content.headline)
    content.headline_accent_word = content.headline_accent_word.strip()
    return content
