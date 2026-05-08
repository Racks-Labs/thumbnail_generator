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

  STEP 1 — Recognizable brand test.
    Does the reel center on a NAMED brand / app / tool / software /
    company / product?

    Use your own judgment as a language model with broad training to
    decide whether the brand has a documented public visual identity
    (an actual logo + defined brand colors) that you can describe
    accurately. If you can confidently name the logo shape and the
    primary brand colors of the brand mentioned, it passes — render
    its actual logo as the hero in brand_product.

    Examples to anchor your judgment (NOT exhaustive — use any brand
    you know): Sony, Google, Adobe, Photoshop, Blender, Autodesk,
    Maya, AutoCAD, Claude, ChatGPT, Apple, Spotify, YouTube, Nike,
    Coca-Cola, Tesla, Figma, Cinema 4D, Unreal, Unity, etc. — and
    any other brand of similar or greater public presence.

    Edge cases:
    - Brand is a small startup / hobby project / single-dev tool with
      no real brand kit you can recall → STEP 2 (theme_artifact or
      person, integrating the wordmark on a laptop / sticker).
    - You're unsure about exact logo or colors but you recognize the
      brand exists → still pick brand_product, but describe the logo
      using only what you're confident about (general shape, dominant
      color). Do NOT invent fake colors. If genuinely unknown, fall
      to STEP 2.

    → Brand passes → "brand_product".
    → When in doubt and the brand IS the topic → DEFAULT TO
      "brand_product". Showing the actual logo is almost always more
      readable than an abstract scene or generic person.
    → If no named brand at all → continue to STEP 2.

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
  TIE-BREAKERS:
  - If the reel mentions ANY named brand / app / tool / software with a
    documented logo → "brand_product" wins. Showing the official logo
    in its real brand colors is the most readable possible thumbnail.
  - If no named brand but a clear topic with visual artifacts (UI, code,
    food, charts) → "theme_artifact".
  - "person" is reserved for genuinely human topics with no brand and
    no visual artifact. Niche default.
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

  Good "brand_product" examples (each uses a different placement —
  rotate creatively, don't repeat the same setup):

  - GOOGLE (box): "The provided Google 'G' logo emerges from a kraft
    cardboard box on a dark wooden table. Polystyrene packing peanuts
    spill around it. Subtle multicolor rim light, dark out-of-focus
    office bokeh, hard key from above-right, deep shadows. Premium
    commercial product photography, hyperrealistic."
  - ADOBE (storefront sign): "The provided Adobe logo glows softly on
    a glass storefront window at dusk, with reflections of city lights
    and a darkened sidewalk in front. Hard key from a streetlamp upper
    left, deep shadows. Cinematic editorial photography, hyperrealistic."
  - APPLE (product hero): "A pristine silver MacBook Pro centered on
    a clean dark slate desk in a darkened minimalist studio, the
    glowing Apple logo on the lid clearly visible. Hard key from
    upper left, deep shadows on the right. No other objects. Premium
    commercial product photography, hyperrealistic."
  - SPOTIFY (mug): "The provided Spotify logo printed in vivid green
    on a matte black ceramic coffee mug, the mug centered on a dark
    polished wooden desk against a near-black backdrop. Wisp of steam
    rising. Hard warm key from upper left, deep shadows. Hyperrealistic
    editorial product photography."
  - BLENDER (hoodie): "The provided Blender logo embroidered on the
    chest of a folded grey heather hoodie laying on a dark walnut
    surface. Hard key light from upper left raking across the fabric
    weave, deep shadows on the right. Hyperrealistic editorial
    photography, premium streetwear catalog feel."
  - SONY (vinyl): "The provided Sony logo printed on the cover of a
    matte black vinyl record sleeve, the sleeve standing upright in a
    record store row, soft warm tungsten light from above, dark
    out-of-focus shop bokeh behind. Hyperrealistic editorial."
  - MULTI-BRAND (Adobe vs Figma): "Two glossy ceramic coffee mugs sit
    side by side on a polished dark wooden desk. The left mug shows
    the provided Adobe logo, the right mug shows the provided Figma
    logo. Hard cool daylight key from upper left, deep shadows.
    Hyperrealistic editorial product photography, comparison still."

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
  - For any reel naming a recognizable software/app/brand (Adobe, Blender, Autodesk, Maya, Claude, Cursor, Figma, etc.): falling back to person + laptop or to abstract theme_artifact is WRONG — these all have official logos and brand colors that should be heroed via brand_product. Always prefer showing the actual logo in vivid official colors over a generic person scene or abstract artifact.
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
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=EXTRACTION_PROMPT.format(transcript=transcript),
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ThumbnailContent,
        ),
    )
    content = ThumbnailContent.model_validate_json(response.text)
    # Hard-enforce headline limits even if LLM ignored them
    content.headline = _sanitize_headline(content.headline)
    content.headline_accent_word = content.headline_accent_word.strip()
    return content
