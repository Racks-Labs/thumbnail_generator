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
    company / product whose logo and brand colors are well-documented
    and visually distinct? The brand does NOT need to be known by the
    general public — it's enough that creators / users in that space
    recognize it. The headline will reinforce the brand name; the logo
    is the strong visual anchor.

    Examples that PASS (use "brand_product"):
      - Tech / consumer: Google, Apple, iPhone, MacBook, iPad, AirPods,
        ChatGPT, OpenAI, YouTube, Instagram, TikTok, WhatsApp, Tesla,
        Spotify, Netflix, Microsoft, Windows, Office, Excel, Amazon, Meta,
        Facebook, X / Twitter, LinkedIn, Uber, PayPal, Visa, Mastercard,
        Disney, Disney+, HBO, Pinterest, Snapchat, Telegram, Zoom.
      - AI / LLM / generative: Claude, Anthropic, ChatGPT, OpenAI,
        Gemini, Copilot, Perplexity, Midjourney, Stable Diffusion,
        DALL-E, Sora, Suno, Runway, ElevenLabs, Pika, Luma, Kling.
      - Creative / design / 2D suites: Adobe, Photoshop, Illustrator,
        Premiere, Premiere Pro, After Effects, Lightroom, InDesign,
        Acrobat, Adobe Creative Cloud, Figma, Canva, Procreate, Sketch,
        Affinity, Notion, Framer, Webflow.
      - 3D / VFX / CAD / engineering software: Blender, Autodesk, Maya,
        3ds Max, AutoCAD, Fusion 360, Revit, Inventor, Cinema 4D, C4D,
        Houdini, ZBrush, Substance Painter, Substance Designer, Marmoset,
        SolidWorks, Rhino, Rhinoceros, Grasshopper, KeyShot, V-Ray, Arnold.
      - Game engines: Unity, Unreal Engine, Godot, GameMaker.
      - Video / audio editing: DaVinci Resolve, Final Cut Pro, iMovie,
        Avid, Logic Pro, Ableton Live, FL Studio, Pro Tools, Audition,
        CapCut, Veo, Filmora.
      - Dev / hosting / coding tools: Cursor, Linear, Replit, Lovable,
        v0, Bolt, GitHub, GitLab, Docker, Vercel, Netlify, AWS, Azure,
        Google Cloud, Cloudflare, Stripe, Supabase, Firebase, Heroku,
        Linux, Ubuntu, Postgres, MongoDB.
      - Productivity / no-code: Slack, Zapier, n8n, Make, Airtable,
        ClickUp, Asana, Trello, Discord, Loom, Calendly.
      - Gaming / streaming: Twitch, Steam, PlayStation, Xbox, Nintendo,
        Roblox, Minecraft, Fortnite, Valorant, League.
      - Mass consumer brands: Nike, Adidas, Coca-Cola, Pepsi,
        McDonald's, Starbucks, BMW, Mercedes, Audi, Toyota, Ford, IKEA.

    Examples that FAIL (use STEP 2):
      Brands so new / niche / unbranded that they have NO documented
      visual identity (no logo, no brand colors), or one-off code
      tools with literally just a CLI name. Very rare. If a quick
      Google search would surface a brand kit / logo guidelines, the
      brand passes.

    → If brand passes → "brand_product".
    → When in doubt → DEFAULT TO "brand_product" if any named tool or
      service is the topic. Showing the logo is almost always more
      readable than an abstract scene.
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
  * MAIN subject: the brand's recognizable mark or product, rendered in a real physical / 3D form, in FULL OFFICIAL BRAND COLORS at high saturation. NOT a flat 2D logo on a screen — physical mark sitting in the scene. NOT desaturated. The logo must be unmistakably itself. Examples (use the ACTUAL brand colors):
    - Google: 3D Google "G" letter in vivid red/yellow/green/blue (#EA4335/#FBBC05/#34A853/#4285F4)
    - Adobe: 3D Adobe "A" stylized logo in vivid Adobe red (#FA0F00), glossy
    - Photoshop: 3D Photoshop "Ps" badge — glossy navy blue (#001E36) rounded square with cyan "Ps" lettering
    - Illustrator: 3D Illustrator "Ai" badge — orange (#FF9A00) rounded square with white "Ai"
    - Premiere Pro: 3D "Pr" badge — violet (#9999FF) with deep violet lettering
    - After Effects: 3D "Ae" badge — purple (#9999FF / #00005B) with violet lettering
    - Lightroom: 3D "Lr" badge — light blue (#31A8FF / #001E36) rounded square
    - Figma: 3D Figma logo — five colored shapes (red #F24E1E, orange #FF7262, green #0ACF83, blue #1ABCFE, purple #A259FF) as physical sculpture
    - Canva: 3D Canva "C" logo in vivid teal-to-purple gradient (#00C4CC to #7D2AE8)
    - Blender: 3D Blender logo — the iconic stylized "blue/orange torus eye" mark (#EA7600 orange + #265787 blue), rendered as a glossy 3D physical mark
    - Autodesk: 3D Autodesk "A" pyramid logo in vivid Autodesk black + white with pyramid geometry, OR a giant 3D Autodesk wordmark in clean white on dark
    - Maya: 3D Autodesk Maya "M" badge — teal/green (#0696D7) glossy rounded square
    - 3ds Max: 3D Autodesk 3ds Max "3" badge — dark blue (#0696D7) glossy
    - AutoCAD: 3D AutoCAD red "A" badge (#E60000)
    - Cinema 4D: 3D C4D logo in vivid blue (#011A6A) glossy
    - Unreal Engine: 3D Unreal "U" logo in vivid black with subtle blue glow
    - Unity: 3D Unity cube logo in white/black official 3D mark
    - DaVinci Resolve: 3D DaVinci wordmark in vivid red/gold accents
    - Final Cut Pro: 3D FCP slate logo in clean silver/black
    - Claude: 3D "Claude" wordmark sculpted in matte material in warm cream (#F0EEE6) and Claude orange (#D97757) tones, premium ceramic look
    - ChatGPT: 3D OpenAI swirl logo in glossy white on black, or in teal (#0FA37F)
    - Gemini: 3D Gemini sparkle / star icon in vivid Google blue-purple gradient
    - Cursor: 3D Cursor logo (the stylized C cursor mark) in clean black/white
    - Notion: 3D Notion "N" sculpture in clean black/white
    - Linear: 3D Linear logo in clean black gradient
    - Apple: a real MacBook or iPhone shown front-and-center, beautifully lit; Apple logo in clean silver
    - Spotify: 3D Spotify circle logo in vivid green (#1DB954)
    - YouTube: 3D YouTube play button in vivid red (#FF0000) on white rounded rectangle
    - Instagram: 3D Instagram camera icon with official magenta-orange-purple gradient
    - Tesla: actual Tesla car, or red T logo
    - Generic fallback for any other named brand: research the brand's
      actual logo and primary brand colors, then describe a 3D physical
      version of that logo in those colors. Do NOT invent fake colors.
  * SECONDARY element (optional, only if it adds to the message): a real physical container or context — a kraft cardboard box (often with a stamp like "GRATIS", "NEW", "FREE" if relevant — but the stamp text will be added separately, just describe the box as having a printable stamp area), gift wrapping, a wooden crate, a museum-style pedestal, a Polaroid-style frame.
  * Setting: dark studio backdrop OR softly out-of-focus modern office bokeh OR clean dark wooden surface. Empty negative space around the hero. The hero MUST dominate the frame.
  * Lighting: dramatic single key light highlighting the brand mark, with light leaks / colored rim light if it matches the brand colors (e.g. Google = subtle multi-color rainbow light leak in the background; Apple = clean white-on-white; Claude = warm cream/orange tones).
  * Color palette: dominated by the BRAND'S OWN color identity (Google = full RGB color pop; Claude = warm cream/orange; OpenAI = teal/black; Apple = silver/white). The rest of the frame stays dark/neutral.
  * Style: premium product photography, commercial advertising still, 3D render quality, hyperrealistic.

  Good "brand_product" examples (note: hero in FULL official colors):
  - GOOGLE: "A tangible 3D rendered Google 'G' letter in vivid full official colors (bright red #EA4335, yellow #FBBC05, green #34A853, blue #4285F4) emerges dramatically from an open kraft cardboard box on a dark wooden table. Polystyrene packing peanuts spill around the G. Subtle multicolor rainbow light leak from the right, deep dark out-of-focus office bokeh in the background. Dramatic key light from above-right, deep shadows on the box. Premium commercial product photography, hyperrealistic."
  - ADOBE: "A tangible 3D rendered Adobe 'A' stylized logo in vivid Adobe red (#FA0F00) with glossy plastic finish, sitting on a polished dark surface in a darkened studio. Subtle red rim light from behind. Hard key light from upper left, deep dark shadow trailing right. Empty negative space around the logo. Premium commercial 3D product photography, hyperrealistic, cinematic."
  - PHOTOSHOP: "A giant 3D Photoshop 'Ps' badge — a glossy rounded square with the iconic deep navy blue background (#001E36) and the bold cyan 'Ps' lettering — sits front and center on a dark polished surface in a moody studio. Cyan rim light from behind, hard key light from above-left, deep shadows. Hyperrealistic 3D product render, cinematic."
  - BLENDER: "A glossy 3D rendering of the iconic Blender logo — the stylized eye-shaped mark with the orange (#EA7600) outer torus and the deep blue (#265787) inner sphere — sits prominently on a polished dark surface in a darkened studio. Subtle orange rim light from behind. Hard key light from upper left, deep shadow trailing right. Empty negative space around the logo. Hyperrealistic 3D product render, cinematic, premium commercial photography."
  - AUTODESK MAYA: "A giant 3D Autodesk Maya 'M' badge — a glossy rounded square with the iconic teal-blue (#0696D7) background and bold white 'Maya' wordmark — front and center on a dark polished surface in a moody studio. Cyan rim light from behind, hard key light from upper left, deep shadows. Hyperrealistic 3D product render, cinematic."
  - AUTOCAD: "A 3D rendered AutoCAD 'A' logo in vivid Autodesk red (#E60000) with glossy finish, sitting on a polished dark concrete surface in a darkened architectural studio. Subtle red rim light, hard key light from upper left, deep shadows trailing right. Hyperrealistic, premium commercial 3D product photography."
  - CLAUDE: "A 3D rendered matte ceramic 'Claude' wordmark sculpture in vivid Claude orange (#D97757) on warm cream (#F0EEE6) base, sitting on a polished dark wooden table against a near-black blurred studio background. Warm rim light from behind. Single warm tungsten key light from upper left rakes across the lettering creating soft warm shadows. Hyperrealistic, premium commercial 3D product still."
  - SPOTIFY: "A glossy 3D Spotify circle logo in vivid Spotify green (#1DB954) with the three sound waves clearly visible, floats above a dark polished surface in a moody studio. Green rim light, hard key light from upper left, deep shadows. Hyperrealistic 3D product render, cinematic."
  - APPLE: "A pristine silver MacBook Pro centered on a clean dark slate desk in a darkened minimalist studio, the glowing Apple logo on the lid clearly visible. Hard key light from upper left, deep shadows on the right. No other objects. Premium commercial product photography, hyperrealistic."

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
