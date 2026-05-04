# racks-thumbnail-generator

Generador automático de miniaturas para reels de Racks. Pipeline:

```
video → audio → transcripción → tema → escena AI (Nanobanana) → texto Pillow (Inter + #BE190F) → miniatura final
```

- Escena cinematográfica generada por Gemini image (Nanobanana 2)
- Headline composited con Pillow usando Inter Black + color exacto de marca
- Adaptable: cada video genera escena/headline diferentes según su contenido
- Color de acento configurable

---

## Requisitos

- macOS o Linux
- [`uv`](https://docs.astral.sh/uv/) (`brew install uv`)
- [`ffmpeg`](https://ffmpeg.org/) (`brew install ffmpeg`)
- API key de Google AI Studio (`GOOGLE_API_KEY`) — [obtener aquí](https://aistudio.google.com/apikey)
- Opcional: `OPENAI_API_KEY` si quieres usar Whisper en vez de Gemini para transcripción

---

## Instalación

### Opción A — Run sin instalar (recomendado para probar)

```bash
uvx --from git+https://github.com/Racks-Labs/thumbnail_generator racks-thumbnail --help
```

`uvx` descarga, instala en entorno aislado y ejecuta — sin dejar nada permanente.

### Opción B — Install permanente (recomendado para uso diario)

```bash
uv tool install git+https://github.com/Racks-Labs/thumbnail_generator
```

Tras esto el comando `racks-thumbnail` está disponible globalmente.

Update:
```bash
uv tool upgrade racks-thumbnail-generator
```

---

## Configuración (primera vez)

```bash
racks-thumbnail config set google-api-key TU_GOOGLE_API_KEY
```

Se guarda en `~/.config/racks-thumbnail/.env` con permisos `600` (solo tu user).

Opcional:
```bash
racks-thumbnail config set openai-api-key TU_OPENAI_KEY    # Si quieres Whisper
racks-thumbnail config set accent-color "#FF6B00"          # Color custom (default #BE190F)
racks-thumbnail config set transcriber whisper             # Backend default (gemini|whisper)
```

Si ejecutas `generate` sin haber seteado la key te la pedirá interactivamente y la guardará.

### Comandos de gestión

```bash
racks-thumbnail config show              # ver config (keys enmascaradas)
racks-thumbnail config get google-api-key
racks-thumbnail config set google-api-key NUEVA_KEY
racks-thumbnail config unset google-api-key
racks-thumbnail config path              # ruta del archivo
```

### Precedencia de configuración

| Source | Ejemplo | Prioridad |
|--------|---------|-----------|
| CLI flag | `--accent-color "#FF0000"` | 1 (max) |
| Env var | `GOOGLE_API_KEY=xxx racks-thumbnail ...` | 2 |
| `.env` local | `./` (cwd actual) | 3 |
| Config global | `~/.config/racks-thumbnail/.env` | 4 (default) |

---

## Uso

```bash
# Básico — usa config global
racks-thumbnail generate video.mp4

# Output dir custom (default: ./output del cwd actual)
racks-thumbnail generate video.mp4 --output ~/Desktop/thumbs

# Color de acento puntual (override del configurado)
racks-thumbnail generate video.mp4 --accent-color "#00B4FF"

# Cambiar transcripción a Whisper (requiere OPENAI_API_KEY)
racks-thumbnail generate video.mp4 --transcriber whisper

# Solo testear transcripción + extracción de tema (sin generar imagen)
racks-thumbnail test-transcribe audio.mp3

# Modelo de imagen alternativo
racks-thumbnail generate video.mp4 --model gemini-3.1-flash-image-preview
```

### Output

Genera 2 archivos en `./output/<cwd>/`:
- `_bg_<timestamp>.png` — escena AI sin texto (intermediate)
- `thumbnail_<timestamp>.png` — miniatura final con texto compuesto

---

## Cómo funciona

1. **Audio**: `ffmpeg` extrae el audio del video → WAV mono 16kHz
2. **Transcripción**: Gemini 2.5 Flash o Whisper transcribe el audio en español
3. **Análisis**: Gemini extrae:
   - `headline` — frase corta de valor (3-6 palabras)
   - `headline_accent_word` — palabra clave que va sobre el bloque rojo
   - `concepto_visual` — descripción cinematográfica de la escena
4. **Imagen de fondo**: Nanobanana 2 (`gemini-2.5-flash-image`) genera la escena editorial sin texto
5. **Composición de texto**: Pillow superpone el headline con Inter Black + bloque de color exacto

El **estilo** (tipografía, colores, layout, reglas de composición) está definido en `racks_thumbnail_generator/templates/default.yaml` y se aplica de forma constante. Lo **variable** (escena, headline) se genera por video.

---

## Estilo de marca

- Tipografía: **Inter** (Black, Bold, SemiBold, Medium, Regular, Light) — bundleada
- Colores: negro `#000000`, blanco `#FFFFFF`, rojo Racks `#BE190F` (configurable)
- Estética: editorial cinematográfico, A24 film still, alto contraste chiaroscuro

Referencia visual: `Guia_estilos_RU.pdf` (en el repo).

---

## Licencia

MIT — uso libre. Solo Racks Labs publica updates oficiales.
