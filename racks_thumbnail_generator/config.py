import os
from pathlib import Path
from enum import Enum

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_FONTS = PACKAGE_ROOT / "assets" / "fonts" / "Inter"
BUNDLED_TEMPLATES = PACKAGE_ROOT / "templates"


def global_config_dir() -> Path:
    """XDG-style: ~/.config/racks-thumbnail/ (override with $XDG_CONFIG_HOME)."""
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "racks-thumbnail"


def global_config_path() -> Path:
    return global_config_dir() / ".env"


class TranscriberBackend(str, Enum):
    GEMINI = "gemini"
    WHISPER = "whisper"


class Settings(BaseSettings):
    # Precedence: env var > local .env > global ~/.config/.../.env
    model_config = {
        "env_file": (str(global_config_path()), ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    google_api_key: str = ""
    openai_api_key: str = ""

    transcriber: TranscriberBackend = TranscriberBackend.GEMINI
    accent_color: str = Field(default="#BE190F")

    fonts_dir: Path = BUNDLED_FONTS
    templates_dir: Path = BUNDLED_TEMPLATES
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "output")

    @model_validator(mode="after")
    def _ensure_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def font_path(self, weight: str = "Regular") -> Path:
        return self.fonts_dir / f"Inter-{weight}.ttf"


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)


# ---- Global config file management ----

CONFIG_KEYS = {"GOOGLE_API_KEY", "OPENAI_API_KEY", "TRANSCRIBER", "ACCENT_COLOR"}


def read_global_config() -> dict[str, str]:
    path = global_config_path()
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def write_global_config(values: dict[str, str]) -> Path:
    path = global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items() if v]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
    return path


def set_global_config_value(key: str, value: str) -> Path:
    key = key.upper().replace("-", "_")
    if key not in CONFIG_KEYS:
        raise ValueError(f"Unknown config key: {key}. Allowed: {sorted(CONFIG_KEYS)}")
    cfg = read_global_config()
    cfg[key] = value
    return write_global_config(cfg)
