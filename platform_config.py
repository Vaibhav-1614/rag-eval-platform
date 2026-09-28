"""Load project settings from config/settings.yml (repo root relative)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent
_SETTINGS_PATH = _ROOT / "config" / "settings.yml"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    p = path or _SETTINGS_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return _ROOT
