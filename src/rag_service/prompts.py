"""Versioned prompt files: prompts/<family>/<version>.md with a YAML front-matter header."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import yaml

from rag_service.config import ROOT

PROMPT_DIR = ROOT / "prompts"


@dataclass(frozen=True)
class Prompt:
    id: str
    template: str
    inputs: tuple[str, ...]
    sha: str  # first 12 hex of the file hash, logged with every trace

    def render(self, **values: str) -> str:
        missing = set(self.inputs) - values.keys()
        if missing:
            raise KeyError(f"prompt {self.id} missing inputs: {sorted(missing)}")
        out = self.template
        # Plain substitution (not str.format) so literal JSON braces in rubrics survive.
        for k in self.inputs:
            out = out.replace("{" + k + "}", str(values[k]))
        return out


@lru_cache(maxsize=16)
def load_prompt(name: str) -> Prompt:
    """name like 'answer/v1' or 'judge/faithfulness_v1'."""
    path = PROMPT_DIR / f"{name}.md"
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path} lacks front matter")
    _, header, body = raw.split("---", 2)
    meta = yaml.safe_load(header) or {}
    return Prompt(
        id=str(meta.get("id", name)),
        template=body.strip() + "\n",
        inputs=tuple(meta.get("inputs") or ()),
        sha=hashlib.sha256(raw.encode()).hexdigest()[:12],
    )
