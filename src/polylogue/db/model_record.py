from __future__ import annotations

from pathlib import Path

from openai.types import Model


class ModelRecord(Model):
    path: Path
    description: str | None = None  # might want to axe this later
