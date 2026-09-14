from __future__ import annotations

from pathlib import Path

from openai.types import Model


class ModelRecord(Model):
    # inherits id: str
    path: Path
    maximum_n_ctx: int | None = None
    description: str | None = None
