from __future__ import annotations

import os
from enum import Enum

API_VERSION: str = "v1"


class Endpoint(Enum):
    CHAT = "chat/completions"
    MODELS = "models"

    # ...


GGUF_TARGET = ".gguf"
MLX_TARGET = "config.json"

TEST_PROMPT = "Q: hello, how are you? A: i am doing "

INFERENCE_MODELS_DIR: str | None = os.getenv("INFERENCE_MODELS_DIR")
