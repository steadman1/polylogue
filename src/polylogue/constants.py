from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

API_VERSION: str = "v1"


class Endpoint(Enum):
    CHAT = "chat/completions"
    MODELS = "models"

    # ...


GGUF_TARGET = ".gguf"
MLX_TARGET = "config.json"

TEST_PROMPT = "Q: hello, how are you? A: i am doing "

GGUF_MODEL_PATH = Path(os.environ["GGUF_MODEL_PATH"])
MLX_MODEL_PATH = Path(os.environ["MLX_MODEL_PATH"])

MOCK_MODEL_ID = "some_model_id"

DB_MODELS_NAMESPACE = "models:registry"
