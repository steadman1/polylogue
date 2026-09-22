from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Load the keys/values from the .env file into os.environ
load_dotenv()

API_VERSION: str = "v1"


class Endpoint(Enum):
    CHAT = "chat/completions"
    MODELS = "models"
    RESPONSES = "responses"
    # ...


GGUF_TARGET = ".gguf"
MLX_TARGET = "config.json"

MOCK_PROMPT = "Q: hello, how are you? A: i am doing "
MOCK_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": MOCK_PROMPT}]}]

GGUF_MODEL_PATH = Path(os.environ["GGUF_MODEL_PATH"])
MLX_MODEL_PATH = Path(os.environ["MLX_MODEL_PATH"])

SERVER_PEPPER_ENCODING = "utf-8"
SERVER_PEPPER = bytes(os.environ["SERVER_PEPPER"], SERVER_PEPPER_ENCODING)

MOCK_MODEL_ID = "some_model_id"

DB_MODELS_NAMESPACE = "models:registry"

DEFAULT_N_CTX = 128_000
