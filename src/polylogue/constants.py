from enum import Enum

API_VERSION: str = "v1"


class Endpoint(Enum):
    CHAT = "chat/completions"
    MODELS = "models"

    # ...


GGUF_TARGET = ".gguf"
MLX_TARGET = "config.json"

TEST_PROMPT = "Q: hello, how are you? A: i am doing "
