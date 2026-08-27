from enum import Enum

API_VERSION: str = "v1"

class Endpoint(Enum):
    CHAT = "chat/completions"
    AUDIO = "audio"

    # ...
