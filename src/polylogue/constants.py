from enum import Enum

API_VERSION: str = "v1"

class EndpointType(Enum):
    CHAT = "chat/completions"
    AUDIO = "audio"
    # ...
