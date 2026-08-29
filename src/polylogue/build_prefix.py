from polylogue.constants import *


def build_prefix(path: Endpoint) -> str:
    return "/" + API_VERSION + "/" + path.value
