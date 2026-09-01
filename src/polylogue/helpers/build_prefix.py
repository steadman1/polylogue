from polylogue.constants import *


def build_prefix(path: Endpoint, without_version: bool = False) -> str:
    if without_version:
        return "/" + path.value
    return "/" + API_VERSION + "/" + path.value
