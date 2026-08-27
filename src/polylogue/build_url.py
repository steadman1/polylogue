from polylogue.constants import *

def build_url(path: Endpoint) -> str:
    return "/" + API_VERSION + "/" + path.value
