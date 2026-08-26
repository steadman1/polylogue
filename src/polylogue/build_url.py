from polylogue.constants import *

# TODO: take in list of Endpoint Path objects instead of just endpoint
# but this is all we need for now
def build_url(type: EndpointType) -> str:
    return f"{API_VERSION}/{type.value}"
