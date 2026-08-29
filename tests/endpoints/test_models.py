from fastapi.testclient import TestClient

from polylogue.build_prefix import build_prefix
from polylogue.constants import Endpoint
from polylogue.fastapi_app import app

client = TestClient(app=app)


def test_list_models() -> None:
    response = client.get(build_prefix(Endpoint.MODELS))

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


def test_get_model() -> None:
    request = {"model": ""}

    response = client.post(build_prefix(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_missing_model_arg_get_model() -> None:
    request = {"model": "", "messages": [], "prompt": "", "stream": True}

    response = client.post(build_prefix(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
