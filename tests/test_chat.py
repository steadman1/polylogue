from fastapi.testclient import TestClient

from polylogue.constants import Endpoint
from polylogue.fastapi_app import app
from polylogue.build_url import build_url

client = TestClient(app=app)

def test_create_completion_streaming() -> None:
    request = {"stream": True}

    response = client.post(build_url(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

def test_create_completion_non_streaming() -> None:
    request = {"stream": False}

    response = client.post(build_url(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
