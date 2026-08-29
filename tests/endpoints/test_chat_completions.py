from fastapi.testclient import TestClient

from polylogue.constants import Endpoint
from polylogue.fastapi_app import app
from polylogue.helpers.build_prefix import build_prefix

client = TestClient(app=app)


def test_list_completions() -> None:
    response = client.get(build_prefix(Endpoint.CHAT))

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


def test_create_completion_non_streaming() -> None:
    request = {"model": "", "messages": [], "prompt": "", "stream": False}

    response = client.post(build_prefix(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


def test_create_completion_streaming() -> None:
    request = {"model": "", "messages": [], "prompt": "", "stream": True}

    response = client.post(build_prefix(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_missing_required_parameters_create_completion():
    requests = [
        {},
        {
            # "model": "",
            "messages": [],
            "prompt": "",
            "stream": True,
        },
        {
            "model": "",
            # "messages": [],
            "prompt": "",
            "stream": True,
        },
        {
            "model": "",
            "messages": [],
            # "prompt": "",
            "stream": True,
        },
    ]
    for request in requests:
        response = client.post(build_prefix(Endpoint.CHAT), json=request)
        assert response.status_code == 422
