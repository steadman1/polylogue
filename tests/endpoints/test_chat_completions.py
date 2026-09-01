from fastapi.testclient import TestClient

from polylogue.constants import MOCK_MODEL_ID, Endpoint
from polylogue.helpers.build_prefix import build_prefix


def test_list_completions(client: TestClient) -> None:
    response = client.get(build_prefix(Endpoint.CHAT))

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


def test_create_completion_non_streaming(client: TestClient) -> None:
    request = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "stream": False,
    }

    response = client.post(build_prefix(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    assert "assistant" in str(response.content)


def test_create_completion_streaming(client: TestClient) -> None:
    request = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "stream": True,
    }

    response = client.post(build_prefix(Endpoint.CHAT), json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "assistant" in str(response.content)


def test_missing_required_parameters_create_completion(client: TestClient):
    requests = [
        {},
        {
            # "model": "",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "test"}]}
            ],
            "stream": True,
        },
        {
            "model": "",
            # "messages": [
            #   {"role": "user", "content": [{"type": "text", "text": "test"}]}
            # ],
            "stream": True,
        },
    ]
    for request in requests:
        response = client.post(build_prefix(Endpoint.CHAT), json=request)
        assert response.status_code == 422
