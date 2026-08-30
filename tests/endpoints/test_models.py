from fastapi.testclient import TestClient

from polylogue.constants import MOCK_MODEL_ID, Endpoint
from polylogue.helpers.build_prefix import build_prefix


def test_list_models(client: TestClient) -> None:
    response = client.get(build_prefix(Endpoint.MODELS))

    assert response.status_code == 200


def test_get_model(client: TestClient) -> None:
    url = build_prefix(Endpoint.MODELS) + "/" + MOCK_MODEL_ID
    response = client.get(url)

    assert response.status_code == 200
