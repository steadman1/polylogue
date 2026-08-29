from fastapi.testclient import TestClient

from polylogue.constants import Endpoint
from polylogue.fastapi_app import app
from polylogue.helpers.build_prefix import build_prefix

client = TestClient(app=app)


def test_list_models() -> None:
    response = client.get(build_prefix(Endpoint.MODELS))

    assert response.status_code == 200


def test_get_model() -> None:
    url = build_prefix(Endpoint.MODELS) + "/some_model"
    response = client.get(url)

    assert response.status_code == 200
