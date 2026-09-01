import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from polylogue.clients import app, get_db
from polylogue.constants import DB_MODELS_NAMESPACE, MLX_MODEL_PATH, MOCK_MODEL_ID
from polylogue.db.model_record import ModelRecord
from polylogue.dependency_injection.mock_db import MockDB


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    mock_db = MockDB()

    mock_model = ModelRecord(
        id=MOCK_MODEL_ID,
        created=0,
        object="model",
        owned_by="",
        path=MLX_MODEL_PATH,
    )
    _ = asyncio.run(
        mock_db.hset(DB_MODELS_NAMESPACE, MOCK_MODEL_ID, mock_model.model_dump_json())
    )

    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def pytest_addoption(parser):
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Runs all tests including skipped tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        # If the flag is passed, remove the skip markers dynamically
        for item in items:
            if (
                item.get_closest_marker("skip")
                or item.get_closest_marker("skipif")
                or item.get_closest_marker("slow")
            ):
                item.own_markers = [
                    m
                    for m in item.own_markers
                    if m.name not in ("skip", "skipif", "slow")
                ]
