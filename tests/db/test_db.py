import asyncio
from collections.abc import Sequence
from pathlib import Path

from openai.types.model import Model

from polylogue.clients import get_redis_client
from polylogue.db.model_record import ModelRecord
from polylogue.db.model_record_manager import ModelRecordManager
from polylogue.db.protocols.db_client import DBClient
from polylogue.db.protocols.record_manager import RecordManager
from polylogue.dependency_injection.mock_db import MockDB

RECORD = ModelRecord(
    id="0",
    created=123456,
    owned_by="this-machine",
    object="model",
    path=Path("/path/with/no/model"),
)


def test_conforms_protocol_db_record() -> None:
    record = RECORD

    assert isinstance(record, ModelRecord)
    assert isinstance(record, Model)


def test_conforms_protocol_redis() -> None:
    _, redis = get_redis_client()

    assert isinstance(redis, DBClient)


def test_conforms_protocol_db_manager() -> None:
    db: MockDB[str] = MockDB()

    assert isinstance(db, DBClient)

    manager = ModelRecordManager(db)

    assert isinstance(manager, RecordManager)


def test_save_db_manager() -> None:
    db: MockDB[str] = MockDB()
    manager = ModelRecordManager(db)

    asyncio.run(manager.save(RECORD))
    json_str = asyncio.run(db.hget(manager.namespace, RECORD.id))

    assert json_str is not None


def test_get_db_manager() -> None:
    db: MockDB[str] = MockDB()
    manager = ModelRecordManager(db)

    asyncio.run(manager.save(RECORD))
    record = asyncio.run(manager.get(RECORD.id))

    assert isinstance(record, ModelRecord)
    assert isinstance(record, Model)


def test_list_db_manager() -> None:
    db: MockDB[str] = MockDB()
    manager = ModelRecordManager(db)

    asyncio.run(manager.save(RECORD))
    records = asyncio.run(manager.list_all())

    assert isinstance(records, Sequence)
    assert all(isinstance(record, ModelRecord) for record in records)


def test_delete_db_manager() -> None:
    db: MockDB[str] = MockDB()
    manager = ModelRecordManager(db)

    asyncio.run(manager.save(RECORD))
    json_str = asyncio.run(db.hget(manager.namespace, RECORD.id))

    assert json_str is not None

    is_deleted = asyncio.run(manager.delete(RECORD.id))
    json_str = asyncio.run(db.hget(manager.namespace, RECORD.id))

    assert is_deleted
    assert json_str is None
