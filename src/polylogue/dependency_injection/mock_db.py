from __future__ import annotations

from collections.abc import Sequence


# implements all redis functions accessed in ModelRecordManager
# written by Gemini
class MockDB[T]:
    def __init__(self) -> None:
        # Nested dictionary: outer key is Redis hash key, inner key is field
        self.store: dict[str, dict[str, T]] = {}

    async def hget(self, key: str, field: str) -> T | None:
        return self.store.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: T) -> int:
        if key not in self.store:
            self.store[key] = {}
        is_new = field not in self.store[key]
        self.store[key][field] = value
        return 1 if is_new else 0

    async def hdel(self, key: str, field: str) -> int:
        if key in self.store and field in self.store[key]:
            del self.store[key][field]
            return 1
        return 0

    async def hgetall(self, key: str) -> dict[str, T]:
        return self.store.get(key, {})

    # Synchronous helper for populating data in pytest fixtures without event loop
    def seed(self, key: str, field: str, value: T) -> None:
        if key not in self.store:
            self.store[key] = {}
        self.store[key][field] = value
