import hashlib
import hmac
import secrets
from collections.abc import Sequence

from polylogue.constants import SERVER_PEPPER, SERVER_PEPPER_ENCODING
from polylogue.db.api_key_record import APIKeyCreateResult, APIKeyRecord
from polylogue.db.protocols.db_client import DBClient


class APIKeyManager:
    def __init__(
        self,
        db_client: DBClient,
        key_prefix: str = "sk_live",
    ) -> None:
        self.db_client = db_client
        self.key_prefix = key_prefix

    def _hash_secret(self, secret: str) -> str:
        return hmac.new(
            SERVER_PEPPER,
            secret.encode(SERVER_PEPPER_ENCODING),
            hashlib.sha256,
        ).hexdigest()

    def _key_lookup_id(self, key_id: str) -> str:
        return f"apikey:meta:{key_id}"

    def _owner_index_id(self, owner_id: str) -> str:
        return f"apikey:owner:{owner_id}"

    # 1. CREATE
    async def create_key(
        self,
        owner_id: str,
        name: str = "default",
        rate_limit: int = 100,
    ) -> APIKeyCreateResult:
        """Generates a key, hashes the secret, saves to DB, and returns the raw key."""
        public_id = secrets.token_hex(8)
        secret = secrets.token_hex(32)
        key_id = f"{self.key_prefix}_{public_id}"
        raw_key = f"{key_id}_{secret}"

        secret_hash = self._hash_secret(secret)
        record = APIKeyRecord(
            key_id=key_id,
            owner_id=owner_id,
            name=name,
            secret_hash=secret_hash,
            rate_limit=rate_limit,
        )

        await self.save_key(record)
        return APIKeyCreateResult(raw_key=raw_key, record=record)

    # 2. SAVE (Create or Update)
    async def save_key(self, record: APIKeyRecord) -> None:
        """Persists or updates an APIKeyRecord and ensures indexing sets are updated."""
        lookup_key = self._key_lookup_id(record.key_id)
        owner_key = self._owner_index_id(record.owner_id)

        mapping = {
            "key_id": record.key_id,
            "owner_id": record.owner_id,
            "name": record.name,
            "secret_hash": record.secret_hash,
            "rate_limit": str(record.rate_limit),
            "is_active": "1" if record.is_active else "0",
            "created_at": str(record.created_at),
            "last_used_at": str(record.last_used_at or ""),
        }

        # Check if client supports async context manager pipeline (redis-py async)
        if hasattr(self.db_client, "pipeline"):
            async with self.db_client.pipeline(transaction=True) as pipe:
                pipe.hset(lookup_key, mapping=mapping)
                pipe.sadd(owner_key, record.key_id)
                await pipe.execute()
        else:
            # Fallback for generic key-value DBClient implementations without pipelining
            await self.db_client.hset(lookup_key, mapping=mapping)  # type: ignore
            await self.db_client.sadd(owner_key, record.key_id)  # type: ignore

    # 3. GET
    async def get_key(self, key_id: str) -> APIKeyRecord | None:
        """Retrieves a stored API key record by its key_id."""
        data = await self.db_client.hgetall(self._key_lookup_id(key_id))
        if not data:
            return None

        # Decode if response contains raw bytes
        clean_data = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else str(v)
            )
            for k, v in data.items()
        }

        return APIKeyRecord(
            key_id=clean_data["key_id"],
            owner_id=clean_data["owner_id"],
            name=clean_data.get("name", "default"),
            secret_hash=clean_data["secret_hash"],
            rate_limit=int(clean_data["rate_limit"]),
            is_active=clean_data["is_active"] == "1",
            created_at=int(clean_data["created_at"]),
            last_used_at=(
                int(clean_data["last_used_at"])
                if clean_data.get("last_used_at")
                else None
            ),
        )

    async def verify_raw_key(self, raw_key: str) -> APIKeyRecord | None:
        parts = raw_key.split("_", 3)
        if len(parts) != 4:
            return None

        key_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
        secret = parts[3]

        record = await self.get_key(key_id)
        if not record or not record.is_active:
            return None

        computed_hash = self._hash_secret(secret)
        if not hmac.compare_digest(record.secret_hash, computed_hash):
            return None

        return record

    # 5. LIST
    async def list_keys(self, owner_id: str) -> Sequence[APIKeyRecord]:
        """Lists all keys associated with a specific owner."""
        key_ids = await self.db_client.smembers(self._owner_index_id(owner_id))
        if not key_ids:
            return []

        results: list[APIKeyRecord] = []
        for kid in key_ids:
            clean_kid = kid.decode() if isinstance(kid, bytes) else str(kid)
            record = await self.get_key(clean_kid)
            if record:
                results.append(record)
        return results

    async def list_owners(self) -> Sequence[str]:
        """Lists all unique owner IDs currently registered in the database."""
        owners = await self.db_client.smembers(self._global_owners_index_id())
        if not owners:
            return []

        return sorted([o.decode() if isinstance(o, bytes) else str(o) for o in owners])

    # 6. DELETE
    async def delete_key(self, key_id: str) -> bool:
        """Deletes key metadata and unlinks it from the owner's index set."""
        record = await self.get_key(key_id)
        if not record:
            return False

        lookup_key = self._key_lookup_id(key_id)
        owner_key = self._owner_index_id(record.owner_id)

        if hasattr(self.db_client, "pipeline"):
            async with self.db_client.pipeline(transaction=True) as pipe:
                pipe.delete(lookup_key)
                pipe.srem(owner_key, key_id)
                await pipe.execute()
        else:
            await self.db_client.delete(lookup_key)  # type: ignore
            await self.db_client.srem(owner_key, key_id)  # type: ignore

        return True
