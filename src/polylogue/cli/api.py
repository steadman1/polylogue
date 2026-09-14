import asyncio

import click

from polylogue.cli.cli import cli
from polylogue.clients import get_redis_client
from polylogue.db.api_key_manager import APIKeyManager


@cli.group(name="api")
def api_group() -> None:
    """API key management command group."""
    return


@api_group.command("create")
@click.option(
    "--owner-id",
    "-o",
    required=True,
    help="Identifier of the key owner (e.g. user ID or service).",
)
@click.option(
    "--name",
    "-n",
    default="default",
    show_default=True,
    help="Descriptive label for this key.",
)
@click.option(
    "--rate-limit",
    "-r",
    default=100,
    type=int,
    show_default=True,
    help="Allowed requests per minute.",
)
def create_key_cmd(owner_id: str, name: str, rate_limit: int) -> None:
    """Generate and store a new API key."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        manager = APIKeyManager(client)
        try:
            result = await manager.create_key(
                owner_id=owner_id, name=name, rate_limit=rate_limit
            )
            click.echo("API key successfully created:")
            click.echo(f"  Key ID:     {result.record.key_id}")
            click.echo(f"  Owner:      {result.record.owner_id}")
            click.echo(f"  Name:       {result.record.name}")
            click.echo(f"  Rate Limit: {result.record.rate_limit} req/min")
            click.echo(f"\nSecret Key:  {result.raw_key}")
            click.echo(
                click.style(
                    "Warning: Save this key now. It will not be displayed again.",
                    fg="yellow",
                )
            )
        finally:
            await client.aclose()
            await pool.disconnect()

    asyncio.run(_runner())


@api_group.command("get")
@click.option(
    "--key-id",
    "-k",
    required=True,
    help="Public key identifier (e.g. sk_live_...).",
)
def get_key_cmd(key_id: str) -> None:
    """Retrieve metadata for an API key by Key ID."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        manager = APIKeyManager(client)
        try:
            record = await manager.get_key(key_id)
            if record:
                click.echo(record.model_dump_json(indent=2))
            else:
                click.echo(f"Key ID '{key_id}' not found.", err=True)
        finally:
            await client.aclose()
            await pool.disconnect()

    asyncio.run(_runner())


@api_group.command("verify")
@click.argument("raw_key", required=True)
def verify_key_cmd(raw_key: str) -> None:
    """Verify if a raw API key token is valid and active."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        manager = APIKeyManager(client)
        try:
            record = await manager.verify_raw_key(raw_key)
            if record:
                click.echo(click.style("Valid API key.", fg="green"))
                click.echo(f"  Key ID: {record.key_id}")
                click.echo(f"  Owner:  {record.owner_id}")
                click.echo(f"  Active: {record.is_active}")
            else:
                click.echo(
                    click.style("Invalid or revoked API key.", fg="red"),
                    err=True,
                )
        finally:
            await client.aclose()
            await pool.disconnect()

    asyncio.run(_runner())


@api_group.command("list")
@click.option(
    "--owner-id",
    "-o",
    required=True,
    help="Owner ID whose keys should be listed.",
)
def list_keys_cmd(owner_id: str) -> None:
    """List all API keys belonging to a specific owner."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        manager = APIKeyManager(client)
        try:
            keys = await manager.list_keys(owner_id)
            if not keys:
                click.echo(f"No keys found for owner '{owner_id}'.")
                return

            click.echo(f"Keys for {owner_id} ({len(keys)}):")
            for rec in keys:
                status = "active" if rec.is_active else "revoked"
                click.echo(
                    f"  - {rec.key_id} | Name: {rec.name} | Status: {status} | Rate Limit: {rec.rate_limit}"
                )
        finally:
            await client.aclose()
            await pool.disconnect()

    asyncio.run(_runner())


@api_group.command("list-owners")
def list_owners_cmd() -> None:
    """List all registered API key owners."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        manager = APIKeyManager(client)
        try:
            owners = await manager.list_owners()
            if not owners:
                click.echo("No API key owners registered.")
                return

            click.echo(f"Registered Owners ({len(owners)}):")
            for owner in owners:
                click.echo(f"  - {owner}")
        finally:
            await client.aclose()
            await pool.disconnect()

    asyncio.run(_runner())


@api_group.command("delete")
@click.option(
    "--key-id", "-k", required=True, help="Public Key ID to delete or revoke."
)
def delete_key_cmd(key_id: str) -> None:
    """Delete an API key by its Key ID."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        manager = APIKeyManager(client)
        try:
            deleted = await manager.delete_key(key_id)
            if deleted:
                click.echo(f"Deleted key: {key_id}")
            else:
                click.echo(f"Key '{key_id}' not found.", err=True)
        finally:
            await client.aclose()
            await pool.disconnect()

    asyncio.run(_runner())
