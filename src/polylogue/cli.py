import asyncio
from datetime import datetime
from pathlib import Path

import click

from polylogue.clients import get_redis_client
from polylogue.db.model_record import ModelRecord
from polylogue.db.model_record_manager import ModelRecordManager


@click.group()
def cli() -> None:
    """CLI management tool for Polylogue model registry."""
    return


@cli.command("get")
@click.option("--model-id", "-m", required=True, help="ID of the model to fetch.")
def get_model_cmd(model_id: str) -> None:
    """Retrieve a model record by ID."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        db = ModelRecordManager(client)

        record = await db.get(model_id)
        if record:
            click.echo(record.model_dump_json(indent=2))
        else:
            click.echo(f"Model '{model_id}' not found.", err=True)

        await client.aclose()
        await pool.disconnect()

    asyncio.run(_runner())


@cli.command("save")
@click.option("--model-id", "-m", required=True, help="ID of the model to save.")
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to model file.",
)
@click.option(
    "--n-ctx",
    "-n",
    required=True,
    type=int,
    help="Maximum length of model's context window.",
)
@click.option(
    "--description",
    "-d",
    required=False,
    help="Helpful description of the model.",
)
def save_model_cmd(model_id: str, path: str, n_ctx: int, description: str) -> None:
    """Register a new model record."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        db = ModelRecordManager(client)

        timestamp_seconds = int(datetime.now().timestamp())
        record = ModelRecord(
            id=model_id,
            created=timestamp_seconds,
            object="model",
            owned_by="local",
            path=Path(path).resolve(),
            maximum_n_ctx=n_ctx,
            description=description,
        )

        await db.save(record)
        click.echo(f"Saved model: {model_id}")

        await client.aclose()
        await pool.disconnect()

    asyncio.run(_runner())


@cli.command("list")
def list_models_cmd() -> None:
    """List all registered models."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        db = ModelRecordManager(client)

        records = await db.list_all()
        for rec in records:
            click.echo(f"- {rec.id} -> {rec.path}")

        await client.aclose()
        await pool.disconnect()

    asyncio.run(_runner())


@cli.command("delete")
@click.option("--model-id", "-m", required=True, help="ID of the model to delete.")
def delete_model_cmd(model_id: str) -> None:
    """Delete a model record by ID."""

    async def _runner() -> None:
        pool, client = get_redis_client()
        db = ModelRecordManager(client)

        deleted = await db.delete(model_id)
        if deleted:
            click.echo(f"Deleted model: {model_id}")
        else:
            click.echo(f"Model '{model_id}' not found.", err=True)

        await client.aclose()
        await pool.disconnect()

    asyncio.run(_runner())


if __name__ == "__main__":
    cli()
