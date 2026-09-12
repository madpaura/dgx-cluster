import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# `app` has to be importable no matter what directory this was invoked from —
# a bare `alembic` CLI call, `init_db()` running it from a background thread,
# or the test suite pointed at a scratch database all land here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Registers every table on Base.metadata as a side effect of the import, the
# same way app.db.init_db does it — a model added to app/models.py without a
# migration shows up here as autogenerate drift instead of a silent gap.
from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app import models  # noqa: E402,F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# alembic.ini deliberately carries no sqlalchemy.url: the database this run
# touches must be the one the app itself would connect to, not a value that
# can drift out of sync with it in a second config file.
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it — for review, or for a DBA
    who applies changes by hand rather than letting the app touch prod."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
