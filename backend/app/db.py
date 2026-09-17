import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings

engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=20, pool_pre_ping=True)

if settings.database_url.startswith("sqlite"):
    # SQLite ignores foreign keys unless asked, and the tests run on SQLite while
    # production runs on Postgres. Without this, deleting a row that something
    # else references passes every test and 500s on the real database — which is
    # exactly how the catalog-delete bug reached a running deployment.
    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Bring the schema up to date with the models.

    This is `alembic upgrade head`, not `Base.metadata.create_all` — create_all
    only fills in tables that do not exist yet, so a column added to an
    existing model would silently never reach a running deployment.
    `Base.metadata` stays the single source of truth either way, since the
    baseline migration is generated from it and CI fails the day they drift
    apart (test_migrations.py).

    A database that already has tables but no `alembic_version` row is one
    that create_all built before this project had migrations — every
    deployment that predates this change looks like that on its next
    restart, and so does the test suite's fresh_db fixture, which rebuilds
    the schema straight from Base.metadata between tests. Either way the
    schema already matches the baseline revision by construction, so it is
    stamped rather than replayed; replaying would try to CREATE TABLE over
    rows that are already there.
    """
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        existing_tables = await conn.run_sync(lambda c: inspect(c).get_table_names())

    def _migrate() -> None:
        cfg = Config(str(ALEMBIC_INI))
        if existing_tables and "alembic_version" not in existing_tables:
            command.stamp(cfg, "head")
        else:
            command.upgrade(cfg, "head")

    # alembic's async env.py drives its own event loop with asyncio.run(),
    # which cannot nest inside the one this coroutine is already running on.
    await asyncio.to_thread(_migrate)
