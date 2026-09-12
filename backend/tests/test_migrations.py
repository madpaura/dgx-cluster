"""Migrations must stay in step with the models.

create_all only fills in tables that do not exist yet, so a column added to an
existing model would never reach a running deployment. Migrations close that
gap — but only while someone remembers to generate one. The drift test below is
what turns "remember to" into "the suite fails".
"""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from app.db import Base

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def _upgraded(tmp_path: Path, monkeypatch) -> str:
    """Build a database by running every migration, as a deployment does.

    env.py resolves the URL from the application settings, which are read once
    at import — so the setting is what has to be redirected here, not the
    environment variable behind it.
    """
    from app.config import settings

    url = f"sqlite:///{tmp_path / 'schema.db'}"
    monkeypatch.setattr(settings, "database_url", url.replace("sqlite://", "sqlite+aiosqlite://"))
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return url


def test_migrating_an_empty_database_creates_every_table_the_models_define(tmp_path, monkeypatch):
    engine = create_engine(_upgraded(tmp_path, monkeypatch))
    try:
        built = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()
    assert built == set(Base.metadata.tables)


def test_the_migrated_schema_has_not_drifted_from_the_models(tmp_path, monkeypatch):
    """The test that catches a model change shipped without a migration.

    Autogenerate against a fully migrated database must find nothing to do. A
    failure here means someone edited app/models.py without running
    `alembic revision --autogenerate`.
    """
    engine = create_engine(_upgraded(tmp_path, monkeypatch))
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()

    # SQLite cannot express every constraint the models declare, so only a
    # table or column present on one side and absent on the other counts.
    structural = [d for d in diff if d[0] in
                  {"add_table", "remove_table", "add_column", "remove_column"}]
    assert not structural, f"models have drifted from migrations: {structural}"


def test_every_migration_can_be_rolled_back(tmp_path, monkeypatch):
    """A migration you cannot undo is one you cannot safely deploy."""
    url = _upgraded(tmp_path, monkeypatch)
    command.downgrade(Config(str(ALEMBIC_INI)), "base")
    engine = create_engine(url)
    try:
        left = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()
    assert left == set(), f"downgrade left tables behind: {sorted(left)}"


def test_there_is_exactly_one_migration_head(tmp_path, monkeypatch):
    """Two heads mean a branched history that alembic refuses to upgrade, and
    it only says so at deploy time."""
    _upgraded(tmp_path, monkeypatch)
    heads = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_heads()
    assert len(heads) == 1, f"expected a single head, found {heads}"


def test_alembic_takes_its_url_from_the_application_settings():
    """One source of truth for which database is being migrated. An alembic.ini
    carrying its own URL is how a migration lands on the wrong one."""
    for line in ALEMBIC_INI.read_text().splitlines():
        if line.strip().startswith("sqlalchemy.url"):
            assert line.split("=", 1)[1].strip() == "", "alembic.ini must not hardcode a URL"
