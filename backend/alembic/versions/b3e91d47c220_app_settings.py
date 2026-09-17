"""portal settings edited in the browser

Revision ID: b3e91d47c220
Revises: ac7fba0af5da
Create Date: 2026-09-17 11:04:22.118430

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b3e91d47c220'
down_revision: Union[str, Sequence[str], None] = 'ac7fba0af5da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """One row per setting group, value as JSON.

    Key/value rather than a column per setting: these are portal preferences that
    come and go, and a migration for each new one buys nothing. The first user is
    the LLM that drafts catalog entries from a model-card URL.
    """
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column(
            "value",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
