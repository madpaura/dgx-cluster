"""gpu sharing: per-deployment vram reservation

Revision ID: ac7fba0af5da
Revises: 967bc7ac782d
Create Date: 2026-09-13 09:19:57.955107

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ac7fba0af5da'
down_revision: Union[str, Sequence[str], None] = '967bc7ac782d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the VRAM a deployment holds on each of its GPUs.

    server_default is required: the column is NOT NULL and the table has rows
    on any existing deployment. Zero is deliberate rather than a placeholder —
    it means "share unknown", and the capacity ledger reads that as the whole
    card, so a model placed before dgxctl tracked shares keeps its GPU to
    itself instead of having a second model scheduled on top of it.
    """
    op.add_column(
        "deployments",
        sa.Column("reserved_mb_per_gpu", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    """Drop the reservation column."""
    op.drop_column('deployments', 'reserved_mb_per_gpu')
