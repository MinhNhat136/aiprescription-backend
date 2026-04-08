"""Initial migration: create prescriptions table.

Revision ID: 001_initial_prescriptions
Revises:
Create Date: 2026-04-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_prescriptions"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the prescriptions table with all required columns."""
    op.create_table(
        "prescriptions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("patient_name", sa.String(length=255), nullable=False),
        sa.Column("diagnosis", sa.Text(), nullable=True),
        sa.Column("medications", sa.Text(), nullable=True),
        # JSON string storing: [{"name": "...", "dosage": "...", "instructions": "..."}]
        sa.Column("instructions", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("doctor_id", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    # Create index on created_at for efficient time-based queries
    op.create_index("ix_prescriptions_created_at", "prescriptions", ["created_at"])
    # Create index on patient_name for search functionality
    op.create_index("ix_prescriptions_patient_name", "prescriptions", ["patient_name"])


def downgrade() -> None:
    """Drop the prescriptions table."""
    op.drop_index("ix_prescriptions_patient_name", table_name="prescriptions")
    op.drop_index("ix_prescriptions_created_at", table_name="prescriptions")
    op.drop_table("prescriptions")
