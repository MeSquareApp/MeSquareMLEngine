"""Write shadow predictions to ml.shadow_predictions.

The backend calls /v1/twin/state nightly after each wearable sync, with a user_id.
Wearable-adjusted bands from that call land here instead of in the response, so the
activation gate can later compare them with real labs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Engine, create_engine, text

from config.settings import Settings
from src.pipeline.twin_pipeline import ShadowPrediction

INSERT_SHADOW = text(
    """
    INSERT INTO ml.shadow_predictions (
        user_id, analyte, mode, as_of, served_band, shadow_band,
        residual_model_version, prior_version, input_hash
    )
    VALUES (
        :user_id, :analyte, :mode, :as_of, :served_band, :shadow_band,
        :residual_model_version, :prior_version, :input_hash
    )
    ON CONFLICT (user_id, analyte, mode, as_of, input_hash) DO NOTHING
    """
)


@dataclass(frozen=True)
class ShadowBatch:
    """Shadow predictions of one twin state call.

    Attributes:
        user_id: Backend user id.
        as_of: Day of the state.
        prior_version: Prior model version.
        input_hash: The state's input hash.
        predictions: Wearable-adjusted bands that were not served.
    """

    user_id: str
    as_of: date
    prior_version: str
    input_hash: str
    predictions: Sequence[ShadowPrediction]


class ShadowStore:
    """Inserts shadow predictions. Re-sending the same batch is a no-op."""

    def __init__(self, engine: Engine) -> None:
        """Keep the SQLAlchemy engine.

        Args:
            engine: Engine connected to the MeSquare database.
        """
        self.engine = engine

    @classmethod
    def from_settings(cls, app_settings: Settings) -> ShadowStore | None:
        """Store for DATABASE_URL, or None when no database is configured.

        Args:
            app_settings: App settings.

        Returns:
            The store, or None.
        """
        if not app_settings.database_url:
            return None
        url = app_settings.sqlalchemy_database_url
        return cls(create_engine(url, pool_pre_ping=True))

    def write(self, batch: ShadowBatch) -> int:
        """Insert one row per shadow prediction.

        Args:
            batch: Shadow predictions of one call.

        Returns:
            Rows sent (duplicates are skipped by the database).
        """
        rows = [
            {
                "user_id": batch.user_id,
                "analyte": item.analyte,
                "mode": item.mode,
                "as_of": batch.as_of,
                "served_band": list(item.served),
                "shadow_band": list(item.shadow),
                "residual_model_version": item.model_version,
                "prior_version": batch.prior_version,
                "input_hash": batch.input_hash,
            }
            for item in batch.predictions
        ]
        if rows:
            with self.engine.begin() as connection:
                connection.execute(INSERT_SHADOW, rows)
        return len(rows)
