from pathlib import Path

import pandas as pd

from config.settings import Settings, settings
from src.components.egfr import add_egfr
from src.components.prior_model import ARTIFACT_DIR, PriorModel, train_prior_models
from src.repository.postgres_db_connection import NhanesRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_training_frame(app_settings: Settings | None = None) -> pd.DataFrame:
    repository = NhanesRepository.from_settings(app_settings or settings)
    try:
        frame = pd.read_sql_query("SELECT * FROM ml.cardiovascular_training", con=repository.engine)
    finally:
        repository.close()
    return add_egfr(frame).drop(columns="scr")


def train_prior(
    app_settings: Settings | None = None, directory: Path = ARTIFACT_DIR
) -> PriorModel:
    frame = load_training_frame(app_settings)
    logger.info("Training prior quantile models on %d rows", len(frame))
    prior = train_prior_models(frame)
    path = prior.save(directory)
    logger.info("Saved prior models to %s", path)
    return prior
