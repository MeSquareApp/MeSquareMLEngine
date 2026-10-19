from config.settings import Settings, settings
from src.components.data_ingestion import fetch_dataset
from src.components.data_transformation import NHANES_FILENAMES, ImportResult
from src.repository.postgres_db_connection import NhanesRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_nhanes(app_settings: Settings | None = None) -> list[ImportResult]:
    app_settings = app_settings or settings
    repository = NhanesRepository.from_settings(app_settings)
    try:
        logger.info("Loading NHANES datasets into schema nhanes")
        repository.ensure_schema()
        results = [
            repository.replace(fetch_dataset(app_settings.nhanes_storage_base_url, filename))
            for filename in NHANES_FILENAMES
        ]
        repository.verify(results)
        logger.info("Loaded %s NHANES tables", len(results))
        return results
    finally:
        repository.close()
