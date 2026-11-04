"""Load NHANES cycle L files and rebuild the training tables."""

from config.settings import Settings, settings
from src.components.data_ingestion import fetch_dataset
from src.components.data_transformation import NHANES_FILENAMES, ImportResult
from src.repository.postgres_db_connection import NhanesRepository
from src.utils.exception import IngestError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_nhanes(app_settings: Settings | None = None) -> list[ImportResult]:
    """Replace NHANES tables, verify them, and rebuild derived tables."""
    app_settings = app_settings or settings
    repository = NhanesRepository.from_settings(app_settings)
    try:
        logger.info("Loading NHANES datasets into schema nhanes")
        repository.ensure_schema()
        repository.drop_derived_tables()
        base_url = app_settings.nhanes_storage_base_url
        if not base_url:
            raise IngestError("NHANES storage base URL is not configured")
        results = [
            repository.replace(fetch_dataset(base_url, filename))
            for filename in NHANES_FILENAMES
        ]
        repository.verify(results)
        repository.verify_codebook()
        repository.load_derived_tables()
        summary = repository.training_view_summary()
        logger.info(
            "cardiovascular_training rows=%s non_hdl=%s hdl=%s hba1c=%s scr=%s",
            summary["rows"],
            summary["non_hdl"],
            summary["hdl"],
            summary["hba1c"],
            summary["scr"],
        )
        logger.info("Loaded %s NHANES tables and derived tables", len(results))
        return results
    finally:
        repository.close()
