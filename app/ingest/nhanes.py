from app.core.config import Settings, settings
from app.ingest.datasets import NHANES_FILENAMES, ImportResult
from app.ingest.reader import fetch_dataset
from app.ingest.repository import NhanesRepository


def load_nhanes(app_settings: Settings | None = None) -> list[ImportResult]:
    app_settings = app_settings or settings
    repository = NhanesRepository.from_settings(app_settings)
    try:
        repository.ensure_schema()
        results = [
            repository.replace(fetch_dataset(app_settings.nhanes_storage_base_url, filename))
            for filename in NHANES_FILENAMES
        ]
        repository.verify(results)
        return results
    finally:
        repository.close()


def main() -> None:
    for result in load_nhanes():
        print(f"{result.filename} -> nhanes.{result.table_name}: {result.row_count} rows")


if __name__ == "__main__":
    main()
