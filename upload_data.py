from src.pipeline.ingest_pipeline import load_nhanes


def main() -> None:
    for result in load_nhanes():
        print(f"{result.filename} -> nhanes.{result.table_name}: {result.row_count} rows")


if __name__ == "__main__":
    main()
