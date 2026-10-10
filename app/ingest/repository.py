from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from app.core.config import Settings
from app.ingest.datasets import Dataset, ImportResult, postgres_type


class NhanesRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def from_settings(cls, app_settings: Settings) -> "NhanesRepository":
        engine = create_engine(app_settings.sqlalchemy_database_url, pool_pre_ping=True)
        return cls(engine)

    def close(self) -> None:
        self.engine.dispose()

    def ensure_schema(self) -> None:
        with self.engine.begin() as connection:
            with _driver_connection(connection).cursor() as cursor:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS nhanes")

    def replace(self, dataset: Dataset) -> ImportResult:
        name = dataset.table_name
        use_primary_key = dataset.uses_seqn_primary_key()
        column_defs: list[sql.Composable] = []
        for column in dataset.columns:
            definition = sql.SQL("{} {}").format(
                sql.Identifier(column.name),
                sql.SQL(postgres_type(column.sas_type)),
            )
            if use_primary_key and column.name == "seqn":
                definition = sql.SQL("{} PRIMARY KEY").format(definition)
            column_defs.append(definition)

        with self.engine.begin() as connection:
            with _driver_connection(connection).cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE TABLE {}.{} ({})").format(
                        sql.Identifier("nhanes"),
                        sql.Identifier(name),
                        sql.SQL(", ").join(column_defs),
                    )
                )
                cursor.execute(
                    sql.SQL("ALTER TABLE {}.{} ENABLE ROW LEVEL SECURITY").format(
                        sql.Identifier("nhanes"),
                        sql.Identifier(name),
                    )
                )
                copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
                    sql.Identifier("nhanes"),
                    sql.Identifier(name),
                    sql.SQL(", ").join(sql.Identifier(column.name) for column in dataset.columns),
                )
                with cursor.copy(copy_sql) as copy:
                    for record in dataset.copy_records():
                        copy.write_row(record)
        return ImportResult(filename=dataset.filename, table_name=name, row_count=len(dataset.frame))

    def verify(self, results: list[ImportResult]) -> None:
        mismatches: list[str] = []
        with self.engine.connect() as connection:
            with _driver_connection(connection).cursor() as cursor:
                for result in results:
                    cursor.execute(
                        sql.SQL("SELECT count(*) FROM {}.{}").format(
                            sql.Identifier("nhanes"),
                            sql.Identifier(result.table_name),
                        )
                    )
                    actual = cursor.fetchone()[0]
                    if actual != result.row_count:
                        mismatches.append(
                            f"{result.table_name}: loaded {result.row_count}, table has {actual}"
                        )
        if mismatches:
            raise RuntimeError("Row count mismatch: " + "; ".join(mismatches))


def _driver_connection(connection: Connection):
    return connection.connection.driver_connection
