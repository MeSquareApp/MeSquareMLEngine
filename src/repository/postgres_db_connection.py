from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from config.settings import Settings
from src.components.data_transformation import Dataset, ImportResult, postgres_type
from src.utils.exception import IngestError


class NhanesRepository:
    def __init__(self, engine: Engine) -> None:
        """Keep the SQLAlchemy engine used for later queries."""
        self.engine = engine

    @classmethod
    def from_settings(cls, app_settings: Settings) -> "NhanesRepository":
        """Open a Postgres connection pool from app settings."""
        engine = create_engine(app_settings.sqlalchemy_database_url, pool_pre_ping=True)
        return cls(engine)

    def close(self) -> None:
        """Close the connection pool."""
        self.engine.dispose()

    def ensure_schema(self) -> None:
        """Create the nhanes schema when it does not already exist."""
        with self.engine.begin() as connection:
            with _driver_connection(connection).cursor() as cursor:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS nhanes")

    def replace(self, dataset: Dataset) -> ImportResult:
        """Drop and recreate one table, then load its rows, in a single transaction."""
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
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier("nhanes"),
                        sql.Identifier(name),
                    )
                )
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
        """Raise when a loaded table's row count does not match the source dataset."""
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
            raise IngestError("Row count mismatch: " + "; ".join(mismatches))


def _driver_connection(connection: Connection):
    """Return the psycopg connection behind a SQLAlchemy connection."""
    return connection.connection.driver_connection
