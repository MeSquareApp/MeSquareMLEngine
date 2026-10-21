from pathlib import Path

from psycopg import sql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from config.settings import Settings
from src.components.data_transformation import (
    EXPECTED_RANGES,
    EXPECTED_ROW_COUNTS,
    ML_TABLES,
    NHANES_CORE_TABLES,
    SAS_DUST_THRESHOLD,
    Dataset,
    ImportResult,
    postgres_type,
)
from src.utils.exception import IngestError

DERIVED_SQL_DIR = Path(__file__).resolve().parents[2] / "sql" / "derived"
NHANES_CORE_SQL_PATH = DERIVED_SQL_DIR / "01_patient_features.sql"
CARDIOVASCULAR_SQL_PATH = DERIVED_SQL_DIR / "02_training_set.sql"
SQL_PATHS = (NHANES_CORE_SQL_PATH, CARDIOVASCULAR_SQL_PATH)

# Schemas left behind by earlier layouts.
_LEGACY_SCHEMAS = ("nhanes_core", "derived")

# Columns that should never retain SAS transport dust after load.
_DUST_CHECK_COLUMNS = (
    ("demographics", "riagendr"),
    ("demographics", "ridstatr"),
    ("blood_pressure_cholesterol_questionnaires", "bpq020"),
    ("blood_pressure_cholesterol_questionnaires", "bpq150"),
    ("smoking_questionnaires", "smq020"),
    ("diabetes_questionnaires", "diq010"),
)


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
        """Create the nhanes and ml schemas when they do not already exist."""
        with self.engine.begin() as connection:
            with _driver_connection(connection).cursor() as cursor:
                cursor.execute("CREATE SCHEMA IF NOT EXISTS nhanes")
                cursor.execute("CREATE SCHEMA IF NOT EXISTS ml")

    def drop_derived_tables(self) -> None:
        """Drop ml tables, and the earlier nhanes_core and derived schemas."""
        with self.engine.begin() as connection:
            with _driver_connection(connection).cursor() as cursor:
                for schema_name in _LEGACY_SCHEMAS:
                    cursor.execute(
                        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                            sql.Identifier(schema_name)
                        )
                    )
                for qualified in (*ML_TABLES, *reversed(NHANES_CORE_TABLES)):
                    _drop_relation(cursor, qualified)

    def load_derived_tables(self) -> None:
        """Rebuild ml.cardiovascular_training in one transaction.

        Patient-feature tables are temporary and are dropped when this transaction commits.
        """
        self.ensure_schema()
        self.drop_derived_tables()
        with self.engine.begin() as connection:
            for path in SQL_PATHS:
                if not path.is_file():
                    raise IngestError(f"Missing SQL file: {path}")
                for statement in _sql_statements(path.read_text(encoding="utf-8")):
                    connection.execute(text(statement))

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

    def verify_codebook(self) -> None:
        """Raise when loaded tables disagree with the cycle L codebook."""
        problems: list[str] = []
        with self.engine.connect() as connection:
            with _driver_connection(connection).cursor() as cursor:
                for table_name, expected in EXPECTED_ROW_COUNTS.items():
                    cursor.execute(
                        sql.SQL("SELECT count(*) FROM {}.{}").format(
                            sql.Identifier("nhanes"),
                            sql.Identifier(table_name),
                        )
                    )
                    actual = cursor.fetchone()[0]
                    if actual != expected:
                        problems.append(f"{table_name}: expected {expected} rows, found {actual}")

                for (table_name, column_name), (low, high) in EXPECTED_RANGES.items():
                    cursor.execute(
                        sql.SQL(
                            "SELECT min({column}), max({column}) "
                            "FROM {schema}.{table} "
                            "WHERE {column} IS NOT NULL"
                        ).format(
                            schema=sql.Identifier("nhanes"),
                            table=sql.Identifier(table_name),
                            column=sql.Identifier(column_name),
                        )
                    )
                    minimum, maximum = cursor.fetchone()
                    if minimum is None or maximum is None:
                        problems.append(f"{table_name}.{column_name}: no non-null values")
                        continue
                    if minimum < low - 1e-6 or maximum > high + 1e-6:
                        problems.append(
                            f"{table_name}.{column_name}: range [{minimum}, {maximum}] "
                            f"outside codebook [{low}, {high}]"
                        )

                for table_name, column_name in _DUST_CHECK_COLUMNS:
                    cursor.execute(
                        sql.SQL(
                            "SELECT count(*) "
                            "FROM {schema}.{table} "
                            "WHERE {column} IS NOT NULL "
                            "AND abs({column}) > 0 "
                            "AND abs({column}) < {threshold}"
                        ).format(
                            schema=sql.Identifier("nhanes"),
                            table=sql.Identifier(table_name),
                            column=sql.Identifier(column_name),
                            threshold=sql.Literal(SAS_DUST_THRESHOLD),
                        )
                    )
                    dust_count = cursor.fetchone()[0]
                    if dust_count:
                        problems.append(
                            f"{table_name}.{column_name}: {dust_count} SAS dust values remain"
                        )
        if problems:
            raise IngestError("Codebook verification failed: " + "; ".join(problems))

    def training_view_summary(self) -> dict[str, int]:
        """Return row counts for ml.cardiovascular_training, overall and with each target."""
        with self.engine.connect() as connection:
            with _driver_connection(connection).cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        count(*) AS rows,
                        count(non_hdl) AS non_hdl,
                        count(hdl) AS hdl,
                        count(hba1c) AS hba1c,
                        count(scr) AS scr
                    FROM ml.cardiovascular_training
                    """
                )
                rows, non_hdl, hdl, hba1c, scr = cursor.fetchone()
        return {
            "rows": rows,
            "non_hdl": non_hdl,
            "hdl": hdl,
            "hba1c": hba1c,
            "scr": scr,
        }


def _sql_statements(script: str) -> list[str]:
    """Split a migration script into executable statements, skipping comments."""
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip()
            buffer.clear()
            if statement:
                statements.append(statement)
    leftover = "\n".join(buffer).strip()
    if leftover:
        statements.append(leftover)
    return statements


def _drop_relation(cursor, qualified: str) -> None:
    """Drop a table or view when one exists under this schema-qualified name."""
    schema, name = qualified.split(".", 1)
    cursor.execute(
        """
        SELECT c.relkind
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = %s
        """,
        (schema, name),
    )
    row = cursor.fetchone()
    if row is None:
        return
    if row[0] == "v":
        statement = "DROP VIEW IF EXISTS {}.{} "
    elif row[0] == "m":
        statement = "DROP MATERIALIZED VIEW IF EXISTS {}.{} "
    else:
        statement = "DROP TABLE IF EXISTS {}.{} "
    cursor.execute(
        sql.SQL(statement).format(sql.Identifier(schema), sql.Identifier(name))
    )


def _driver_connection(connection: Connection):
    """Return the psycopg connection behind a SQLAlchemy connection."""
    return connection.connection.driver_connection
