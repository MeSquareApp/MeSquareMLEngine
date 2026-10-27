from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]


def sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        rewritten = "postgresql+psycopg://" + database_url.removeprefix("postgresql+asyncpg://")
        return _asyncpg_query_for_psycopg(rewritten)
    if database_url.startswith("postgresql+"):
        return database_url
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    return database_url


def _asyncpg_query_for_psycopg(database_url: str) -> str:
    parts = urlsplit(database_url)
    query = [
        ("sslmode", value) if key == "ssl" else (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))


class Settings(BaseSettings):
    # Optional so the prediction API can start without DB credentials.
    database_url: str | None = None
    nhanes_storage_base_url: str | None = None
    model_dir: Path = Field(default_factory=lambda: _ROOT / "artifacts" / "quantile")

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def sqlalchemy_database_url(self) -> str:
        if not self.database_url:
            raise ValueError("DATABASE_URL is not configured")
        return sqlalchemy_url(self.database_url)


settings = Settings()
