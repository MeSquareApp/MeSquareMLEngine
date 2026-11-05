from contextlib import contextmanager
from datetime import date

from config.settings import Settings
from src.pipeline.twin_pipeline import ShadowPrediction
from src.repository.shadow_store import ShadowBatch, ShadowStore


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list = []

    def execute(self, statement, rows) -> None:
        self.calls.append((str(statement), rows))


class FakeEngine:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    @contextmanager
    def begin(self):
        yield self.connection


def test_write_sends_one_row_per_shadow_prediction():
    # Arrange
    engine = FakeEngine()
    store = ShadowStore(engine)  # type: ignore[arg-type]
    batch = ShadowBatch(
        user_id="user-1",
        as_of=date(2026, 3, 1),
        prior_version="prior-1",
        input_hash="sha256:abc",
        predictions=[
            ShadowPrediction("hba1c", "change", (5.6, 5.9, 6.2), (5.5, 5.8, 6.1), "r1")
        ],
    )

    # Act
    written = store.write(batch)

    # Assert
    statement, rows = engine.connection.calls[0]
    assert written == 1
    assert "ml.shadow_predictions" in statement
    assert "ON CONFLICT" in statement
    assert rows[0]["served_band"] == [5.6, 5.9, 6.2]
    assert rows[0]["mode"] == "change"


def test_empty_batch_does_not_touch_the_database():
    # Arrange
    engine = FakeEngine()
    store = ShadowStore(engine)  # type: ignore[arg-type]
    batch = ShadowBatch("user-1", date(2026, 3, 1), "prior-1", "sha256:abc", [])

    # Act
    written = store.write(batch)

    # Assert
    assert written == 0
    assert engine.connection.calls == []


def test_no_store_without_database_url():
    # Arrange
    app_settings = Settings(database_url=None)

    # Act
    store = ShadowStore.from_settings(app_settings)

    # Assert
    assert store is None
