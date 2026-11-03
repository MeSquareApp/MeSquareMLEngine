"""Persist and load trained quantile model bundles."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import orjson

from src.components.model_config import FEATURES, QUANTILES, TARGETS, CategoryLevels
from src.utils.exception import MeSquareError

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
GIT_TIMEOUT_SECONDS = 5

ModelsByTarget = dict[str, dict[float, lgb.LGBMRegressor]]
Metrics = dict[str, object]


class ArtifactError(MeSquareError):
    """Raised when a model artifact cannot be saved or loaded."""


@dataclass
class QuantileModelBundle:
    """Trained models plus everything needed to reproduce their inputs.

    Attributes:
        models: Fitted models keyed by target, then tau.
        category_levels: Ordered levels used to encode categorical features.
        feature_list: Feature order of the model input matrix.
        version: Version id, also the folder name under the model directory.
        metrics: Cross-validation and holdout metrics.
        quantiles: Quantile levels of every target.
        targets: Biomarkers with models.
        training_row_counts: Labeled rows per target.
        interval_widening: Conformal widening of the p10-p90 interval per target.
        git_sha: Commit the models were trained from, if known.
        trained_at: ISO timestamp of training.
    """

    models: ModelsByTarget
    category_levels: CategoryLevels
    feature_list: list[str]
    version: str
    metrics: Metrics = field(default_factory=dict)
    quantiles: tuple[float, ...] = QUANTILES
    targets: tuple[str, ...] = TARGETS
    training_row_counts: dict[str, int] = field(default_factory=dict)
    interval_widening: dict[str, float] = field(default_factory=dict)
    git_sha: str | None = None
    trained_at: str | None = None

    def save(self, root: Path) -> Path:
        """Write model.joblib and metadata.json under root/<version>/.

        Args:
            root: Model directory.

        Returns:
            The version folder.
        """
        out_dir = Path(root) / self.version
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": self.version,
            "trained_at": self.trained_at,
            "git_sha": self.git_sha,
            "targets": list(self.targets),
            "quantiles": list(self.quantiles),
            "features": self.feature_list,
            "category_levels": self.category_levels,
            "training_row_counts": self.training_row_counts,
            "interval_widening": self.interval_widening,
            "metrics": self.metrics,
        }
        joblib.dump({**metadata, "models": self.models}, out_dir / MODEL_FILENAME)
        (out_dir / METADATA_FILENAME).write_bytes(
            orjson.dumps(
                metadata,
                option=orjson.OPT_INDENT_2
                | orjson.OPT_APPEND_NEWLINE
                | orjson.OPT_NON_STR_KEYS
                | orjson.OPT_SERIALIZE_NUMPY,
                default=str,
            )
        )
        return out_dir

    @classmethod
    def load(cls, directory: Path) -> QuantileModelBundle:
        """Load a bundle from one version folder.

        Args:
            directory: Folder containing model.joblib.

        Returns:
            The loaded bundle.

        Raises:
            ArtifactError: If model.joblib is missing.
        """
        path = Path(directory) / MODEL_FILENAME
        if not path.exists():
            raise ArtifactError(f"No model artifact at {path}")
        payload = joblib.load(path)
        return cls(
            models={
                target: {float(tau): model for tau, model in tau_models.items()}
                for target, tau_models in payload["models"].items()
            },
            category_levels=payload["category_levels"],
            feature_list=list(payload.get("features") or payload["feature_list"]),
            version=str(payload["version"]),
            metrics=dict(payload.get("metrics") or {}),
            quantiles=tuple(float(tau) for tau in payload["quantiles"]),
            targets=tuple(payload.get("targets") or TARGETS),
            training_row_counts=dict(payload.get("training_row_counts") or {}),
            interval_widening=dict(payload.get("interval_widening") or {}),
            git_sha=payload.get("git_sha"),
            trained_at=payload.get("trained_at"),
        )

    @classmethod
    def load_latest(cls, root: Path) -> QuantileModelBundle:
        """Load the newest version under root. Version ids sort chronologically.

        Args:
            root: Model directory.

        Returns:
            The newest bundle.

        Raises:
            ArtifactError: If root is missing or holds no bundles.
        """
        root = Path(root)
        if not root.exists():
            raise ArtifactError(f"Model directory does not exist: {root}")
        candidates = sorted(
            path for path in root.iterdir() if (path / MODEL_FILENAME).exists()
        )
        if not candidates:
            raise ArtifactError(f"No model artifacts found under {root}")
        return cls.load(candidates[-1])


def new_version_id() -> str:
    """UTC timestamp version id, for example 20261103T1542Z."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%MZ")


def resolve_git_sha() -> str | None:
    """Short commit hash of the working tree, or None outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() or None


def build_bundle(
    models: ModelsByTarget,
    category_levels: CategoryLevels,
    metrics: Metrics,
    training_row_counts: dict[str, int],
    version: str | None = None,
) -> QuantileModelBundle:
    """Wrap trained models with version, git, and timestamp metadata.

    Args:
        models: Fitted models keyed by target, then tau.
        category_levels: Levels used to encode categorical features.
        metrics: Training metrics.
        training_row_counts: Labeled rows per target.
        version: Version id. Defaults to the current UTC timestamp.

    Returns:
        The bundle, not yet saved.
    """
    return QuantileModelBundle(
        models=models,
        category_levels=category_levels,
        feature_list=list(FEATURES),
        version=version or new_version_id(),
        metrics=metrics,
        training_row_counts=training_row_counts,
        git_sha=resolve_git_sha(),
        trained_at=datetime.now(UTC).isoformat(),
    )
