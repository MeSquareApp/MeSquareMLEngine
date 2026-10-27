"""Persist and load trained quantile model bundles."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from src.components.model_config import FEATURES, QUANTILES, TARGETS
from src.utils.exception import MeSquareError


class ArtifactError(MeSquareError):
    """Raised when a model artifact cannot be saved or loaded."""


MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass
class QuantileModelBundle:
    models: dict[str, dict[float, lgb.LGBMRegressor]]
    category_levels: dict[str, list]
    feature_list: list[str]
    version: str
    metrics: dict[str, Any] = field(default_factory=dict)
    quantiles: tuple[float, ...] = QUANTILES
    targets: tuple[str, ...] = TARGETS
    training_row_counts: dict[str, int] = field(default_factory=dict)
    git_sha: str | None = None
    trained_at: str | None = None

    def save(self, root: Path) -> Path:
        """Write model.joblib and metadata.json under root/<version>/."""
        out_dir = Path(root) / self.version
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "models": self.models,
            "category_levels": self.category_levels,
            "feature_list": self.feature_list,
            "version": self.version,
            "quantiles": list(self.quantiles),
            "targets": list(self.targets),
            "training_row_counts": self.training_row_counts,
            "git_sha": self.git_sha,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }
        joblib.dump(payload, out_dir / MODEL_FILENAME)

        metadata = {
            "version": self.version,
            "trained_at": self.trained_at,
            "git_sha": self.git_sha,
            "targets": list(self.targets),
            "quantiles": list(self.quantiles),
            "features": self.feature_list,
            "category_levels": self.category_levels,
            "training_row_counts": self.training_row_counts,
            "metrics": self.metrics,
        }
        (out_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return out_dir

    @classmethod
    def load(cls, directory: Path) -> "QuantileModelBundle":
        path = Path(directory) / MODEL_FILENAME
        if not path.exists():
            raise ArtifactError(f"No model artifact at {path}")
        payload = joblib.load(path)
        quantiles = tuple(float(q) for q in payload["quantiles"])
        models: dict[str, dict[float, lgb.LGBMRegressor]] = {}
        for target, tau_models in payload["models"].items():
            models[target] = {float(tau): model for tau, model in tau_models.items()}
        return cls(
            models=models,
            category_levels=payload["category_levels"],
            feature_list=list(payload["feature_list"]),
            version=str(payload["version"]),
            metrics=dict(payload.get("metrics") or {}),
            quantiles=quantiles,
            targets=tuple(payload.get("targets") or TARGETS),
            training_row_counts=dict(payload.get("training_row_counts") or {}),
            git_sha=payload.get("git_sha"),
            trained_at=payload.get("trained_at"),
        )

    @classmethod
    def load_latest(cls, root: Path) -> "QuantileModelBundle":
        root = Path(root)
        if not root.exists():
            raise ArtifactError(f"Model directory does not exist: {root}")
        candidates = sorted(
            (p for p in root.iterdir() if p.is_dir() and (p / MODEL_FILENAME).exists()),
            key=lambda p: p.name,
        )
        if not candidates:
            raise ArtifactError(f"No model artifacts found under {root}")
        return cls.load(candidates[-1])


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")


def resolve_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_bundle(
    models: dict[str, dict[float, lgb.LGBMRegressor]],
    category_levels: dict[str, list],
    metrics: dict[str, Any],
    training_row_counts: dict[str, int],
    version: str | None = None,
) -> QuantileModelBundle:
    return QuantileModelBundle(
        models=models,
        category_levels=category_levels,
        feature_list=list(FEATURES),
        version=version or new_version_id(),
        metrics=metrics,
        training_row_counts=training_row_counts,
        git_sha=resolve_git_sha(),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
