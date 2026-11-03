"""Online prediction for quantile biomarker models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.components.feature_engineering import prepare_features, records_to_frame
from src.components.model_artifact import QuantileModelBundle
from src.components.model_config import QUANTILE_LABELS, QUANTILES, TARGET_UNITS, TARGETS
from src.components.model_trainer import predict_quantiles


class QuantilePredictor:
    def __init__(self, bundle: QuantileModelBundle) -> None:
        self.bundle = bundle

    @property
    def version(self) -> str:
        return self.bundle.version

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one prediction dict per patient with p10/p50/p90 for each target."""
        frame = records_to_frame(records)
        features = prepare_features(frame, self.bundle.category_levels)
        results: list[dict[str, Any]] = []

        # Precompute predictions per target for the whole batch.
        pred_by_target = {
            target: predict_quantiles(
                self.bundle.models[target],
                features,
                self.bundle.quantiles or QUANTILES,
            )
            for target in self.bundle.targets or TARGETS
        }

        quantiles = self.bundle.quantiles or QUANTILES
        for i in range(len(records)):
            predictions: dict[str, dict[str, float | str]] = {}
            for target in self.bundle.targets or TARGETS:
                row = pred_by_target[target][i]
                entry: dict[str, float | str] = {"unit": TARGET_UNITS[target]}
                for j, tau in enumerate(quantiles):
                    label = QUANTILE_LABELS.get(tau, f"p{int(tau * 100)}")
                    entry[label] = float(row[j])
                predictions[target] = entry
            results.append({"predictions": predictions})
        return results

    def predict_quantiles(self, profile: pd.DataFrame) -> dict[str, dict[float, float]]:
        """Quantiles for the first row of profile, keyed by target then tau. Used by the twin."""
        features = prepare_features(profile.iloc[[0]], self.bundle.category_levels)
        quantiles = self.bundle.quantiles or QUANTILES
        result: dict[str, dict[float, float]] = {}
        for target in self.bundle.targets or TARGETS:
            row = predict_quantiles(self.bundle.models[target], features, quantiles)[0]
            result[target] = {tau: float(value) for tau, value in zip(quantiles, row, strict=True)}
        return result
