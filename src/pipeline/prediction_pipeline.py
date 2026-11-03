"""Online prediction for quantile biomarker models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.components.feature_engineering import (
    FeatureValue,
    FloatArray,
    prepare_features,
    records_to_frame,
)
from src.components.model_artifact import QuantileModelBundle
from src.components.model_config import QUANTILE_LABELS, TARGET_UNITS
from src.components.model_evaluation import widen_interval
from src.components.model_trainer import predict_quantiles

PatientRecord = Mapping[str, FeatureValue]
TargetPrediction = dict[str, float | str]


class QuantilePredictor:
    """Population quantile predictions from a trained bundle."""

    def __init__(self, bundle: QuantileModelBundle) -> None:
        """Wrap a loaded bundle.

        Args:
            bundle: Trained models and their encoding metadata.
        """
        self.bundle = bundle

    @property
    def version(self) -> str:
        """Version id of the loaded bundle."""
        return self.bundle.version

    def predict(
        self, records: Sequence[PatientRecord]
    ) -> list[dict[str, dict[str, TargetPrediction]]]:
        """Quantiles per target for a batch of patients.

        Args:
            records: Patient features with NHANES codes.

        Returns:
            One {"predictions": {target: {"unit", "p10", "p50", "p90"}}} per patient.
        """
        matrices = self._quantile_matrices(records)
        labels = [
            QUANTILE_LABELS.get(tau, f"p{round(tau * 100)}")
            for tau in self.bundle.quantiles
        ]
        return [
            {
                "predictions": {
                    target: {
                        "unit": TARGET_UNITS[target],
                        **dict(zip(labels, map(float, matrix[row]), strict=True)),
                    }
                    for target, matrix in matrices.items()
                }
            }
            for row in range(len(records))
        ]

    def predict_quantiles(self, record: PatientRecord) -> dict[str, dict[float, float]]:
        """Population quantiles for one patient. Used by the digital twin.

        Args:
            record: Patient features with NHANES codes.

        Returns:
            Quantile values keyed by target, then tau.
        """
        return {
            target: dict(zip(self.bundle.quantiles, map(float, matrix[0]), strict=True))
            for target, matrix in self._quantile_matrices([record]).items()
        }

    def _quantile_matrices(
        self, records: Sequence[PatientRecord]
    ) -> dict[str, FloatArray]:
        """Calibrated quantile matrix of shape (patients, quantiles) per target."""
        features = prepare_features(
            records_to_frame(records), self.bundle.category_levels
        )
        quantiles = self.bundle.quantiles
        return {
            target: widen_interval(
                predict_quantiles(self.bundle.models[target], features, quantiles),
                quantiles,
                self.bundle.interval_widening.get(target, 0.0),
            )
            for target in self.bundle.targets
        }
