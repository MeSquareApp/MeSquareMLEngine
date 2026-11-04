"""Wearable residual model and its activation gate (goal 6).

The model predicts how far a user's next lab lands from the stage below it, on the
log scale, from wearable window features:

- "level" mode, for wearables-only users: residual against the population p50.
- "change" mode, for lab-first users with enough wearable data after the lab:
  residual against the last lab.

It is served only after gate_residual_model() passes on real consented pairs. Until
then its output goes to shadow predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold

from src.components.model_evaluation import conformal_widening, fix_crossing
from src.utils.exception import MeSquareError

FloatArray = npt.NDArray[np.float64]
ResidualMode = Literal["level", "change"]

QUANTILES = (0.10, 0.50, 0.90)
MIN_DAYS_AFTER_LAB = 60
# Pre-set sample size: the gate refuses to judge with fewer users.
GATE_MIN_USERS = 200
GATE_MIN_IMPROVEMENT = 0.10
GATE_COVERAGE = (0.75, 0.85)
GATE_FOLDS = 5
BOOTSTRAP_DRAWS = 2000
RANDOM_STATE = 42
MODEL_FILENAME = "residual.joblib"
RESIDUAL_TREES = 200
RESIDUAL_LEARNING_RATE = 0.05
RESIDUAL_LEAVES = 7
RESIDUAL_MIN_CHILD_SAMPLES = 20


class ResidualModelError(MeSquareError):
    """Raised when a residual model cannot be trained, saved, or loaded."""


@dataclass(frozen=True)
class PairSet:
    """Paired data: a wearable window, then a later real lab, per row.

    Attributes:
        user_ids: User of every row. CV never splits a user.
        features: Wearable window features, NaN where a window was insufficient.
        anchor: Natural-unit centre of the stage below (last lab or population p50).
        next_lab: The later real lab, natural units.
        baselines: Baseline (p10, p50, p90) bands to beat, keyed "last_lab" and/or
            "population", each shaped (rows, 3).
    """

    user_ids: npt.NDArray[np.str_]
    features: FloatArray
    anchor: FloatArray
    next_lab: FloatArray
    baselines: dict[str, FloatArray]


@dataclass(frozen=True)
class GateResult:
    """Outcome of the activation gate.

    Attributes:
        passed: True only when every check passed.
        reasons: Failed checks, empty when passed.
        metrics: Losses, improvements, CI and coverage behind the decision.
    """

    passed: bool
    reasons: list[str]
    metrics: dict[str, float]


@dataclass
class ResidualModel:
    """Quantile models of the log residual for one analyte and mode.

    Attributes:
        analyte: Canonical analyte name.
        mode: "level" or "change".
        models: Fitted LightGBM models keyed by tau.
        version: Version id reported in model_versions.
        widening: Conformal widening of the log-scale p10-p90 band, from the
            model's own out-of-fold predictions.
    """

    analyte: str
    mode: ResidualMode
    models: dict[float, lgb.LGBMRegressor]
    version: str
    widening: float = 0.0

    def predict_band(self, features: FloatArray, anchor: FloatArray) -> FloatArray:
        """Adjusted (p10, p50, p90) bands in natural units.

        Args:
            features: Feature matrix, shape (rows, features).
            anchor: Natural-unit centre of the stage below, per row.

        Returns:
            Bands of shape (rows, 3).
        """
        residuals = widen(
            fix_crossing(
                np.column_stack(
                    [self.models[tau].predict(features) for tau in QUANTILES]
                )
            ),
            self.widening,
        )
        return np.exp(np.log(anchor)[:, None] + residuals)

    def save(self, root: Path) -> Path:
        """Write the model under root/<analyte>_<mode>/.

        Args:
            root: Residual model directory.

        Returns:
            The model folder.
        """
        folder = root / f"{self.analyte}_{self.mode}"
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, folder / MODEL_FILENAME)
        return folder


def load_residual_models(root: Path) -> dict[tuple[str, str], ResidualModel]:
    """Every saved residual model under root, keyed (analyte, mode).

    Args:
        root: Residual model directory. A missing directory means no models.

    Returns:
        Loaded models.

    Raises:
        ResidualModelError: If a saved file is not a ResidualModel.
    """
    models: dict[tuple[str, str], ResidualModel] = {}
    for path in sorted(root.glob(f"*/{MODEL_FILENAME}")) if root.exists() else []:
        model = joblib.load(path)
        if not isinstance(model, ResidualModel):
            raise ResidualModelError(f"{path} does not hold a ResidualModel")
        models[model.analyte, model.mode] = model
    return models


def fit_quantile_residuals(
    features: FloatArray, residuals: FloatArray
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one LightGBM quantile model per tau to log residuals.

    Args:
        features: Feature matrix.
        residuals: log(next_lab) - log(anchor).

    Returns:
        Models keyed by tau.
    """
    return {
        tau: lgb.LGBMRegressor(
            objective="quantile",
            alpha=tau,
            n_estimators=RESIDUAL_TREES,
            learning_rate=RESIDUAL_LEARNING_RATE,
            num_leaves=RESIDUAL_LEAVES,
            min_child_samples=RESIDUAL_MIN_CHILD_SAMPLES,
            random_state=RANDOM_STATE,
            verbose=-1,
        ).fit(features, residuals)
        for tau in QUANTILES
    }


def mean_pinball(bands: FloatArray, observed: FloatArray) -> FloatArray:
    """Per-row pinball loss averaged over p10, p50, p90, in natural units.

    Args:
        bands: Predicted (p10, p50, p90), shape (rows, 3).
        observed: Observed values.

    Returns:
        Loss per row.
    """
    taus = np.array(QUANTILES)
    error = observed[:, None] - bands
    return np.mean(np.maximum(taus * error, (taus - 1) * error), axis=1)


def widen(residuals: FloatArray, widening: float) -> FloatArray:
    """Move the p10 and p90 residual columns apart by a conformal widening.

    Args:
        residuals: Log-residual quantiles, shape (rows, 3).
        widening: Amount for each side.

    Returns:
        Widened residuals, still non-crossing.
    """
    widened = residuals + np.array([-widening, 0.0, widening])
    return fix_crossing(widened)


def _calibration_widening(residuals: FloatArray, predicted: FloatArray) -> float:
    """CQR widening of predicted log-residual quantiles against actual residuals."""
    weights = np.ones(len(residuals))
    return conformal_widening(residuals, predicted[:, 0], predicted[:, 2], weights)


def _out_of_fold_residuals(
    pairs: PairSet,
) -> tuple[FloatArray, npt.NDArray[np.int_]]:
    """Raw residual quantiles from models that never saw each row's user."""
    residuals = np.log(pairs.next_lab) - np.log(pairs.anchor)
    predicted = np.empty((len(residuals), len(QUANTILES)))
    fold_ids = np.empty(len(residuals), dtype=np.int_)
    folds = GroupKFold(n_splits=GATE_FOLDS).split(pairs.features, groups=pairs.user_ids)
    for fold, (train_rows, test_rows) in enumerate(folds):
        models = fit_quantile_residuals(
            pairs.features[train_rows], residuals[train_rows]
        )
        predicted[test_rows] = fix_crossing(
            np.column_stack(
                [models[tau].predict(pairs.features[test_rows]) for tau in QUANTILES]
            )
        )
        fold_ids[test_rows] = fold
    return predicted, fold_ids


def out_of_fold_bands(pairs: PairSet) -> FloatArray:
    """Calibrated model bands for every row, never fitted or calibrated on its user.

    Each fold's widening comes from the out-of-fold predictions of the other folds
    (cross-conformal), so coverage is judged honestly.

    Args:
        pairs: Paired data.

    Returns:
        Bands of shape (rows, 3) in natural units.
    """
    residuals = np.log(pairs.next_lab) - np.log(pairs.anchor)
    predicted, fold_ids = _out_of_fold_residuals(pairs)
    calibrated = np.empty_like(predicted)
    for fold in np.unique(fold_ids):
        rest, here = fold_ids != fold, fold_ids == fold
        widening = _calibration_widening(residuals[rest], predicted[rest])
        calibrated[here] = widen(predicted[here], widening)
    return np.exp(np.log(pairs.anchor)[:, None] + calibrated)


def bootstrap_lower_bound(
    difference: FloatArray, user_ids: npt.NDArray[np.str_]
) -> float:
    """2.5th percentile of the mean loss difference, resampling whole users.

    Args:
        difference: Baseline loss minus model loss, per row.
        user_ids: User of every row.

    Returns:
        Lower end of the 95% bootstrap CI. Above zero means the model is better.
    """
    users, inverse = np.unique(user_ids, return_inverse=True)
    per_user_sum = np.bincount(inverse, weights=difference)
    per_user_count = np.bincount(inverse)
    rng = np.random.default_rng(RANDOM_STATE)
    draws = rng.integers(0, len(users), size=(BOOTSTRAP_DRAWS, len(users)))
    means = per_user_sum[draws].sum(axis=1) / per_user_count[draws].sum(axis=1)
    return float(np.quantile(means, 0.025))


def gate_residual_model(pairs: PairSet) -> GateResult:
    """Decide whether the residual model has earned activation.

    Checks: at least GATE_MIN_USERS users; on held-out users (GroupKFold), at least
    GATE_MIN_IMPROVEMENT lower mean pinball loss than every baseline; a user-level
    bootstrap 95% CI of the gain over the strongest baseline that excludes zero; and
    80% band coverage within GATE_COVERAGE.

    Args:
        pairs: Real consented pairs (or synthetic data in tests and notebooks).

    Returns:
        The decision, failed checks, and metrics.
    """
    n_users = len(np.unique(pairs.user_ids))
    if n_users < GATE_MIN_USERS:
        return GateResult(
            False,
            [f"only {n_users} users; the gate needs {GATE_MIN_USERS}"],
            {"users": float(n_users)},
        )
    oof = out_of_fold_bands(pairs)
    model_loss = mean_pinball(oof, pairs.next_lab)
    covered = (pairs.next_lab >= oof[:, 0]) & (pairs.next_lab <= oof[:, 2])
    metrics: dict[str, float] = {
        "users": float(n_users),
        "model_pinball": float(model_loss.mean()),
        "coverage_80": float(covered.mean()),
    }
    reasons: list[str] = []
    baseline_losses = {
        name: mean_pinball(band, pairs.next_lab)
        for name, band in pairs.baselines.items()
    }
    for name, loss in baseline_losses.items():
        improvement = 1.0 - model_loss.mean() / loss.mean()
        metrics[f"{name}_pinball"] = float(loss.mean())
        metrics[f"improvement_vs_{name}"] = float(improvement)
        if improvement < GATE_MIN_IMPROVEMENT:
            reasons.append(
                f"{100 * improvement:.1f}% better than {name}; needs "
                f"{100 * GATE_MIN_IMPROVEMENT:.0f}%"
            )
    strongest = min(baseline_losses, key=lambda name: baseline_losses[name].mean())
    lower = bootstrap_lower_bound(
        baseline_losses[strongest] - model_loss, pairs.user_ids
    )
    metrics["ci_lower_gain"] = lower
    if lower <= 0:
        reasons.append(f"95% CI of the gain over {strongest} includes zero")
    low, high = GATE_COVERAGE
    if not low <= metrics["coverage_80"] <= high:
        reasons.append(
            f"80% coverage {metrics['coverage_80']:.3f} outside {low:.2f}-{high:.2f}"
        )
    return GateResult(not reasons, reasons, metrics)


def train_residual_model(
    pairs: PairSet,
    analyte: str,
    mode: ResidualMode,
    version: str,
) -> ResidualModel:
    """Fit the production residual model on every pair.

    Args:
        pairs: Paired data.
        analyte: Canonical analyte name.
        mode: "level" or "change".
        version: Version id.

    Returns:
        The fitted, calibrated model. Serving it still requires a passed gate.
    """
    residuals = np.log(pairs.next_lab) - np.log(pairs.anchor)
    predicted, _ = _out_of_fold_residuals(pairs)
    return ResidualModel(
        analyte,
        mode,
        fit_quantile_residuals(pairs.features, residuals),
        version,
        widening=_calibration_widening(residuals, predicted),
    )
