"""Population prior: LightGBM quantile models trained on ml.cardiovascular_training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

TARGETS = ("non_hdl", "hdl", "hba1c", "egfr", "sbp")
FEATURES = (
    "sex",
    "age",
    "race_eth",
    "weight_kg",
    "waist_cm",
    "smoking",
    "told_high_bp",
    "bp_meds",
    "told_high_chol",
    "chol_meds",
    "diabetes_dx",
    "prediabetes_dx",
    "insulin",
    "diabetes_pills",
)
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
# Fixed category levels so a single-row profile encodes the same way as training data.
CATEGORY_LEVELS = {
    "sex": [1.0, 2.0],
    "race_eth": [1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
    "smoking": ["never", "former", "current"],
}
DIABETES_DX_CODES = {"no": 0.0, "borderline": 0.5, "yes": 1.0}
LGBM_PARAMS = {"num_leaves": 15, "min_child_samples": 50}
RANDOM_STATE = 42
TEST_SIZE = 0.2

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "prior"
ARTIFACT_FILE = "prior_model.joblib"


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame. Missing feature columns become NaN."""
    features = pd.DataFrame(index=frame.index)
    for column in FEATURES:
        features[column] = frame[column] if column in frame.columns else np.nan
    if features["diabetes_dx"].dtype == object or str(features["diabetes_dx"].dtype) in (
        "string",
        "str",
    ):
        features["diabetes_dx"] = features["diabetes_dx"].map(DIABETES_DX_CODES)
    for column in FEATURES:
        if column in CATEGORY_LEVELS:
            continue
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    for column, levels in CATEGORY_LEVELS.items():
        values = features[column]
        if column != "smoking":
            values = pd.to_numeric(values, errors="coerce").astype(float)
        features[column] = pd.Categorical(values, categories=levels)
    return features


def make_lgbm(tau: float, **params) -> lgb.LGBMRegressor:
    defaults = dict(
        objective="quantile",
        alpha=tau,
        n_estimators=2000,
        learning_rate=0.03,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    defaults.update(LGBM_PARAMS)
    defaults.update(params)
    return lgb.LGBMRegressor(**defaults)


def fix_crossing(pred_matrix: np.ndarray) -> np.ndarray:
    """Monotone rearrangement across quantile columns."""
    return np.sort(pred_matrix, axis=1)


@dataclass
class PriorModel:
    """One LightGBM model per (target, quantile)."""

    models: dict[str, dict[float, lgb.LGBMRegressor]]
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict_matrix(self, profile: pd.DataFrame, target: str) -> np.ndarray:
        features = prepare_features(profile)
        columns = [self.models[target][tau].predict(features) for tau in QUANTILES]
        return fix_crossing(np.column_stack(columns))

    def predict_quantiles(self, profile: pd.DataFrame) -> dict[str, dict[float, float]]:
        """Quantiles for the first row of profile, keyed by target then tau."""
        result: dict[str, dict[float, float]] = {}
        for target in self.models:
            row = self.predict_matrix(profile.iloc[[0]], target)[0]
            result[target] = {tau: float(value) for tau, value in zip(QUANTILES, row, strict=True)}
        return result

    def save(self, directory: Path = ARTIFACT_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT_FILE
        joblib.dump({"models": self.models, "metrics": self.metrics}, path)
        return path

    @classmethod
    def load(cls, directory: Path = ARTIFACT_DIR) -> PriorModel:
        payload = joblib.load(directory / ARTIFACT_FILE)
        return cls(models=payload["models"], metrics=payload["metrics"])


def _fit_quantile(
    tau: float, X_train: pd.DataFrame, y_train: np.ndarray, **params
) -> lgb.LGBMRegressor:
    """Pick the tree count by early stopping on a validation split, then refit on all of X_train."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    probe = make_lgbm(tau, **params)
    probe.fit(
        X_fit,
        y_fit,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    n_estimators = max(50, int(probe.best_iteration_ or probe.n_estimators))
    model = make_lgbm(tau, n_estimators=n_estimators, **params)
    model.fit(X_train, y_train)
    return model


def train_prior_models(
    frame: pd.DataFrame, targets: tuple[str, ...] = TARGETS, **params
) -> PriorModel:
    """Train every (target, tau) model and score pinball loss and coverage on a held-out split."""
    models: dict[str, dict[float, lgb.LGBMRegressor]] = {}
    metric_rows = []
    for target in targets:
        subset = frame.dropna(subset=[target])
        X = prepare_features(subset)
        y = subset[target].astype(float).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        models[target] = {tau: _fit_quantile(tau, X_train, y_train, **params) for tau in QUANTILES}
        predictions = fix_crossing(
            np.column_stack([models[target][tau].predict(X_test) for tau in QUANTILES])
        )
        for j, tau in enumerate(QUANTILES):
            baseline = np.full_like(y_test, np.quantile(y_train, tau))
            metric_rows.append(
                {
                    "target": target,
                    "tau": tau,
                    "n_train": len(y_train),
                    "test_pinball": mean_pinball_loss(y_test, predictions[:, j], alpha=tau),
                    "baseline_pinball": mean_pinball_loss(y_test, baseline, alpha=tau),
                    "coverage": float(np.mean(y_test <= predictions[:, j])),
                }
            )
    return PriorModel(models=models, metrics=pd.DataFrame(metric_rows))
