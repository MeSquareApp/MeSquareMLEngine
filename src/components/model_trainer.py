"""LightGBM quantile model training and cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.components.model_config import (
    DEFAULT_N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    LEARNING_RATE,
    MIN_N_ESTIMATORS,
    N_SPLITS,
    PARAM_GRID,
    QUANTILES,
    RANDOM_STATE,
)
from src.components.model_evaluation import pinball_loss
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CVResult:
    config_name: str
    cfg: dict[str, int]
    n_estimators_by_tau: dict[float, int]
    pinball_mean_by_tau: dict[float, float]
    mean_pinball: float


def config_name(cfg: dict[str, int]) -> str:
    return f"leaves{cfg['num_leaves']}_mcs{cfg['min_child_samples']}"


def make_lgbm(tau: float, **params) -> lgb.LGBMRegressor:
    defaults = dict(
        objective="quantile",
        alpha=tau,
        n_estimators=DEFAULT_N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    defaults.update(params)
    return lgb.LGBMRegressor(**defaults)


def fit_with_es(
    model: lgb.LGBMRegressor,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> lgb.LGBMRegressor:
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def prepare_xy(
    frame: pd.DataFrame,
    target: str,
    feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = frame[target].notna()
    X = feature_frame.loc[mask].reset_index(drop=True)
    y = frame.loc[mask, target].astype(float).to_numpy()
    return X, y


def cross_validate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
    param_grid: list[dict[str, int]] | None = None,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> CVResult:
    """Select the best PARAM_GRID config and per-tau n_estimators via KFold."""
    grid = param_grid or PARAM_GRID
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict] = []

    for cfg_i, cfg in enumerate(grid):
        name = config_name(cfg)
        for tau in quantiles:
            for fold, (tr_ix, va_ix) in enumerate(kf.split(X_train)):
                X_tr, X_va = X_train.iloc[tr_ix], X_train.iloc[va_ix]
                y_tr, y_va = y_train[tr_ix], y_train[va_ix]
                model = make_lgbm(tau, **cfg)
                fit_with_es(model, X_tr, y_tr, X_va, y_va)
                pred = model.predict(X_va)
                best_iter = int(
                    getattr(model, "best_iteration_", model.n_estimators) or model.n_estimators
                )
                rows.append(
                    {
                        "config": name,
                        "cfg_i": cfg_i,
                        "tau": tau,
                        "fold": fold,
                        "pinball": pinball_loss(y_va, pred, tau),
                        "best_iter": best_iter,
                    }
                )
        mean_pin = float(np.mean([r["pinball"] for r in rows if r["config"] == name]))
        logger.info("  %s: mean pinball=%.4f", name, mean_pin)

    cv_df = pd.DataFrame(rows)
    by_cfg = (
        cv_df.groupby("config", as_index=False)["pinball"]
        .mean()
        .sort_values("pinball")
    )
    best_name = str(by_cfg.iloc[0]["config"])
    best_cfg_i = int(cv_df.loc[cv_df["config"] == best_name, "cfg_i"].iloc[0])
    best_cfg = grid[best_cfg_i]

    n_estimators_by_tau: dict[float, int] = {}
    pinball_mean_by_tau: dict[float, float] = {}
    for tau in quantiles:
        sub = cv_df[(cv_df["config"] == best_name) & (cv_df["tau"] == tau)]
        n_estimators_by_tau[tau] = max(
            MIN_N_ESTIMATORS, int(round(float(sub["best_iter"].mean())))
        )
        pinball_mean_by_tau[tau] = float(sub["pinball"].mean())

    return CVResult(
        config_name=best_name,
        cfg=best_cfg,
        n_estimators_by_tau=n_estimators_by_tau,
        pinball_mean_by_tau=pinball_mean_by_tau,
        mean_pinball=float(by_cfg.iloc[0]["pinball"]),
    )


def fit_final(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, int],
    n_estimators_by_tau: dict[float, int],
    quantiles: tuple[float, ...] = QUANTILES,
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one quantile model per tau on the full labeled set."""
    models: dict[float, lgb.LGBMRegressor] = {}
    for tau in quantiles:
        model = make_lgbm(tau, n_estimators=n_estimators_by_tau[tau], **cfg)
        model.fit(X, y)
        models[tau] = model
    return models


def predict_quantiles(
    models: dict[float, lgb.LGBMRegressor],
    X: pd.DataFrame,
    quantiles: tuple[float, ...] = QUANTILES,
) -> np.ndarray:
    """Predict a (n_samples, n_quantiles) matrix in quantile order."""
    from src.components.model_evaluation import fix_crossing

    pred_mat = np.zeros((len(X), len(quantiles)))
    for j, tau in enumerate(quantiles):
        pred_mat[:, j] = models[tau].predict(X)
    return fix_crossing(pred_mat)
