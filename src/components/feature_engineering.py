"""Feature preparation shared by training and inference."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.components.model_config import (
    CATEGORICAL,
    DEFAULT_CATEGORY_LEVELS,
    DIABETES_DX_MAP,
    FEATURES,
)


def infer_category_levels(frame: pd.DataFrame) -> dict[str, list]:
    """Build category levels from a training frame, falling back to defaults."""
    levels: dict[str, list] = {}
    for col in CATEGORICAL:
        if col not in frame.columns:
            levels[col] = list(DEFAULT_CATEGORY_LEVELS[col])
            continue
        series = frame[col]
        if col == "smoking":
            observed = [v for v in series.dropna().astype(str).unique().tolist()]
            defaults = list(DEFAULT_CATEGORY_LEVELS[col])
            levels[col] = defaults + [v for v in observed if v not in defaults]
        else:
            observed = sorted(
                float(v) for v in pd.to_numeric(series, errors="coerce").dropna().unique()
            )
            defaults = [float(v) for v in DEFAULT_CATEGORY_LEVELS[col]]
            levels[col] = defaults + [v for v in observed if v not in defaults]
    return levels


def prepare_features(
    frame: pd.DataFrame,
    category_levels: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Map diabetes_dx, coerce numerics, and cast categoricals with fixed levels.

    Fixed category levels must match those saved at training time so LightGBM
    sees the same integer codes at inference.
    """
    work = frame.copy()
    levels = category_levels or DEFAULT_CATEGORY_LEVELS

    if "diabetes_dx" in work.columns:
        def _map_diabetes_dx(value: object) -> object:
            if pd.isna(value):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            key = str(value).strip().lower()
            return DIABETES_DX_MAP.get(key, value)

        work["diabetes_dx"] = work["diabetes_dx"].map(_map_diabetes_dx)

    for col in FEATURES:
        if col not in work.columns:
            work[col] = pd.NA
        if col not in CATEGORICAL:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in CATEGORICAL:
        col_levels = levels.get(col, DEFAULT_CATEGORY_LEVELS[col])
        if col == "smoking":
            work[col] = work[col].astype(object)
            work[col] = work[col].where(work[col].isna(), work[col].astype(str))
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = pd.Categorical(work[col], categories=col_levels)

    return work[list(FEATURES)].copy()


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a feature DataFrame from a list of patient dicts."""
    if not records:
        return pd.DataFrame(columns=list(FEATURES))
    return pd.DataFrame(records)
