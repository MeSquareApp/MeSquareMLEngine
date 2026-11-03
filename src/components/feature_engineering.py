"""Feature preparation shared by training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

from src.components.model_config import (
    CATEGORICAL,
    DEFAULT_CATEGORY_LEVELS,
    DIABETES_DX_MAP,
    FEATURES,
    CategoryLevels,
)

FloatArray = npt.NDArray[np.float64]
FeatureValue = str | float | int | None

STRING_FEATURES = frozenset({"smoking", "diabetes_dx"})
FEATURE_SCHEMA: dict[str, type[pl.DataType]] = {
    name: pl.String if name in STRING_FEATURES else pl.Float64 for name in FEATURES
}


def infer_category_levels(frame: pl.DataFrame) -> CategoryLevels:
    """Build category levels from a training frame, after the default levels.

    Levels seen in the data but not in the defaults are appended in sorted order, so
    the integer code of every default level never changes.

    Args:
        frame: Training frame with the CATEGORICAL columns.

    Returns:
        Ordered levels per categorical feature.
    """
    levels: CategoryLevels = {}
    for name in CATEGORICAL:
        defaults = list(DEFAULT_CATEGORY_LEVELS[name])
        if name not in frame.columns:
            levels[name] = defaults
            continue
        dtype = pl.String if name in STRING_FEATURES else pl.Float64
        observed = frame[name].cast(dtype, strict=False).drop_nulls().unique().sort()
        levels[name] = defaults + [
            level for level in observed.to_list() if level not in defaults
        ]
    return levels


def records_to_frame(records: Sequence[Mapping[str, FeatureValue]]) -> pl.DataFrame:
    """Build a feature frame from patient records. Unknown keys are ignored.

    Args:
        records: Patient feature dicts with NHANES codes.

    Returns:
        Frame with every column in FEATURES. Missing values are null.
    """
    return pl.DataFrame(list(records), schema=FEATURE_SCHEMA, strict=False)


def prepare_features(
    frame: pl.DataFrame, category_levels: CategoryLevels | None = None
) -> FloatArray:
    """Encode features as the float matrix the LightGBM models expect.

    Categorical features become the index of their level in category_levels.
    diabetes_dx maps through DIABETES_DX_MAP. Unknown or missing values become NaN.
    The levels must match the ones saved at training time.

    Example:
        >>> frame = records_to_frame([{"sex": 2, "smoking": "former", "age": 50}])
        >>> prepare_features(frame)[0, :3].tolist()
        [1.0, 50.0, nan]

    Args:
        frame: Frame with feature columns. Missing columns are treated as null.
        category_levels: Ordered levels per categorical feature. Defaults to
            DEFAULT_CATEGORY_LEVELS.

    Returns:
        Matrix of shape (rows, len(FEATURES)) in FEATURES order.
    """
    levels = category_levels or DEFAULT_CATEGORY_LEVELS
    missing = [name for name in FEATURES if name not in frame.columns]
    frame = frame.with_columns(pl.lit(None).alias(name) for name in missing)
    return (
        frame.select(_encode(name, levels) for name in FEATURES)
        .to_numpy()
        .astype(np.float64)
    )


def _encode(name: str, levels: CategoryLevels) -> pl.Expr:
    """Expression that turns one feature column into floats."""
    column = pl.col(name)
    if name in CATEGORICAL:
        dtype = pl.String if name in STRING_FEATURES else pl.Float64
        names = levels[name]
        column = column.cast(dtype, strict=False).replace_strict(
            names, list(range(len(names))), default=None, return_dtype=pl.Float64
        )
    elif name == "diabetes_dx":
        column = (
            column.cast(pl.String, strict=False)
            .str.strip_chars()
            .str.to_lowercase()
            .replace_strict(
                list(DIABETES_DX_MAP),
                list(DIABETES_DX_MAP.values()),
                default=None,
                return_dtype=pl.Float64,
            )
        )
    else:
        column = column.cast(pl.Float64, strict=False)
    return column.alias(name)
