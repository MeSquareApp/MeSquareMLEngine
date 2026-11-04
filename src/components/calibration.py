"""Subgroup-aware calibration of the p10-p90 interval (goal 4: fair across groups).

Each row belongs to a calibration group: sex x age band. Conformalized quantile
regression is run per group (Mondrian conformal), so every group gets its own
widening and the interval covers about 80% within each group, not only on average.
Groups with too few calibration rows fall back to the pooled widening.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.components.model_config import (
    DEFAULT_CATEGORY_LEVELS,
    FEATURES,
    MIN_CALIBRATION_GROUP_ROWS,
)
from src.components.model_evaluation import (
    FloatArray,
    conformal_widening,
    interval_columns,
)

POOLED = "all"
AGE_BANDS = ((20.0, 39.999), (40.0, 59.999), (60.0, 120.0))
AGE_BAND_LABELS = ("20-39", "40-59", "60+")
SEX_LABELS = {1.0: "male", 2.0: "female"}
_SEX_COLUMN = FEATURES.index("sex")
_AGE_COLUMN = FEATURES.index("age")


def age_band(age: float) -> str:
    """Age band label used for calibration and subgroup reporting.

    Args:
        age: Age in years.

    Returns:
        "20-39", "40-59", "60+", or "unknown".
    """
    for (low, high), label in zip(AGE_BANDS, AGE_BAND_LABELS, strict=True):
        if low <= age <= high:
            return label
    return "unknown"


def calibration_groups(features: FloatArray) -> list[str]:
    """Calibration group of every row of an encoded feature matrix.

    Sex is read from its category code (the index into DEFAULT_CATEGORY_LEVELS).

    Args:
        features: Matrix from prepare_features().

    Returns:
        One "sex:age_band" label per row, for example "female:40-59".
    """
    sex_levels = DEFAULT_CATEGORY_LEVELS["sex"]
    labels = []
    for code, age in zip(
        features[:, _SEX_COLUMN], features[:, _AGE_COLUMN], strict=True
    ):
        sex = (
            SEX_LABELS.get(float(sex_levels[int(code)]), "unknown")
            if not np.isnan(code) and int(code) < len(sex_levels)
            else "unknown"
        )
        band = "unknown" if np.isnan(age) else age_band(float(age))
        labels.append(f"{sex}:{band}")
    return labels


def mondrian_widening(
    values: FloatArray,
    predictions: FloatArray,
    weights: FloatArray,
    groups: Sequence[str],
    quantiles: tuple[float, ...],
) -> dict[str, float]:
    """Conformal widening of the p10-p90 interval per calibration group.

    Args:
        values: Observed values of the calibration rows.
        predictions: Out-of-sample quantile predictions, shape (rows, quantiles).
        weights: Survey weights.
        groups: Calibration group of every row.
        quantiles: Quantile levels of the prediction columns.

    Returns:
        Widening per group with at least MIN_CALIBRATION_GROUP_ROWS rows, plus the
        pooled widening under POOLED.
    """
    lower, upper = interval_columns(quantiles)
    group_array = np.asarray(groups)
    table = {
        POOLED: conformal_widening(
            values, predictions[:, lower], predictions[:, upper], weights
        )
    }
    for group in sorted(set(groups)):
        rows = group_array == group
        if rows.sum() < MIN_CALIBRATION_GROUP_ROWS:
            continue
        table[group] = conformal_widening(
            values[rows],
            predictions[rows, lower],
            predictions[rows, upper],
            weights[rows],
        )
    return table


def widening_per_row(groups: Sequence[str], table: dict[str, float]) -> FloatArray:
    """Look up each row's widening, falling back to the pooled value.

    Args:
        groups: Calibration group of every row.
        table: Output of mondrian_widening().

    Returns:
        Widening per row.
    """
    pooled = table.get(POOLED, 0.0)
    return np.array([table.get(group, pooled) for group in groups])


def cross_conformal_covered(
    values: FloatArray,
    predictions: FloatArray,
    weights: FloatArray,
    groups: Sequence[str],
    fold_ids: npt.NDArray[np.int_],
) -> FloatArray:
    """Whether each row is covered when calibrated only on the other folds.

    Every row's widening comes from mondrian_widening() on the out-of-fold rows of
    the other folds, so no row calibrates itself. This judges the calibration on
    every training row, a much larger sample than the holdout.

    Args:
        values: Observed values.
        predictions: Out-of-fold quantile predictions, shape (rows, 3) at p10/p50/p90.
        weights: Survey weights.
        groups: Calibration group of every row.
        fold_ids: CV fold of every row.

    Returns:
        1.0 where the row's value is inside its widened interval, else 0.0.
    """
    quantiles = (0.10, 0.50, 0.90)
    lower, upper = interval_columns(quantiles)
    group_array = np.asarray(groups)
    covered = np.empty(len(values))
    for fold in np.unique(fold_ids):
        rest, here = fold_ids != fold, fold_ids == fold
        table = mondrian_widening(
            values[rest],
            predictions[rest],
            weights[rest],
            list(group_array[rest]),
            quantiles,
        )
        widening = widening_per_row(list(group_array[here]), table)
        covered[here] = (values[here] >= predictions[here, lower] - widening) & (
            values[here] <= predictions[here, upper] + widening
        )
    return covered


def subgroup_coverage(
    covered: FloatArray, weights: FloatArray, groups: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Weighted interval coverage by sex and by age band, with the largest gap.

    Args:
        covered: 1.0 where the observed value was inside the interval, else 0.0.
        weights: Survey weights.
        groups: Calibration group of every row ("sex:age_band").

    Returns:
        {"sex": {...}, "age_band": {...}}, each mapping group to coverage plus
        "gap", the largest minus the smallest coverage in points.
    """
    report: dict[str, dict[str, float]] = {}
    for dimension, position in (("sex", 0), ("age_band", 1)):
        labels = np.array([group.split(":")[position] for group in groups])
        shares = {
            label: float(
                np.average(covered[labels == label], weights=weights[labels == label])
            )
            for label in sorted(set(labels))
            if label != "unknown"
        }
        shares["gap"] = 100.0 * (max(shares.values()) - min(shares.values()))
        report[dimension] = shares
    return report
