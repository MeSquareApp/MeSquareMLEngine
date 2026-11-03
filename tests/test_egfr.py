import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.components.egfr import (
    add_egfr,
    egfr_ckd_epi_2021,
    egfr_ckd_epi_2021_array,
    with_egfr,
)


@pytest.mark.parametrize(
    ("sex", "expected"),
    # Published CKD-EPI 2021 values for age 50, creatinine 1.0 mg/dL.
    [(1, 91.7), (2, 68.6)],
)
def test_egfr_2021_reference_values(sex, expected):
    # Arrange
    scr, age = 1.0, 50

    # Act
    egfr = egfr_ckd_epi_2021(scr, age, sex)

    # Assert
    assert round(egfr, 1) == expected


def test_egfr_2021_rejects_unknown_sex():
    # Arrange
    sex = 3

    # Act / Assert
    with pytest.raises(ValueError):
        egfr_ckd_epi_2021(1.0, 50, sex)


def test_array_returns_nan_for_unknown_sex_or_missing_input():
    # Arrange
    scr = np.array([1.0, 1.0, np.nan])
    sex = np.array([1.0, 9.0, 2.0])

    # Act
    egfr = egfr_ckd_epi_2021_array(scr, 50, sex)

    # Assert
    assert not np.isnan(egfr[0])
    assert np.isnan(egfr[1:]).all()


def test_with_egfr_adds_nullable_column():
    # Arrange
    frame = pl.DataFrame({"scr": [1.0, None], "age": [50.0, 60.0], "sex": [1.0, 2.0]})

    # Act
    result = with_egfr(frame)

    # Assert
    assert round(result["egfr"][0], 1) == 91.7
    assert result["egfr"][1] is None


def test_add_egfr_matches_polars_version():
    # Arrange
    frame = pd.DataFrame({"scr": [0.8, 1.4], "age": [40, 70], "sex": [2, 1]})

    # Act
    result = add_egfr(frame)

    # Assert
    expected = with_egfr(pl.DataFrame(frame.to_dict("list")))["egfr"].to_numpy()
    np.testing.assert_allclose(result["egfr"].to_numpy(), expected)
