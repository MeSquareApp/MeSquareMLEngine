"""NHANES demographic codes used by the quantile models."""

from __future__ import annotations

from enum import Enum


class Sex(Enum):
    """NHANES RIAGENDR sex codes."""

    MALE = ("Male", 1)
    FEMALE = ("Female", 2)

    label: str
    code: int

    def __new__(cls, label: str, code: int) -> "Sex":
        obj = object.__new__(cls)
        obj._value_ = label
        obj.label = label
        obj.code = code
        return obj

    @classmethod
    def from_code(cls, code: int) -> "Sex":
        for member in cls:
            if member.code == code:
                return member
        raise ValueError(f"Unknown sex code: {code!r}")


class Ethnicity(Enum):
    """NHANES RIDRETH3 race/ethnicity codes."""

    MEXICAN_AMERICAN = ("Mexican American", 1)
    OTHER_HISPANIC = ("Other Hispanic", 2)
    NON_HISPANIC_WHITE = ("Non-Hispanic White", 3)
    NON_HISPANIC_BLACK = ("Non-Hispanic Black", 4)
    NON_HISPANIC_ASIAN = ("Non-Hispanic Asian", 6)
    OTHER_RACE_INCLUDING_MULTI_RACIAL = ("Other Race - Including Multi-Racial", 7)

    label: str
    code: int

    def __new__(cls, label: str, code: int) -> "Ethnicity":
        obj = object.__new__(cls)
        obj._value_ = label
        obj.label = label
        obj.code = code
        return obj

    @classmethod
    def from_code(cls, code: int) -> "Ethnicity":
        for member in cls:
            if member.code == code:
                return member
        raise ValueError(f"Unknown ethnicity code: {code!r}")
