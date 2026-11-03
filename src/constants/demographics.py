"""NHANES demographic codes used by the quantile models."""

from __future__ import annotations

from enum import Enum
from typing import Self


class CodedLabel(Enum):
    """Enum of (label, code) pairs. The value is the label, as the API accepts it."""

    label: str
    code: int

    def __new__(cls, label: str, code: int) -> Self:
        """Store the label as the value and keep the NHANES code alongside."""
        member = object.__new__(cls)
        member._value_ = label
        member.label = label
        member.code = code
        return member

    @classmethod
    def from_code(cls, code: int) -> Self:
        """Member for an NHANES code.

        Args:
            code: NHANES numeric code.

        Returns:
            The matching member.

        Raises:
            ValueError: If no member has that code.
        """
        for member in cls:
            if member.code == code:
                return member
        raise ValueError(f"Unknown {cls.__name__} code: {code!r}")


class Sex(CodedLabel):
    """NHANES RIAGENDR sex codes."""

    MALE = ("Male", 1)
    FEMALE = ("Female", 2)


class Ethnicity(CodedLabel):
    """NHANES RIDRETH3 race/ethnicity codes."""

    MEXICAN_AMERICAN = ("Mexican American", 1)
    OTHER_HISPANIC = ("Other Hispanic", 2)
    NON_HISPANIC_WHITE = ("Non-Hispanic White", 3)
    NON_HISPANIC_BLACK = ("Non-Hispanic Black", 4)
    NON_HISPANIC_ASIAN = ("Non-Hispanic Asian", 6)
    OTHER_RACE_INCLUDING_MULTI_RACIAL = ("Other Race - Including Multi-Racial", 7)
