"""Contains the utility functions."""

from __future__ import annotations

import calendar
from typing import Tuple

import numpy as np


def are_equals(values: Tuple) -> bool:
    for i in range(len(values[1:])):
        if values[i - 1] != values[i]:
            return False
    return True


def get_month_number(month: int | str) -> int:
    if isinstance(month, int):
        if 1 <= month <= 12:
            return month
        else:
            raise Exception("The month number must be between 1 and 12.")
    elif isinstance(month, str):
        month_lower = month.lower()
        names = list(map(lambda value: value.lower(), calendar.month_name[1:]))
        abbr = list(map(lambda value: value.lower(), calendar.month_abbr[1:]))
        if month_lower in names:
            return names.index(month_lower) + 1
        elif month_lower in abbr:
            return abbr.index(month_lower) + 1
        else:
            raise Exception("The string '{}' does not represents a month.".format(month))
    else:
        assert False


def from_str_array_to_float_matrix(str_array: np.ndarray[str]) -> np.ndarray[np.float64]:
    str_matrix = list(map(lambda row: row[:-1].split(","), str_array))
    float_matrix = np.array(str_matrix).astype(float)
    return float_matrix
