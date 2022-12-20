"""Contains the functions for check values."""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


# ----------------------------------------------------------------------
# Single check


def check_inside(
        value: T,
        valid_values: Sequence,
        label: str,
        active: bool = True
) -> None:
    """Throws an exception if `value` does not belong to `valid_values`."""
    if active and value not in valid_values:
        possible_values_text = ", ".join(map(lambda item: str(item), valid_values))
        raise Exception("The value of '{}' must be one of the following: {}."
                        .format(label, possible_values_text))


def check_between(
        value: int | float,
        lb: int | float,
        ub: int | float,
        label: str
) -> None:
    """Throws an exception if `value` is not between `lb` and `ub` (included)."""
    if not (lb <= value <= ub):
        raise Exception("The value of '{}' must be between {} and {}."
                        .format(label, lb, ub))


def check_strictly_positive(
        value: int | float,
        label: str,
        active: bool = True
) -> None:
    """Throws an exception if `value` is not strictly positive."""
    if active and not value > 0:
        raise Exception("The value of '{}' must be strictly positive.".format(label))


def check_sum_equal(
        addends: Sequence[int | float],
        expected: int | float,
        labels: Sequence[str],
        active: bool = True
) -> None:
    """Throws an exception if the sum of `addends` is not 'expected'."""
    assert len(addends) == len(labels)
    if active and sum(addends) != expected:
        labels = ", ".join(["'{}'".format(label) for label in labels])
        raise Exception("The sum of {} must be {}.".format(labels, expected))


def check_shape(
        array: np.ndarray,
        shape: Tuple[int, ...],
        label: str
) -> None:
    """Throws an exception if 'array' has not 'shape' as shape."""
    if array.shape != shape:
        raise Exception("The shape of '{}' must be {}.".format(label, shape))


# ----------------------------------------------------------------------
# Multiple check


def check_all_inside(
        values: Sequence[T],
        valid_values: Sequence,
        labels: Sequence[str],
        active: bool = True
) -> None:
    """Like `check_inside` but the check is made for all elements of `values`."""
    if len(labels) == 1:
        labels = list(itertools.repeat(labels[0], len(values)))
    assert len(values) == len(labels)
    for value, label in zip(values, labels):
        check_inside(value, valid_values, label, active)


def check_all_between(
        values: Sequence[int | float],
        lb: int | float,
        ub: int | float,
        labels: Sequence[str]
) -> None:
    """Like `check_between` but the check is made for all elements of `values`."""
    assert len(values) == len(labels)
    for value, label in zip(values, labels):
        check_between(value, lb, ub, label)


def check_all_strictly_positive(
        values: Sequence[int | float],
        labels: Sequence[str],
        active: bool = True
) -> None:
    """Like `check_strictly_positive` but the check is made for all elements
    of `values`."""
    assert len(values) == len(labels)
    for value, label in zip(values, labels):
        check_strictly_positive(value, label, active)
