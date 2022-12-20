"""Defines the `ResultStats` class."""

from __future__ import annotations

import datetime
from numbers import Real
from typing import List

import mip


class ResultStats:
    """Represents the computation statistics of a result."""

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
            self,
            status: mip.OptimizationStatus = mip.OptimizationStatus.OPTIMAL,
            time: datetime.timedelta = datetime.timedelta(),
            obj_value: int | float | Real = 0,
            obj_bound: int | float | Real = 0
    ):
        self._status = status
        self._time = time
        self._obj_value = obj_value
        self._obj_bound = obj_bound

    # ----------------------------------------------------------------------
    # Properties.

    @property
    def status(self) -> mip.OptimizationStatus:
        """The final state of computation."""
        return self._status

    @property
    def time(self) -> datetime.timedelta:
        """Time used to obtain the solution."""
        return self._time

    @property
    def obj_value(self) -> int | float:
        """Value of the objective function."""
        return self._obj_value

    @property
    def obj_bound(self) -> int | float:
        """Value of dual bound used."""
        return self._obj_bound

    # ----------------------------------------------------------------------
    # Methods

    def to_list(self, text=False) -> List:
        """Returns a list representing the object."""
        return [str(self.time) if text else self.time,
                self.status.name if text else self.status,
                self.obj_value, self.obj_bound]

    def add(self, other: int | float | ResultStats) -> ResultStats:
        """Returns a result representing the sum of the current and `other`
        statistics.

        In particular, if `other` is an integer, returns a statistics equal
        to the current ones in which the past value is added to the value of
        the objective function."""
        if isinstance(other, ResultStats):
            # TODO: Set correct status.
            assert self.status == other.status == mip.OptimizationStatus.OPTIMAL
            return ResultStats(status=mip.OptimizationStatus.OPTIMAL,
                               time=self.time + other.time,
                               obj_value=self.obj_value + other.obj_value,
                               obj_bound=self.obj_bound + other.obj_bound)

        elif isinstance(other, int) or isinstance(other, float):
            return ResultStats(status=self.status,
                               time=self.time,
                               obj_value=self.obj_value + other,
                               obj_bound=self.obj_bound)

        else:
            raise Exception("Is not possible append {} to a ResultStats."
                            .format(type(other)))

    # ----------------------------------------------------------------------

    def __add__(self, other: int | float | ResultStats) -> ResultStats:
        """Adds `other` to these stats."""
        return self.add(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ResultStats):
            return self.status == other.status and \
                self.time == other.time and \
                self.obj_value == other.obj_value and \
                self.obj_bound == other.obj_bound
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return "status: {}, time: {}, objective value: {:.2f}, " \
               "objective bound: {:.2f}".format(self.status.name, self.time,
                                                self.obj_value, self.obj_bound)

    def __repr__(self) -> str:
        return self.__str__()
