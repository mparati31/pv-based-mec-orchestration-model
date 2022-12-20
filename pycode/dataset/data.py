"""Contains values concerning the data."""

from __future__ import annotations

from enum import Enum

import numpy as np

# ----------------------------------------------------------------------
# Instances parameters

n_ap = 1419
n_facility = 10

# ----------------------------------------------------------------------
# Instances values

datasets = np.array(["A", "B"])
slot_times = np.array([12, 24, 48, 96])
instance_numbers = np.array([1, 2, 3, 4, 5])
energy_profiles = np.array([1, 2])
energy_production_functions = {
    "constant": lambda d, max_d: 1,
    "linear": lambda d, max_d: d / max_d,
    "quadratic": lambda d, max_d: d ** 2 / max_d ** 2,
    "exponential": lambda d, max_d: np.e ** (d / 5000) / np.e ** (max_d / 5000)
}

# ----------------------------------------------------------------------

# Coordinates of the central point used to calculate the energy produced.
center = (514962, 5034533)


# ----------------------------------------------------------------------
# ModelType class

class ModelType(Enum):
    # Normal model, no heuristic.
    NORMAL = 0

    # Heuristic migration cost 0.
    HEURISTIC_MIGRATION_0 = 1
    # Heuristic migration cost infinite.
    HEURISTIC_MIGRATION_INF = 2

    # Heuristic splitting time slots in ranges with same demand.
    HEURISTIC_SPLIT_12_EQUALS = 3
    HEURISTIC_SPLIT_24_EQUALS = 4
    HEURISTIC_SPLIT_48_EQUALS = 5

    # Heuristic splitting time slots in ranges using the distances in absolute value.
    HEURISTIC_SPLIT_12_LAD = 6
    HEURISTIC_SPLIT_24_LAD = 7
    HEURISTIC_SPLIT_48_LAD = 8

    # Heuristic splitting time slots in ranges using the least squares of distances.
    HEURISTIC_SPLIT_12_LSD = 9
    HEURISTIC_SPLIT_24_LSD = 10
    HEURISTIC_SPLIT_48_LSD = 11

    HEURISTIC_MEDIAN_12_LAD = 12
    HEURISTIC_MEDIAN_24_LAD = 13
    HEURISTIC_MEDIAN_48_LAD = 14

    HEURISTIC_MEDIAN_12_LSD = 15
    HEURISTIC_MEDIAN_24_LSD = 16
    HEURISTIC_MEDIAN_48_LSD = 17

    @property
    def is_splitted(self) -> bool:
        if self.name == "NORMAL":
            return False
        else:
            return self.name.split("_")[1] == "SPLIT"

    @property
    def is_median(self) -> bool:
        if self.name == "NORMAL":
            return False
        else:
            return self.name.split("_")[1] == "MEDIAN"

    @property
    def split_type(self) -> str | None:
        if self.is_splitted or self.is_median:
            return self.name.split("_")[-1]
        else:
            return None

    @property
    def n_splits(self) -> int | None:
        if self.is_splitted or self.is_median:
            return int(self.name.split("_")[-2])
        else:
            return None
