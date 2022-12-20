"""Contains functions to load and write computation statistics."""

from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd

from pycode.dataset import data
from pycode.dataset.data import ModelType
from pycode.src.ResultStats import ResultStats
from pycode.utility.checks import check_inside
from pycode.utility.utils import get_month_number


# ----------------------------------------------------------------------
# Static energy profiles instances stats.

def load_static_energy(filepath: str | Path) -> pd.DataFrame:
    return pd.read_csv(filepath, index_col=(0, 1, 2, 3))


# noinspection DuplicatedCode
def write_static_energy(
        filepath: str | Path,
        dataset: str,
        time_slots: int,
        instance_number: int,
        energy_profile: int,
        new_stats: ResultStats,
        decimals: int = 3,
        check_already_exists: bool = True
) -> None:
    # Check input params.
    check_inside(dataset, data.datasets, "dataset")
    check_inside(time_slots, data.slot_times, "slot_times")
    check_inside(instance_number, data.instance_numbers, "instance_number")
    check_inside(energy_profile, data.energy_profiles, "energy_profile")

    stats = load_static_energy(filepath)
    index = (dataset, time_slots, instance_number, energy_profile)
    if check_already_exists and index in stats.index:
        raise Exception("There is already a record with index {}.".format(index))
    stats.loc[index] = new_stats.to_list(text=True)
    stats.sort_index(inplace=True)
    stats.to_csv(filepath, float_format="%.{}f".format(decimals))


# ----------------------------------------------------------------------
# Distance based energy instances stats

def load_distances_based_energy(filepath: str | Path) -> pd.DataFrame:
    return pd.read_csv(filepath, index_col=(0, 1, 2, 3, 4, 5, 6))


# noinspection DuplicatedCode
def write_distances_based_energy(
        filepath: str | Path,
        dataset: str,
        time_slots: int,
        instance_number: int,
        reference_profile: int,
        energy_production_function: str,
        month: int | str,
        model_type: ModelType,
        new_stats: ResultStats,
        decimals: int = 3,
        check_already_exists: bool = True,
) -> None:
    # Check input params.
    check_inside(dataset, data.datasets, "dataset")
    check_inside(time_slots, data.slot_times, "slot_times")
    check_inside(instance_number, data.instance_numbers, "instance_number")
    check_inside(reference_profile, data.energy_profiles, "reference_profile")
    check_inside(energy_production_function, data.energy_production_functions,
                 "energy_production_function")

    month = get_month_number(month)
    month_abbr = calendar.month_abbr[month]
    stats = load_distances_based_energy(filepath)
    index = (model_type.name, dataset, time_slots, instance_number, month_abbr,
             reference_profile, energy_production_function)
    if check_already_exists and index in stats.index:
        raise Exception("There is already a record with index {}.".format(index))
    stats.loc[index] = new_stats.to_list(text=True)
    stats.sort_index(inplace=True)
    stats.to_csv(filepath, float_format="%.{}f".format(decimals))
