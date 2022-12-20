"""Contains the functions that provide access to the data."""

from __future__ import annotations

import calendar
from typing import Dict

import numpy as np
import pandas as pd

from pycode.dataset import data, facilities, instances, irradiation, paths, \
    result_stats, splits
from pycode.dataset.data import ModelType
from pycode.src.ResultStats import ResultStats
from pycode.src.instance import BaseInstance, EnergyInstance
from pycode.src.result import Result
from pycode.utility.checks import check_inside
from pycode.utility.utils import get_month_number


# ----------------------------------------------------------------------
# Facilities

def get_facilities_positions() -> pd.DataFrame:
    return facilities.load_positions(paths.resolve_facilities_positions())


def get_facilities_distances_from_center() -> pd.Series:
    return facilities.load_distances_from_point(paths.resolve_facilities_positions(),
                                                data.center)


# ----------------------------------------------------------------------
# Irradiation

def get_facility_irradiation(facility: int) -> Dict[str, pd.DataFrame]:
    return irradiation.load(paths.resolve_facility_irradiation(facility))


def get_facility_irradiation_all_months_avg(facility: int) -> pd.DataFrame:
    return irradiation.load_months_avg(paths.resolve_facility_irradiation(facility))


def get_all_facilities_irradiation_month_avg(month: int | str) -> pd.DataFrame:
    month = get_month_number(month)
    irr_dict = dict()
    for k in range(data.n_facility):
        month_abbr = calendar.month_abbr[month]
        irr_dict[k] = get_facility_irradiation_all_months_avg(k)[month_abbr]
    return pd.DataFrame(irr_dict)


# ----------------------------------------------------------------------
# Splits

def get_split_points(
        splits_type: str,
        instance_number: int,
        splits_number: int
) -> np.ndarray[int]:
    # Check params.
    check_inside(instance_number, data.instance_numbers, "instance_number")

    if splits_type.upper() == "LAD":
        path = paths.resolve_lad_splits(instance_number)
    elif splits_type.upper() == "LSD":
        path = paths.resolve_lsd_splits(instance_number)
    elif splits_type.upper() == "EQUALS":
        path = paths.resolve_equals_splits(instance_number)
    else:
        raise Exception("The value of 'splits_type' must be 'EQUALS', 'LAD' or 'LSD'.")
    return splits.load_splits(path)[splits_number]


def get_split_medians(
        splits_type: str,
        instance_number: int,
        splits_number: int
) -> np.ndarray[int]:
    # Check params.
    check_inside(instance_number, data.instance_numbers, "instance_number")

    if splits_type.upper() == "LAD":
        path = paths.resolve_lad_splits(instance_number)
    elif splits_type.upper() == "LSD":
        path = paths.resolve_lsd_splits(instance_number)
    elif splits_type.upper() == "EQUALS":
        raise Exception("Does not exists medians for 'EQUALS' splits type.")
    else:
        raise Exception("The value of 'splits_type' must be 'LAD' or 'LSD'.")
    return splits.load_medians(path)[splits_number]


# ----------------------------------------------------------------------
# Instances

def get_base_instance(
        dataset: str,
        time_slots: int,
        instance_number: int
) -> BaseInstance:
    return instances.load_base(
        paths.resolve_instance(dataset, time_slots, instance_number)
    )


def get_static_energy_instance(
        dataset: str,
        time_slots: int,
        instance_number: int,
        reference_profile: int,
        **kwargs
) -> EnergyInstance:
    # Check params.
    check_inside(reference_profile, data.energy_profiles, "energy_profile")

    filepath = paths.resolve_instance(dataset, time_slots, instance_number)
    if reference_profile == 1:
        return instances.load_with_energy_profile_1(filepath, **kwargs)
    elif reference_profile == 2:
        return instances.load_with_energy_profile_2(filepath, **kwargs)
    else:
        raise Exception("The value of 'energy_profile' must be 1 or 2.")


def get_distances_based_energy_instance(
        dataset: str,
        time_slots: int,
        instance_number: int,
        reference_profile: int,
        energy_production_function: str,
        month: int | str,
        **kwargs
) -> EnergyInstance:
    # Check params.
    check_inside(energy_production_function, data.energy_production_functions.keys(),
                 "energy_production_function")

    reference_instance = get_static_energy_instance(dataset=dataset,
                                                    time_slots=time_slots,
                                                    instance_number=instance_number,
                                                    reference_profile=reference_profile,
                                                    **kwargs)
    function = data.energy_production_functions[energy_production_function]
    distances_from_center = get_facilities_distances_from_center().to_numpy()
    month_irradiation = get_all_facilities_irradiation_month_avg(month).to_numpy()

    return instances.load_with_distances_based_energy_data(
        reference_instance=reference_instance,
        energy_production_func=function,
        distances_from_center=distances_from_center,
        irradiation=month_irradiation,
        **kwargs
    )


# ----------------------------------------------------------------------
# Results stats

def get_static_energy_results_stats() -> pd.DataFrame:
    return result_stats.load_static_energy(
        paths.resolve_static_energy_results_stats()
    )


def get_distances_based_energy_results_stats() -> pd.DataFrame:
    return result_stats.load_distances_based_energy(
        paths.resolve_distances_based_energy_results_stats()
    )


def add_static_energy_result_stats(
        dataset: str,
        time_slots: int,
        instance_number: int,
        energy_profile: int,
        stats: ResultStats,
        check_already_exists: bool = True
) -> None:
    result_stats.write_static_energy(
        filepath=paths.resolve_static_energy_results_stats(),
        dataset=dataset,
        time_slots=time_slots,
        instance_number=instance_number,
        energy_profile=energy_profile,
        new_stats=stats,
        check_already_exists=check_already_exists
    )


def add_distances_based_energy_result_stats(
        dataset: str,
        time_slots: int,
        instance_number: int,
        reference_profile: int,
        energy_production_function: str,
        month: int | str,
        model_type: ModelType,
        stats: ResultStats,
        check_already_exists: bool = True
) -> None:
    result_stats.write_distances_based_energy(
        filepath=paths.resolve_distances_based_energy_results_stats(),
        dataset=dataset,
        time_slots=time_slots,
        instance_number=instance_number,
        reference_profile=reference_profile,
        energy_production_function=energy_production_function,
        month=month,
        model_type=model_type,
        new_stats=stats,
        check_already_exists=check_already_exists
    )


# ----------------------------------------------------------------------
# Results


def get_static_energy_result(
        dataset: str,
        time_slots: int,
        instance_number: int,
        energy_profile: int
) -> Result:
    return Result.from_file(paths.resolve_static_energy_result(
        dataset=dataset,
        time_slots=time_slots,
        instance_number=instance_number,
        energy_profile=energy_profile)
    )


def get_distances_based_energy_result(
        dataset: str,
        time_slots: int,
        instance_number: int,
        reference_profile: int,
        energy_production_function: str,
        month: int | str,
        model_type: ModelType
) -> Result:
    return Result.from_file(
        paths.resolve_distances_energy_result(
            dataset=dataset,
            time_slots=time_slots,
            instance_number=instance_number,
            energy_production_function=energy_production_function,
            reference_profile=reference_profile,
            month=month,
            model_type=model_type)
    )
