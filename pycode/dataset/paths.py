"""Contains functions to obtain the path to the files containing the data."""

from __future__ import annotations

import calendar
from pathlib import Path

from pycode.dataset import data
from pycode.dataset.data import ModelType
from pycode.utility.checks import check_inside
from pycode.utility.utils import get_month_number

base_path = Path(__file__).parents[2] / "data"

# ----------------------------------------------------------------------
# Facility positions

facilities_coords_folder = "facilities_positions/facilities_coord_umt.csv"


def resolve_facilities_positions() -> Path:
    return base_path / facilities_coords_folder


# ----------------------------------------------------------------------
# Irradiation

irradiation_folder = "irradiation"


def resolve_facility_irradiation(facility: int) -> Path:
    check_inside(facility, range(data.n_facility), "facility")
    return base_path / irradiation_folder / "k{}.json".format(facility)


# ----------------------------------------------------------------------
# Splits

splits_folder = "splits"


def resolve_lad_splits(instance_number: int) -> Path:
    check_inside(instance_number, data.instance_numbers, "instance_number")
    return base_path / splits_folder / "out_{}_LAD.log".format(instance_number)


def resolve_lsd_splits(instance_number: int) -> Path:
    check_inside(instance_number, data.instance_numbers, "instance_number")
    return base_path / splits_folder / "out_{}_LSD.log".format(instance_number)


def resolve_equals_splits(instance_number: int) -> Path:
    check_inside(instance_number, data.instance_numbers, "instance_number")
    return base_path / splits_folder / "out_{}_EQUALS.log".format(instance_number)


# ----------------------------------------------------------------------
# Instances

instances_folder = "instances"


def resolve_instance(dataset: str, time_slots: int, instance_number: int) -> Path:
    check_inside(dataset, data.datasets, "dataset")
    check_inside(time_slots, data.slot_times, "time_slots")
    check_inside(instance_number, data.instance_numbers, "instance_number")

    return base_path / instances_folder / "dataset_{}_{}t_{}.dat" \
        .format(dataset, time_slots, instance_number)


# ----------------------------------------------------------------------
# Results

results_folder = "results"
results_static_energy_folder = "static_energy_profiles"
results_distances_energy_folder = "distances_based_energy_profiles"
results_normal_folder = "normal_model"
results_heuristic_migration_0_folder = "migration_0"
results_heuristic_migration_inf_folder = "migration_inf"
results_heuristic_splits_folder_format = "split_{}_{}"
results_heuristic_medians_folder_format = "medians_{}_{}"


def resolve_static_energy_result(
        dataset: str,
        time_slots: int,
        instance_number: int,
        energy_profile: int
) -> Path:
    check_inside(dataset, data.datasets, "dataset")
    check_inside(time_slots, data.slot_times, "time_slots")
    check_inside(instance_number, data.instance_numbers, "instance_number")
    check_inside(energy_profile, data.energy_profiles, "energy_profile")

    folder_path = base_path / results_folder / results_static_energy_folder
    return folder_path / "result_{}_{}t_{}_{}.csv" \
        .format(dataset, time_slots, instance_number, energy_profile)


def resolve_static_energy_results_stats() -> Path:
    return base_path / results_folder / results_static_energy_folder / "stats.csv"


def resolve_distances_energy_result(
        dataset: str,
        time_slots: int,
        instance_number: int,
        energy_production_function: str,
        reference_profile: int,
        month: int | str,
        model_type: ModelType
) -> Path:
    check_inside(dataset, data.datasets, "dataset")
    check_inside(time_slots, data.slot_times, "time_slots")
    check_inside(instance_number, data.instance_numbers, "instance_number")
    check_inside(reference_profile, data.energy_profiles, "energy_instance_reference")
    check_inside(energy_production_function, data.energy_production_functions,
                 "energy_production_function")

    month = get_month_number(month)
    filepath = base_path / results_folder / results_distances_energy_folder
    if model_type == ModelType.NORMAL:
        filepath /= results_normal_folder
    elif model_type == ModelType.HEURISTIC_MIGRATION_0:
        filepath /= results_heuristic_migration_0_folder
    elif model_type == ModelType.HEURISTIC_MIGRATION_INF:
        filepath /= results_heuristic_migration_inf_folder
    elif model_type.is_splitted:
        filepath /= results_heuristic_splits_folder_format.format(model_type.n_splits,
                                                                  model_type.split_type)
    elif model_type.is_median:
        filepath /= results_heuristic_medians_folder_format.format(model_type.n_splits,
                                                                   model_type.split_type)
    else:
        raise AssertionError()
    filepath /= calendar.month_abbr[month].lower()
    filename = "result_{}_{}t_{}_{}_{}.csv".format(dataset, time_slots, instance_number,
                                                   reference_profile,
                                                   energy_production_function)
    return filepath / filename


def resolve_distances_based_energy_results_stats() -> Path:
    return base_path / results_folder / results_distances_energy_folder / "stats.csv"
