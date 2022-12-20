"""Contains functions for execute instances and save the results."""

from __future__ import annotations

import calendar
import itertools
from datetime import datetime
from typing import List

from pycode.dataset import api, paths
from pycode.dataset.data import ModelType
from pycode.src.models.execute import execute
from pycode.utility import utils


def exec_and_save_static_energy(
        dataset: str,
        time_slots: int,
        instance_number: int,
        energy_profile: int,
        save_result: bool = True,
        save_stats: bool = True,
        overwrite_if_result_exist: bool = False,
        overwrite_if_stats_exist: bool = False,
        show_result_stats: bool = True,
        skip_if_exist: bool = False,
        **kwargs
) -> None:
    result_path = paths.resolve_static_energy_result(dataset,
                                                     time_slots,
                                                     instance_number,
                                                     energy_profile)
    if skip_if_exist and result_path.exists():
        print("SKIPPED")
    else:
        instance = api.get_static_energy_instance(dataset,
                                                  time_slots,
                                                  instance_number,
                                                  energy_profile)
        result, stats = execute(instance, ModelType.NORMAL, **kwargs)
        if show_result_stats:
            print(stats)
        if save_result:
            result.save_to_file(result_path,
                                overwrite_if_exists=overwrite_if_result_exist)
        if save_stats:
            api.add_static_energy_result_stats(
                dataset, time_slots, instance_number, energy_profile, stats,
                check_already_exists=not overwrite_if_stats_exist)


def exec_and_save_distances_based(
        dataset: str,
        time_slots: int,
        instance_number: int,
        reference_profile: int,
        energy_production_function: str,
        month: int | str,
        model_type: ModelType,
        save_result: bool = True,
        save_stats: bool = True,
        overwrite_if_result_exist: bool = False,
        overwrite_if_stats_exist: bool = False,
        show_result_stats: bool = True,
        skip_if_exist: bool = False,
        **kwargs
) -> None:
    result_path = paths.resolve_distances_energy_result(dataset,
                                                        time_slots,
                                                        instance_number,
                                                        energy_production_function,
                                                        reference_profile,
                                                        month,
                                                        model_type)
    if skip_if_exist and result_path.exists():
        print("SKIPPED")
    else:
        instance = api.get_distances_based_energy_instance(dataset,
                                                           time_slots,
                                                           instance_number,
                                                           reference_profile,
                                                           energy_production_function,
                                                           month)
        # Splits.
        split_points = None
        medians = None
        if model_type.is_splitted:
            split_points = api.get_split_points(model_type.split_type,
                                                instance_number,
                                                model_type.n_splits)
        elif model_type.is_median:
            split_points = api.get_split_points(model_type.split_type,
                                                instance_number,
                                                model_type.n_splits)
            medians = api.get_split_medians(model_type.split_type,
                                            instance_number,
                                            model_type.n_splits)

        result, stats = execute(instance,
                                model_type,
                                split_points=split_points,
                                medians=medians,
                                **kwargs)
        if show_result_stats:
            print(stats)
        if save_result:
            result.save_to_file(result_path,
                                overwrite_if_exists=overwrite_if_result_exist)
        if save_stats:
            api.add_distances_based_energy_result_stats(
                dataset,
                time_slots,
                instance_number,
                reference_profile,
                energy_production_function,
                month,
                model_type,
                stats,
                check_already_exists=not overwrite_if_stats_exist
            )


def exec_and_save_all_static_energy(
        datasets: List[str],
        all_time_slots: List[int],
        instance_numbers: List[int],
        energy_profiles: List[int],
        save_result: bool = True,
        save_stats: bool = True,
        overwrite_if_result_exist: bool = False,
        overwrite_if_stats_exist: bool = False,
        show_running_instance: bool = True,
        show_result_stats: bool = True,
        skip_if_exist: bool = False,
        **kwargs
) -> None:
    count = 0
    product = list(itertools.product(datasets, all_time_slots, instance_numbers,
                                     energy_profiles))
    for dataset, time_slots, instance_number, energy_profile in product:
        if show_running_instance:
            print(50 * "-")
            print("{}, {}t, {}, {}".format(dataset, time_slots, instance_number,
                                           energy_profile))
            print("Started at: {}".format(str(datetime.now().time())))
            count += 1
            print("({} of {})".format(count, len(product)))
            print(50 * "-")
        exec_and_save_static_energy(dataset,
                                    time_slots,
                                    instance_number,
                                    energy_profile,
                                    save_result=save_result,
                                    save_stats=save_stats,
                                    overwrite_if_result_exist=overwrite_if_result_exist,
                                    overwrite_if_stats_exist=overwrite_if_stats_exist,
                                    show_result_stats=show_result_stats,
                                    skip_if_exist=skip_if_exist,
                                    **kwargs)


def exec_and_save_all_distances_based(
        datasets: List[str],
        all_time_slots: List[int],
        instance_numbers: List[int],
        reference_profiles: List[int],
        energy_production_functions: List[str],
        months: List[int | str],
        model_types: List[ModelType],
        save_result: bool = True,
        save_stats: bool = True,
        overwrite_if_result_exist: bool = False,
        overwrite_if_stats_exist: bool = False,
        show_running_instance: bool = True,
        show_result_stats: bool = True,
        skip_if_exist: bool = False,
        **kwargs
) -> None:
    count = 0
    product = list(itertools.product(datasets,
                                     all_time_slots,
                                     instance_numbers,
                                     reference_profiles,
                                     energy_production_functions,
                                     months,
                                     model_types))
    for values in product:
        dataset, time_slots, instance_number, reference_profile, \
            energy_production_function, month, model_type = values
        if show_running_instance:
            month_num = utils.get_month_number(month)
            print(50 * "-")
            print("{}, {}t, {}, {}, {}, {}".format(dataset,
                                                   time_slots,
                                                   instance_number,
                                                   energy_production_function,
                                                   calendar.month_name[month_num],
                                                   model_type.name))
            print("Started at: {}".format(str(datetime.now().time())))
            count += 1
            print("({} of {})".format(count, len(product)))
            print(50 * "-")
        exec_and_save_distances_based(dataset,
                                      time_slots,
                                      instance_number,
                                      reference_profile,
                                      energy_production_function,
                                      month,
                                      model_type,
                                      save_result=save_result,
                                      save_stats=save_stats,
                                      overwrite_if_result_exist=overwrite_if_result_exist,
                                      overwrite_if_stats_exist=overwrite_if_stats_exist,
                                      show_result_stats=show_result_stats,
                                      skip_if_exist=skip_if_exist,
                                      **kwargs)
