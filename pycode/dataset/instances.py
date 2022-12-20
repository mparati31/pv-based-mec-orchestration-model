"""Contains functions for loading instances."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from pycode.src.instance import BaseInstance, EnergyInstance
from pycode.utility.checks import check_between, check_shape
from pycode.utility.utils import from_str_array_to_float_matrix


def load_base(filepath: str | Path) -> BaseInstance:
    path = Path(filepath)
    with open(path, "r") as file_reader:
        lines = np.array(file_reader.readlines())

    n_ap = int(lines[0])
    n_facility = int(lines[1])

    # Calculates indexes separating data within the file.
    n_param_lines = 7
    splits = np.array([n_param_lines, n_ap, n_facility, n_ap])
    split_indexes = [splits[:i].sum() + n_rows for i, n_rows in enumerate(splits)]

    params, demands, facilities_dist, aps_facilities_dist, _ = np.split(lines,
                                                                        split_indexes)
    return BaseInstance(n_ap=n_ap,
                        n_facility=n_facility,
                        time_slots=int(params[2]),
                        alpha=float(params[3]),
                        beta=float(params[4]),
                        C=np.full(n_facility, int(params[6])),
                        d=from_str_array_to_float_matrix(demands).T,
                        l=from_str_array_to_float_matrix(facilities_dist),
                        m=from_str_array_to_float_matrix(aps_facilities_dist),
                        make_copy=False)


# noinspection PyPep8Naming
def load_with_energy_profile_1(
        filepath: str | Path,
        alpha: float = 0.3333333333333333,
        beta: float = 0.3333333333333333,
        gamma: float = 0.3333333333333333,
        initial_battery_percent: float = 0,
        check_alpha_beta_gamma_sum=True
) -> EnergyInstance:
    # Checks params.
    check_between(initial_battery_percent, 0.0, 1.0, "initial_battery_percent")

    base_instance = load_base(filepath)
    G = base_instance.C / 2
    c = np.full((base_instance.time_slots, base_instance.n_facility), 1)
    e = np.full((base_instance.time_slots, base_instance.n_facility),
                base_instance.C[0] / 2)
    initial_battery_charge = initial_battery_percent * G

    return EnergyInstance.from_base_instance(
        base_instance=base_instance,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        G=G,
        c=c,
        e=e,
        p=initial_battery_charge,
        check_alpha_beta_gamma_sum=check_alpha_beta_gamma_sum,
        make_copy=False
    )


# noinspection PyPep8Naming
def load_with_energy_profile_2(
        filepath: str | Path,
        alpha: float = 0.3333333333333333,
        beta: float = 0.3333333333333333,
        gamma: float = 0.3333333333333333,
        initial_battery_percent: float = 0,
        check_alpha_beta_gamma_sum=True
) -> EnergyInstance:
    # Checks params.
    check_between(initial_battery_percent, 0.0, 1.0, "initial_battery_percent")

    base_instance = load_base(filepath)
    sigma = base_instance.d.sum() / (base_instance.n_facility * base_instance.time_slots)
    G = np.full(base_instance.n_facility, sigma)
    c = np.full((base_instance.time_slots, base_instance.n_facility), 1)
    e = np.full((base_instance.time_slots, base_instance.n_facility), sigma)
    initial_battery_charge = initial_battery_percent * G

    return EnergyInstance.from_base_instance(
        base_instance=base_instance,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        G=G,
        c=c,
        e=e,
        p=initial_battery_charge,
        check_alpha_beta_gamma_sum=check_alpha_beta_gamma_sum,
        make_copy=False
    )


# noinspection PyArgumentList
def load_with_distances_based_energy_data(
        reference_instance: EnergyInstance,
        energy_production_func: Callable[[int | float, int | float], int | float],
        distances_from_center: np.ndarray,
        irradiation: np.ndarray,
        alpha: float = 0.3333333333333333,
        beta: float = 0.3333333333333333,
        gamma: float = 0.3333333333333333,
        initial_battery_percent: float = 0,
        check_alpha_beta_gamma_sum=True
) -> EnergyInstance:
    # Checks params.
    check_shape(distances_from_center, (reference_instance.n_facility,),
                "distances_from_center")
    check_shape(irradiation, (24, reference_instance.n_facility), "irradiation")
    check_between(initial_battery_percent, 0.0, 1.0, "initial_battery_percent")
    # For others number of time slots does not work.
    assert reference_instance.time_slots in [24, 48, 96]

    max_distance = distances_from_center.max()
    energy_production_func_map = map(lambda d: energy_production_func(d, max_distance),
                                     distances_from_center)
    energy_production_func_values = np.fromiter(energy_production_func_map, dtype=float)
    mu = reference_instance.e.sum() / (energy_production_func_values *
                                       irradiation).sum()
    production = mu * energy_production_func_values
    e_24t = production * irradiation / (reference_instance.time_slots // 24)
    e = e_24t.repeat(reference_instance.time_slots // 24, axis=0)
    initial_battery_charge = initial_battery_percent * reference_instance.G

    return EnergyInstance(n_ap=reference_instance.n_ap,
                          n_facility=reference_instance.n_facility,
                          time_slots=reference_instance.time_slots,
                          alpha=alpha,
                          beta=beta,
                          gamma=gamma,
                          C=reference_instance.C,
                          G=reference_instance.G,
                          d=reference_instance.d,
                          l=reference_instance.l,
                          m=reference_instance.m,
                          c=reference_instance.c,
                          e=e,
                          p=initial_battery_charge,
                          check_alpha_beta_gamma_sum=check_alpha_beta_gamma_sum,
                          make_copy=False)
