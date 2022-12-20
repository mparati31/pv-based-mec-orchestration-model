"""Contains functions used to compute instances using each model type."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import mip
import numpy as np

from pycode.dataset.data import ModelType
from pycode.src.ResultStats import ResultStats
from pycode.src.instance import EnergyInstance
from pycode.src.models import energy_model, migration_0_model, migration_inf_model, \
    migration_inf_model_energy_tracked
from pycode.src.result import EmptyResult, Result


def _execute_splitted(
        instance: EnergyInstance,
        splits: Sequence[int],
        model: Callable[[EnergyInstance, str, float, int, bool],
        Tuple[Result, ResultStats]],
        solver: str,
        gap: int,
        threads: int,
        verbose: bool
) -> Tuple[EmptyResult | Result, ResultStats]:
    result_tot = EmptyResult(instance.n_ap, instance.n_facility)
    stats_tot = ResultStats()

    sub_instances = instance.split(splits)
    # Each sub-instance is performed independently and the results merged.
    for i, sub in enumerate(sub_instances):
        # In the first sub-model the energy initially present in the facility
        # battery is that in the instance, while from the second onward it is
        # that advanced from the last time slot of the previous sub-model.
        if i > 0:
            # noinspection PyUnresolvedReferences
            sub.p = result_tot.g[-1]

        result_sub, stats_sub = model(sub, solver, gap, threads, verbose)

        # Aps migration cost.
        migration_cost = 0
        if i > 0:
            # Calculates the cost of ap migrations between the previous
            # sub-instance and the current sub-instance.
            # The total cost is the sum of the individual costs.
            # For each ap, it checks whether the facility to which it was
            # assigned in the last time slot of the previous instance is
            # different from the one to which it is assigned in the first
            # slot of the next instance (k_old != k_new): in this case,
            # the cost of the move is given by the product of the demand
            # of the ap in the last slot of the previous instance (given
            # that it represents the amount of traffic to be moved) and
            # the network distance between the two facilities.
            for ap in range(instance.n_ap):
                # noinspection PyUnresolvedReferences
                k_old = result_tot.get_facility_to_witch_is_connected(
                    result_tot.time_slots - 1,
                    ap
                )
                k_new = result_sub.get_facility_to_witch_is_connected(0, ap)
                if k_old != k_new:
                    migration_cost += sub_instances[i - 1].d[-1, ap] * instance.l[k_old,
                    k_new]

        result_tot += result_sub
        # The migration cost is multiplied by alpha to ensure that it has the
        # same weight across model types.
        stats_tot += stats_sub + (instance.alpha * migration_cost)

    return result_tot, stats_tot


def compose_result(
        instance: EnergyInstance,
        x: np.ndarray[int],
        split_points: Sequence[int]
) -> Tuple[Result, float]:
    A = range(instance.n_ap)
    K = range(instance.n_facility)
    x_full = np.zeros((instance.time_slots, instance.n_ap, instance.n_facility))
    g_full = np.zeros((instance.time_slots, instance.n_facility))
    v_full = np.zeros((instance.time_slots, instance.n_facility))
    z_full = np.zeros((instance.time_slots, instance.n_facility))
    migration_costs = 0
    t = 0
    mapping_len = np.concatenate(([split_points[0] + 1],
                                  np.diff(np.concatenate((split_points, [95])))))
    for j, map_len in enumerate(mapping_len):
        for _ in range(map_len):
            x_full[t] = x[j]
            if t == 0:
                p = np.zeros(instance.n_facility)
            else:
                p = g_full[t - 1]
            v_full[t] = [sum([instance.d[t, i] * x_full[t, i, k] for i in A]) for k in K]
            g_full[t] = [max(0, min(p[k] + instance.e[t, k] - v_full[t, k],
                                    instance.G[k]))
                         for k in K]
            z_full[t] = [max(0, v_full[t, k] - (p[k] + instance.e[t, k])) for k in K]

            t += 1

        if j > 0:
            for i in A:
                k_prev = np.argmax(x_full[t - 1 - mapping_len[j], i])
                k_next = np.argmax(x[0, i])
                if k_prev != k_next:
                    migration_costs += instance.d[t - 1 - mapping_len[j], i] * \
                                       instance.l[k_prev, k_next]

    obj = instance.alpha * migration_costs + \
          instance.beta * sum([instance.d[t, i] * instance.m[i, k] * x_full[t, i, k]
                               for k in K for i in A for t in range(96)]) + \
          instance.gamma * sum([instance.c[t, k] * z_full[t, k]
                                for k in K for t in range(96)])

    return Result(x_full, g_full, v_full, z_full), obj


def execute(
        instance: EnergyInstance,
        model_type: ModelType,
        split_points: Sequence[int] = None,
        medians: Sequence[int] = None,
        solver: str = mip.GUROBI,
        gap: int = 0.1,
        threads: int = -1,
        verbose: bool = True
) -> Tuple[Result, ResultStats]:
    if model_type == ModelType.NORMAL:
        return energy_model.run_instance(instance=instance,
                                         solver=solver,
                                         gap=gap,
                                         threads=threads,
                                         verbose=verbose)

    elif model_type == ModelType.HEURISTIC_MIGRATION_INF:
        return migration_inf_model.run_instance(instance=instance,
                                                solver=solver,
                                                gap=gap,
                                                threads=threads,
                                                verbose=verbose)

    elif model_type == ModelType.HEURISTIC_MIGRATION_0:
        # The model is split to generate sub-instances of 1 time slot.
        migr_0_split_points = range(instance.time_slots)
        return _execute_splitted(instance=instance,
                                 splits=migr_0_split_points,
                                 model=migration_0_model.run_instance,
                                 solver=solver,
                                 gap=gap,
                                 threads=threads,
                                 verbose=verbose)

    # All the following types of models run the instance by splitting and
    # computing it using the heuristic model with migrations to infinity
    # (with energy tracked).
    elif model_type in [ModelType.HEURISTIC_SPLIT_12_EQUALS,
                        ModelType.HEURISTIC_SPLIT_24_EQUALS,
                        ModelType.HEURISTIC_SPLIT_48_EQUALS,
                        ModelType.HEURISTIC_SPLIT_12_LAD,
                        ModelType.HEURISTIC_SPLIT_24_LAD,
                        ModelType.HEURISTIC_SPLIT_48_LAD,
                        ModelType.HEURISTIC_SPLIT_12_LSD,
                        ModelType.HEURISTIC_SPLIT_24_LSD,
                        ModelType.HEURISTIC_SPLIT_48_LSD]:
        if split_points is None:
            raise Exception("Parameter 'split_points' must be not None.")
        if instance.time_slots != 96:
            raise AssertionError()
        # The instance is shifted by 20 time slots so that it starts at 5
        # AM and ends at 4 AM (instead of 0 AM - 11 PM).
        instance = instance.shift(20)
        return _execute_splitted(instance=instance,
                                 splits=split_points,
                                 model=migration_inf_model_energy_tracked.run_instance,
                                 solver=solver,
                                 gap=gap,
                                 threads=threads,
                                 verbose=verbose)

    elif model_type in [ModelType.HEURISTIC_MEDIAN_12_LAD,
                        ModelType.HEURISTIC_MEDIAN_24_LAD,
                        ModelType.HEURISTIC_MEDIAN_48_LAD,
                        ModelType.HEURISTIC_MEDIAN_12_LSD,
                        ModelType.HEURISTIC_MEDIAN_24_LSD,
                        ModelType.HEURISTIC_MEDIAN_48_LSD]:
        if split_points is None:
            raise Exception("Parameter 'split_points' must be not None.")
        elif medians is None:
            raise Exception("Parameter 'medians' must be not None.")
        elif instance.time_slots != 96:
            raise AssertionError()
        # The instance is shifted by 20 time slots so that it starts at 5
        # AM and ends at 4 AM (instead of 0 AM - 11 PM).
        instance = instance.shift(20)
        simple_instance = instance[medians]
        result, stats = energy_model.run_instance(instance=simple_instance,
                                                  solver=solver,
                                                  gap=gap,
                                                  threads=threads,
                                                  verbose=verbose)
        result_full, obj = compose_result(instance, np.rint(result.x), split_points)

        return result_full, ResultStats(stats.status, stats.time, obj, 0)

    else:
        raise AssertionError()
