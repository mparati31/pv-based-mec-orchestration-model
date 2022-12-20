"""Generates split points."""

from datetime import timedelta
from typing import List, Sequence, Tuple

import mip
import numpy as np

from pycode.dataset import api, data, paths
from pycode.src.models.split_array_same_sum_chunks_model import run_instance


# noinspection PyShadowingNames
def run_heuristic(
        subs: Sequence[np.ndarray],
        n_chunks_subs: Sequence[int],
        m: float
) -> Tuple[List[np.ndarray], mip.OptimizationStatus, float, timedelta]:
    splits_tot, status_tot, obj_tot, time_tot = [], mip.OptimizationStatus.OPTIMAL, 0, \
        timedelta()
    for sub, n_chunks_sub in zip(subs, n_chunks_subs):
        splits, status, obj, time = run_instance(array=sub,
                                                 n_chunks=n_chunks_sub,
                                                 m=m,
                                                 solver="GUROBI",
                                                 gap=0.1,
                                                 threads=-1,
                                                 verbose=True)
        splits_tot += [splits]
        status_tot = status if status_tot == mip.OptimizationStatus.OPTIMAL \
            else status_tot
        obj_tot += obj
        time_tot += time

    return splits_tot, status_tot, obj_tot, time_tot


def generate_ranges_str(splits: np.ndarray) -> str:
    # Split points are increased by 1 so that the time slots are between 1 and 96.
    return "\n".join(["{}) {}".format(n, " ".join([str(i) for i in a + 1]))
                      for n, a in enumerate(np.split(np.arange(96), splits + 1))])


if __name__ == "__main__":
    for instance_number in data.instance_numbers:
        instance = api.get_base_instance("B", 96, instance_number).shift(20)
        d_time_slots = instance.d.sum(axis=1)

        output = ""

        # 12 sub instances (no heuristic).
        splits12, status12, obj12, time12 = run_instance(array=d_time_slots,
                                                         n_chunks=12,
                                                         m=d_time_slots.sum() / 12,
                                                         solver="GUROBI",
                                                         gap=0.1,
                                                         threads=-1,
                                                         verbose=True)
        assert status12 == mip.OptimizationStatus.OPTIMAL
        output += "12 SLOTS\n"
        output += "status: optimal,\nobjective: {:.2f},\ntime: {}\n".format(obj12, time12)
        output += "{}\n".format(generate_ranges_str(splits12))

        # 24 sub instances (splits d in two sub array of 48 and 48).
        subs = np.split(d_time_slots, [48])
        n_chunks_subs = [round((sub.sum() * 24) / d_time_slots.sum()) for sub in subs]
        assert sum(n_chunks_subs) == 24
        splits24_list, status24, obj24, time24 = run_heuristic(
            subs=subs,
            n_chunks_subs=n_chunks_subs,
            m=d_time_slots.sum() / 24
        )
        assert status24 == mip.OptimizationStatus.OPTIMAL
        splits24 = np.concatenate([splits24_list[0],
                                   [47],
                                   np.array(splits24_list[1]) + 48])
        output += "24 SLOTS (heuristic 24-24)\n"
        output += "status: optimal,\nobjective: {:.2f},\ntime: {}\n".format(obj24, time24)
        output += "{}\n".format(generate_ranges_str(splits24))

        # 48 sub instances (splits d in two sub array of 48 and 48).
        subs = np.split(d_time_slots, [48])
        n_chunks_subs = [round((sub.sum() * 48) / d_time_slots.sum()) for sub in subs]
        assert sum(n_chunks_subs) == 48
        splits48_list, status48, obj48, time48 = run_heuristic(
            subs=subs,
            n_chunks_subs=n_chunks_subs,
            m=d_time_slots.sum() / 48
        )
        assert status48 == mip.OptimizationStatus.OPTIMAL
        splits48 = np.concatenate([splits48_list[0],
                                   [47],
                                   np.array(splits48_list[1]) + 48])
        output += "48 SLOTS (heuristic 24-24)\n"
        output += "status: optimal, objective:\n{:.2f},\ntime: {}\n".format(obj48, time48)
        output += "{}".format(generate_ranges_str(splits48))

        # Save the results to a file.
        filepath = paths.resolve_equals_splits(instance_number)
        with open(filepath, "w") as outfile:
            outfile.write(output)
