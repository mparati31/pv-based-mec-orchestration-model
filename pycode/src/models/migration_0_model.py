"""Implementation of model with migration at cost 0"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import mip
import numpy as np

from pycode.src.ResultStats import ResultStats
from pycode.src.instance import EnergyInstance
from pycode.src.result import Result


# noinspection DuplicatedCode,PyPep8Naming
def run_instance(
        instance: EnergyInstance,
        solver: str,
        gap: float,
        threads: int,
        verbose: bool
) -> Tuple[Result, ResultStats]:
    A = np.arange(instance.n_ap)
    K = np.arange(instance.n_facility)

    model = mip.Model("migration_0_model", solver_name=solver)
    model.verbose = int(verbose)
    model.threads = threads
    model.max_mip_gap = gap
    model.emphasis = mip.SearchEmphasis.OPTIMALITY

    # Variables.
    x = np.array([[model.add_var(var_type=mip.BINARY) for _ in K] for _ in A])
    g = np.array([model.add_var(var_type=mip.CONTINUOUS, lb=0) for _ in K])
    v = np.array([model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=instance.C[k])
                  for k in K])
    z = np.array([model.add_var(var_type=mip.CONTINUOUS, lb=0) for _ in K])

    # Objective function.
    model.objective = mip.minimize(
        instance.beta * mip.xsum(instance.d[0, i] * instance.m[i, k] * x[i, k] for k in K
                                 for i in A) +
        instance.gamma * mip.xsum(instance.c[0, k] * z[k] for k in K)
    )

    # Constraints.
    # (1)
    for k in K:
        model += mip.xsum(instance.d[0, i] * x[i, k] for i in A) == v[k]
    # (2)
    for i in A:
        model += mip.xsum(x[i, k] for k in K) == 1
    # (3)
    for k in K:
        # noinspection PyUnresolvedReferences
        model += z[k] + instance.e[0, k] + instance.p[k] == v[k] + g[k]

    # Find the optimal.
    timestamp_start = datetime.now()
    status = model.optimize()
    timestamp_end = datetime.now()

    x_correct_shape = np.reshape(x, (instance.time_slots, instance.n_ap,
                                     instance.n_facility))
    g_correct_shape = np.reshape(g, (instance.time_slots, instance.n_facility))
    v_correct_shape = np.reshape(v, (instance.time_slots, instance.n_facility))
    z_correct_shape = np.reshape(z, (instance.time_slots, instance.n_facility))

    result = Result.from_variables(x=x_correct_shape,
                                   g=g_correct_shape,
                                   v=v_correct_shape,
                                   z=z_correct_shape,
                                   g_ub=instance.G,
                                   make_copy=False)
    stats = ResultStats(status=status,
                        time=timestamp_end - timestamp_start,
                        obj_value=model.objective_value,
                        obj_bound=model.objective_bound)
    return result, stats
