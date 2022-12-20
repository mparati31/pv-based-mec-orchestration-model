"""Implementation of normal energy model."""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Tuple

import mip
import numpy as np

from pycode.src.ResultStats import ResultStats
from pycode.src.instance import EnergyInstance
from pycode.src.result import Result


# noinspection PyArgumentList,PyPep8Naming,DuplicatedCode
def run_instance(
        instance: EnergyInstance,
        solver: str,
        gap: float,
        threads: int,
        verbose: bool
) -> Tuple[Result, ResultStats]:
    A = np.arange(instance.n_ap)
    K = np.arange(instance.n_facility)
    T = np.arange(instance.time_slots)

    model = mip.Model("energy_model", solver_name=solver)
    model.verbose = int(verbose)
    model.threads = threads
    model.max_mip_gap = gap
    model.emphasis = mip.SearchEmphasis.OPTIMALITY

    # Variables.
    x = np.array([[[model.add_var(var_type=mip.BINARY)
                    for _ in K] for _ in A] for _ in T])
    y = np.array([[[[model.add_var(var_type=mip.BINARY)
                     for _ in K] for _ in K] for _ in A] for _ in T])
    g = np.array([[model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=instance.G[k])
                   for k in K] for _ in T])
    v = np.array([[model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=instance.C[k])
                   for k in K] for _ in T])
    z = np.array([[model.add_var(var_type=mip.CONTINUOUS, lb=0)
                   for _ in K] for _ in T])

    # Objective function.
    model.objective = mip.minimize(
        instance.alpha * mip.xsum(instance.d[t, i] * instance.l[j, k] * y[t, i, j, k]
                                  for (j, k) in itertools.product(K, K) for i in A
                                  for t in T) +
        instance.beta * mip.xsum(instance.d[t, i] * instance.m[i, k] * x[t, i, k]
                                 for k in K for i in A for t in T) +
        instance.gamma * mip.xsum(instance.c[t, k] * z[t, k] for k in K for t in T)
    )

    # Constraints.
    # (1)
    for k, t in itertools.product(K, T):
        model += mip.xsum(instance.d[t, i] * x[t, i, k] for i in A) == v[t, k]
    # (2)
    for i, t, in itertools.product(A, T):
        model += mip.xsum(x[t, i, k] for k in K) == 1
    # (3)
    for i, k, t in itertools.product(A, K, T[1:]):
        model += x[t, i, k] == mip.xsum(y[t, i, l, k] for l in K)
    # (4)
    for i, k, t in itertools.product(A, K, T[:-1]):
        model += x[t, i, k] == mip.xsum(y[t + 1, i, k, l] for l in K)
    # (5)
    for k in K:
        t_min = T.min()
        model += z[t_min, k] + instance.e[t_min, k] + instance.p[k] >= \
                 v[t_min, k] + g[t_min, k]
    # (6)
    for k, t in itertools.product(K, T[1:]):
        model += z[t, k] + instance.e[t, k] + g[t - 1, k] >= v[t, k] + g[t, k]

    # Find the optimal.
    timestamp_start = datetime.now()
    status = model.optimize()
    timestamp_end = datetime.now()

    result = Result.from_variables(x=x, g=g, v=v, z=z, make_copy=False)
    stats = ResultStats(status=status,
                        time=timestamp_end - timestamp_start,
                        obj_value=model.objective_value,
                        obj_bound=model.objective_bound)
    return result, stats
