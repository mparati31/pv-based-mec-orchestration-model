"""Implementation of model to generate array splits."""

import itertools
from datetime import datetime, timedelta
from numbers import Real
from typing import Tuple

import mip
import numpy as np


def _extract_split_points(x: np.ndarray) -> np.ndarray:
    extract_var_value = np.vectorize(lambda var: var.x)
    x_values = extract_var_value(x)
    split_points = []
    for sub_array in x_values:
        split_points += [np.where(sub_array > 0.9)[0][-1]]
    split_points.sort()
    return np.array(split_points[:-1])


def run_instance(
        array: np.ndarray,
        n_chunks: int,
        m: float,
        solver: str,
        gap: float,
        threads: int,
        verbose: bool
) -> Tuple[np.ndarray, mip.OptimizationStatus, Real, timedelta]:
    I = np.arange(n_chunks)
    J = np.arange(len(array))

    model = mip.Model("generate_splits_model", solver_name=solver)
    model.verbose = int(verbose)
    model.threads = threads
    model.max_mip_gap = gap
    model.emphasis = mip.SearchEmphasis.OPTIMALITY

    # Variables.
    x = np.array([[model.add_var(var_type=mip.BINARY) for _ in J] for _ in I])
    s = np.array([[model.add_var(var_type=mip.BINARY) for _ in J] for _ in I])
    d = np.array([model.add_var(var_type=mip.CONTINUOUS) for _ in I])

    # Objective function.
    model.objective = mip.minimize(
        mip.xsum(d[i] for i in I)
    )

    # Constraints.
    # (1)
    for j in J:
        model += mip.xsum(x[i, j] for i in I) == 1
    # (2)
    for i in I:
        model += mip.xsum(x[i, j] for j in J) >= 1
    # (3)
    model += x[0, 0] == 1
    # (4)
    model += s[0, 0] == 1
    # (5)
    for i in I[1:]:
        model += s[i, 0] == 0
    # (6)
    for i, j in itertools.product(I, J[1:]):
        model += s[i, j] >= x[i, j] - x[i, j - 1]
    # (7)
    for i in I:
        model += mip.xsum(s[i, j] for j in J) == 1
    # (8)
    for i in I:
        model += d[i] >= mip.xsum(x[i, j] * array[j] for j in J) - m
    # (9)
    for i in I:
        model += d[i] >= -(mip.xsum(x[i, j] * array[j] for j in J) - m)

    # Find the optimal.
    timestamp_start = datetime.now()
    status = model.optimize()
    timestamp_end = datetime.now()

    return _extract_split_points(x), status, model.objective_value, \
        timestamp_end - timestamp_start
