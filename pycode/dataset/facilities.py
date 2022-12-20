"""Contains functions to load data concerning facility locations."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_positions(filepath: str | Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_distances_from_point(
        filepath: str | Path,
        point: Tuple[int | float, int | float]
) -> pd.Series:
    coords = load_positions(filepath)
    distances = map(lambda coord: np.linalg.norm(coord - point), coords.values)
    return pd.Series(distances)
