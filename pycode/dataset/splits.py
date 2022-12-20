"""Contains functions for loading split points."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load(filepath: str | Path) -> List[np.ndarray[np.int]]:
    with open(filepath, "r") as f:
        lines = np.array(f.readlines())

    header_lines = 4
    splits = np.array([header_lines, 12, header_lines, 24, header_lines])
    split_indexes = [splits[:i].sum() + n_rows for i, n_rows in enumerate(splits)]
    _, splits_12, _, splits_24, _, splits_48 = np.split(lines, split_indexes)

    return [splits_12, splits_24, splits_48]


def load_splits(filepath: str | Path) -> Dict[int, np.ndarray[np.int]]:
    extract_splits = lambda splits_txt: map(lambda x: re.findall("[0-9]+", x)[-1:][0],
                                            splits_txt[:-1])

    splits_12, splits_24, splits_48 = _load(filepath)

    # All split points should be decremented by 1 since in the file they are indexed
    # from 1 to 96, but in the model from 0 to 95.
    return {
        12: np.fromiter(extract_splits(splits_12), dtype=int) - 1,
        24: np.fromiter(extract_splits(splits_24), dtype=int) - 1,
        48: np.fromiter(extract_splits(splits_48), dtype=int) - 1
    }


def load_medians(filepath: str | Path) -> Dict[int, np.ndarray[np.int]]:
    extract_medians = lambda splits_txt: map(lambda x: re.findall("[0-9]+", x)[0],
                                             splits_txt)

    splits_12, splits_24, splits_48 = _load(filepath)

    # All split points should be decremented by 1 since in the file they are indexed
    # from 1 to 96, but in the model from 0 to 95.
    return {
        12: np.fromiter(extract_medians(splits_12), dtype=int) - 1,
        24: np.fromiter(extract_medians(splits_24), dtype=int) - 1,
        48: np.fromiter(extract_medians(splits_48), dtype=int) - 1
    }
