from pathlib import Path
from unittest import TestCase

import numpy as np
import numpy.testing

from pycode.dataset import splits

resources = Path(__file__).parent / "resources"


class Test(TestCase):
    def setUp(self) -> None:
        self.filepath = resources / "splits.log"

    def test_load_splits(self):
        expected = {
            12: np.array([6, 9, 13, 21, 30, 48, 55, 71, 75, 80, 85]),
            24: np.array([4, 7, 8, 10, 13, 21, 29, 32, 45, 48, 51, 55, 60, 65, 70, 71, 73,
                          75, 76, 79, 81, 84, 88]),
            48: np.array([2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 22, 29, 30, 31, 32,
                          45, 48, 49, 51, 55, 58, 60, 61, 62, 63, 64, 66, 69, 70, 71,
                          73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 90])
        }
        splits_loaded = splits.load_splits(self.filepath)

        self.assertEqual(expected.keys(), splits_loaded.keys())

        np.testing.assert_array_equal(expected[12], splits_loaded[12])

        np.testing.assert_array_equal(expected[24], splits_loaded[24])

        np.testing.assert_array_equal(expected[48], splits_loaded[48])
