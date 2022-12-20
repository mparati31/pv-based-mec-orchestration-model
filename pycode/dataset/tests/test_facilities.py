from pathlib import Path
from unittest import TestCase

import pandas as pd

from pycode.dataset import facilities

resources = Path(__file__).parent / "resources"


class Test(TestCase):
    def setUp(self) -> None:
        self.filepath = resources / "facility_positions.csv"

    def test_load_positions(self):
        expected = pd.DataFrame(data={"x": [0, 2, 0, 2], "y": [0, 0, 2, 2]})

        pd.testing.assert_frame_equal(expected, facilities.load_positions(self.filepath))

    def test_load_distances_from_point(self):
        expected = pd.Series([1.0, 2.236068, 1.0, 2.236068])

        pd.testing.assert_series_equal(
            expected,
            facilities.load_distances_from_point(self.filepath, (0, 1))
        )
