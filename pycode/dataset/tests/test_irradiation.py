import calendar
from pathlib import Path
from unittest import TestCase

import pandas as pd

from pycode.dataset import irradiation

resources = Path(__file__).parent / "resources"


class Test(TestCase):
    def setUp(self) -> None:
        self.filepath = resources / "irradiation.json"

    def test_load_data(self):
        expected = dict()
        i = 0
        for month in calendar.month_abbr[1:]:
            expected[month] = pd.DataFrame(data={1: [i, i + 1], 2: [i + 2, i + 3]})
            i += 10

        result = irradiation.load(self.filepath)

        self.assertEqual(expected.keys(), result.keys())
        for key in expected.keys():
            pd.testing.assert_frame_equal(expected[key], result[key])

    def test_load_months_avg(self):
        expected_data = dict()
        i = 1.
        for month in calendar.month_abbr[1:]:
            expected_data[month] = [i, i + 1]
            i += 10
        expected = pd.DataFrame(data=expected_data)

        pd.testing.assert_frame_equal(expected,
                                      irradiation.load_months_avg(self.filepath))
