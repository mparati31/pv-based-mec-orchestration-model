from unittest import TestCase

import numpy.testing

from pycode.src.result import *
from pycode.src.result import (_from_conn_str_to_list, _from_dataframe_to_g,
                               _from_dataframe_to_v, _from_dataframe_to_x,
                               _from_dataframe_to_z)

RESOURCES = Path(__file__).parent / "resources"


# noinspection DuplicatedCode
class TestResult(TestCase):
    def setUp(self) -> None:
        self.n_ap = 4
        self.n_facility = 2
        self.time_slots = 3

        self.df = pd.DataFrame.from_dict(
            data={
                "t": [0, 0, 1, 1, 2, 2],
                "k": [0, 1, 0, 1, 0, 1],
                "g": [10., 11., 12., 13., 14., 15.],
                "v": [20., 21., 22., 23., 24., 25.],
                "z": [30., 31., 32., 33., 34., 35.],
                "conn": [
                    "0 1 2 3",
                    "",
                    "2",
                    "0 1 3",
                    "1 3",
                    "0 2"
                ]
            }
        )
        self.df.set_index(["t", "k"], inplace=True)

        self.x = np.array([
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0]
            ],
            [
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1]
            ],
            [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0]
            ]
        ])

        self.g = np.array([
            [10., 11],
            [12, 13],
            [14, 15]
        ])

        self.v = np.array([
            [20., 21],
            [22, 23],
            [24, 25]
        ])

        self.z = np.array([
            [30., 31],
            [32, 33],
            [34, 35]
        ])

    def test_from_conn_str_to_list(self):
        self.assertEqual([], _from_conn_str_to_list(""))

        self.assertEqual([1, 1], _from_conn_str_to_list("1 1"))

        self.assertEqual([1, 2, 3, 4, 5], _from_conn_str_to_list("1 2 3 4 5"))

    def test_from_dataframe_to_x(self):
        np.testing.assert_array_equal(
            self.x,
            _from_dataframe_to_x(self.df, self.n_ap, self.n_facility, self.time_slots)
        )

    def test_from_dataframe_to_g(self):
        np.testing.assert_array_equal(
            self.g,
            _from_dataframe_to_g(self.df, self.n_facility, self.time_slots)
        )

    def test_from_dataframe_to_v(self):
        np.testing.assert_array_equal(
            self.v,
            _from_dataframe_to_v(self.df, self.n_facility, self.time_slots)
        )

    def test_from_dataframe_to_z(self):
        np.testing.assert_array_equal(
            self.z,
            _from_dataframe_to_z(self.df, self.n_facility, self.time_slots)
        )
