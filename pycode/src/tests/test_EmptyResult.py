from copy import copy
from unittest import TestCase

import numpy as np
import pandas as pd

from pycode.src.result import EmptyResult, Result


# noinspection DuplicatedCode
class TestEmptyResult(TestCase):
    def setUp(self) -> None:
        self.empty_result = EmptyResult(4, 2)

        df = pd.DataFrame.from_dict(
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
        df.set_index(["t", "k"], inplace=True)
        self.result = Result.from_pd_dataframe(df)

    def test_correct_init(self):
        self.assertEqual(4, self.empty_result.n_ap)

        self.assertEqual(2, self.empty_result.n_facility)

        self.assertEqual(0, len(self.empty_result))

        self.assertEqual(self.empty_result, copy(self.empty_result))

        self.assertRaises(Exception, self.empty_result.__getitem__, 0)

        self.assertEqual("Result of an instance composed of 0 time slots "
                         "with 4 access points and 2 facilities",
                         str(self.empty_result))

        self.assertEqual(str(self.empty_result), repr(self.empty_result))

    def test_wrong_init(self):
        self.assertRaises(Exception, EmptyResult, 0, 2)

        self.assertRaises(Exception, EmptyResult, 4, 0)

    def test_append(self):
        self.assertEqual(self.empty_result.append(self.result),
                         self.empty_result + self.result)

        expected = Result.from_pd_dataframe(self.result.to_pd_dataframe())

        self.assertEqual(self.empty_result.append(self.result), expected)

        self.assertRaises(Exception, self.empty_result.append, EmptyResult(1, 2))

        self.assertRaises(Exception, self.empty_result.append, EmptyResult(4, 1))

        wrong_ap_result = Result(x=np.zeros((1, 1, 2)),
                                 g=np.zeros((1, 2)),
                                 v=np.zeros((1, 2)),
                                 z=np.zeros((1, 2)))

        self.assertRaises(Exception, self.empty_result.append, wrong_ap_result)

        wrong_facility_result = Result(x=np.zeros((1, 4, 1)),
                                       g=np.zeros((1, 1)),
                                       v=np.zeros((1, 1)),
                                       z=np.zeros((1, 1)))

        self.assertRaises(Exception, self.empty_result.append, wrong_facility_result)

        self.assertRaises(Exception, self.empty_result.append, "string")

    def test_equals(self):
        self.assertTrue(self.empty_result == self.empty_result)

        self.assertFalse(self.empty_result == "string")

        other_empty_result = EmptyResult(4, 3)

        self.assertTrue(self.empty_result != other_empty_result)
