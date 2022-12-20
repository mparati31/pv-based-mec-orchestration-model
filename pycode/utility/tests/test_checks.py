from unittest import TestCase

from pycode.utility.checks import *


class Test(TestCase):
    def test_check_inside(self):
        self.assertIsNone(check_inside(1, [1, 2, 3, 4, 5], "param"))

        with self.assertRaisesRegex(Exception,
                                    "The value of 'param' must be one of the "
                                    "following: 1, 2, 3, 4, 5."):
            check_inside(1, ["1", "2", "3", "4", "5"], "param")

    def test_check_between(self):
        self.assertIsNone(check_between(0.5, 0.0, 1.0, "param"))

        with self.assertRaisesRegex(Exception,
                                    "The value of 'param' must be between 0.0 and 1.0."):
            check_between(-0.2, 0.0, 1.0, "param")

        with self.assertRaisesRegex(Exception,
                                    "The value of 'param' must be between 0.0 and 1.0."):
            check_between(1.2, 0.0, 1.0, "param")

    def test_check_strictly_positive(self):
        self.assertIsNone(check_strictly_positive(1, "param"))

        with self.assertRaisesRegex(Exception,
                                    "The value of 'param' must be strictly positive."):
            check_strictly_positive(0, "param")

    def test_check_sum_equal(self):
        self.assertIsNone(check_sum_equal([0.5, 0.5], 1.0, ("param1", "param2")))

        with self.assertRaisesRegex(Exception,
                                    "The sum of 'param1', 'param2' must be 1.0"):
            check_sum_equal([0.5, 0.6], 1.0, ("param1", "param2"))

    def test_check_shape(self):
        self.assertIsNone(check_shape(np.zeros((2, 5)), (2, 5), "param"))

        with self.assertRaisesRegex(Exception,
                                    r"The shape of 'param' must be \(2, 5\)."):
            check_shape(np.zeros((5, 2)), (2, 5), "param")
