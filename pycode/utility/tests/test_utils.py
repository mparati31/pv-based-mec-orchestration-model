from unittest import TestCase

from pycode.utility.utils import *


class Test(TestCase):

    def test_get_month_number(self):
        self.assertEqual(5, get_month_number(5))

        with self.assertRaisesRegex(Exception,
                                    "The month number must be between 1 and 12."):
            get_month_number(13)

        self.assertEqual(4, get_month_number("Apr"))

        self.assertEqual(1, get_month_number("jan"))

        self.assertEqual(3, get_month_number("mAr"))

        with self.assertRaisesRegex(Exception,
                                    "The string 'Arp' does not represents a month."):
            get_month_number("Arp")

        self.assertEqual(10, get_month_number("October"))

        self.assertEqual(12, get_month_number("december"))

        self.assertEqual(7, get_month_number("JULY"))

        with self.assertRaisesRegex(Exception,
                                    "The string 'Juli' does not represents a month."):
            get_month_number("Juli")

        self.assertRaises(AssertionError, get_month_number, 1.0)
