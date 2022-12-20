import datetime
from unittest import TestCase

import mip

from pycode.src.ResultStats import ResultStats


class TestResultStats(TestCase):
    def setUp(self) -> None:
        self.result_stats = ResultStats(status=mip.OptimizationStatus.OPTIMAL,
                                        time=datetime.timedelta(minutes=12, seconds=3),
                                        obj_value=100.1234,
                                        obj_bound=200.1)

    def test_correct_init(self):
        self.assertEqual(mip.OptimizationStatus.OPTIMAL, self.result_stats.status)

        self.assertEqual(datetime.timedelta(0, minutes=12, seconds=3),
                         self.result_stats.time)

        self.assertEqual(100.1234, self.result_stats.obj_value)

        self.assertEqual(200.1, self.result_stats.obj_bound)

        self.assertEqual("status: OPTIMAL, time: 0:12:03, objective value: 100.12, "
                         "objective bound: 200.10", str(self.result_stats))

        self.assertEqual("status: OPTIMAL, time: 0:12:03, objective value: 100.12, "
                         "objective bound: 200.10", repr(self.result_stats))

    def test_get_status_name(self):
        self.assertEqual(self.result_stats.status.name, "OPTIMAL")

    def test_to_list(self):
        self.assertEqual(self.result_stats.to_list(),
                         [datetime.timedelta(minutes=12, seconds=3),
                          mip.OptimizationStatus.OPTIMAL, 100.1234, 200.1])

    def test_append(self):
        other = ResultStats(
            status=mip.OptimizationStatus.OPTIMAL,
            time=datetime.timedelta(minutes=45, seconds=67, milliseconds=89),
            obj_value=1000,
            obj_bound=2000
        )

        expected_stats = ResultStats(
            status=mip.OptimizationStatus.OPTIMAL,
            time=datetime.timedelta(minutes=57, seconds=70, milliseconds=89),
            obj_value=1100.1234,
            obj_bound=2200.1
        )

        self.assertEqual(self.result_stats.add(other), self.result_stats + other)

        self.assertEqual(self.result_stats.add(other), other.add(self.result_stats))

        self.assertEqual(expected_stats, self.result_stats.add(other))

        expected_float = ResultStats(
            status=mip.OptimizationStatus.OPTIMAL,
            time=datetime.timedelta(minutes=12, seconds=3),
            obj_value=125.1284,
            obj_bound=200.1
        )

        self.assertEqual(expected_float, self.result_stats.add(25.005))

    def test_equals(self):
        other = ResultStats(
            status=mip.OptimizationStatus.OPTIMAL,
            time=datetime.timedelta(minutes=45, seconds=67, milliseconds=89),
            obj_value=1000,
            obj_bound=2000
        )

        self.assertTrue(self.result_stats == self.result_stats)

        self.assertFalse(self.result_stats == other)

        self.assertFalse(self.result_stats == "string")

        self.assertTrue(self.result_stats != other)

        self.assertFalse(self.result_stats != self.result_stats)
