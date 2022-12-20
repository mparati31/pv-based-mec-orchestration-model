from pathlib import Path
from unittest import TestCase

import pandas as pd

from pycode.dataset import result_stats

resources = Path(__file__).parent / "resources"


class Test(TestCase):
    def setUp(self) -> None:
        self.static_filepath = resources / "static_energy_stats.csv"

        self.static_energy_stats_data = {
            "dataset": ["A", "B"],
            "slot_time": [10, 20],
            "instance_number": [1, 2],
            "energy_profile": [3, 4],
            "exec_time": ["0:15:03", "1:32:12.123"],
            "status": ["OPTIMAL", "OPTIMAL"],
            "obj_value": [1000, 2000],
            "obj_bound": [1001, 2002]
        }
        self.static_energy_stats = pd.DataFrame(self.static_energy_stats_data)
        self.static_energy_stats.set_index(
            ["dataset", "slot_time", "instance_number", "energy_profile"],
            inplace=True
        )

        self.distance_based_filepath = resources / "distance_based_energy_stats.csv"
        self.distance_based_energy_stats_data = {
            "type": ["HEURISTIC_MIGRATION_0", "NORMAL"],
            "dataset": ["A", "B"],
            "slot_time": [10, 20],
            "instance_number": [1, 2],
            "month": ["Jul", "Sep"],
            "reference_profile": [3, 4],
            "energy_production_function": ["constant", "linear"],
            "exec_time": ["0:15:03", "1:32:12.123"],
            "status": ["OPTIMAL", "OPTIMAL"],
            "obj_value": [1000, 2000],
            "obj_bound": [1001, 2002]
        }
        self.distance_based_energy_stats = pd.DataFrame(
            self.distance_based_energy_stats_data
        )
        self.distance_based_energy_stats.set_index(
            ["type", "dataset", "slot_time", "instance_number", "month",
             "reference_profile", "energy_production_function"],
            inplace=True
        )

    def test_load_static_energy_stats(self):
        pd.testing.assert_frame_equal(
            self.static_energy_stats,
            result_stats.load_static_energy(self.static_filepath)
        )

    def test_write_static_energy_stats(self):
        pass

    def test_load_distances_based_energy_stats(self):
        pd.testing.assert_frame_equal(
            self.distance_based_energy_stats,
            result_stats.load_distances_based_energy(self.distance_based_filepath)
        )

    def test_write_distances_based_energy_stats(self):
        pass
