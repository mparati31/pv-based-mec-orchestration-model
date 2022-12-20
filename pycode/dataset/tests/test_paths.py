from unittest import TestCase

from pycode.dataset.paths import *


class Test(TestCase):
    def test_resolve_facilities_positions(self):
        self.assertTrue(resolve_facilities_positions().exists())

    def test_resolve_facility_irradiation(self):
        self.assertTrue(resolve_facility_irradiation(0).exists())

        self.assertTrue(resolve_facility_irradiation(1).exists())

        self.assertRaises(Exception, resolve_facility_irradiation, 10)

    def test_resolve_splits(self):
        self.assertTrue(resolve_lad_splits(1).exists())

        self.assertTrue(resolve_lsd_splits(1).exists())

        self.assertRaises(Exception, resolve_lad_splits, 0)

        self.assertRaises(Exception, resolve_lsd_splits, 6)

    def test_resolve_instance(self):
        self.assertTrue(resolve_instance("A", 12, 1).exists())

        self.assertTrue(resolve_instance("B", 96, 3).exists())

        self.assertRaises(Exception, resolve_instance, "C", 12, 1)

        self.assertRaises(Exception, resolve_instance, "A", 11, 1)

        self.assertRaises(Exception, resolve_instance, "A", 12, 0)

    def test_resolve_static_energy_result(self):
        self.assertTrue(resolve_static_energy_result("B", 48, 1, 2).exists())

        self.assertRaises(Exception, resolve_instance, "A", 12, 1, 0)

    def test_resolve_static_energy_results_stats(self):
        self.assertTrue(resolve_static_energy_results_stats().exists())

    def test_resolve_distances_energy_result(self):
        self.assertTrue(resolve_distances_energy_result("B", 48, 1, "constant", 2, 7,
                                                        ModelType.NORMAL).exists())

        self.assertTrue(resolve_distances_energy_result("B", 48, 1, "linear", 2, "jul",
                                                        ModelType.HEURISTIC_MIGRATION_0)
                        .exists())

        self.assertTrue(resolve_distances_energy_result("B", 48, 1, "linear", 2, 9,
                                                        ModelType.HEURISTIC_MIGRATION_INF)
                        .exists())

        self.assertRaises(Exception, resolve_instance, "B", 48, 3, "constant", 2, 7,
                          ModelType.NORMAL)

        self.assertRaises(Exception, resolve_instance, "B", 48, 2, "const", 2, 7,
                          ModelType.NORMAL)

        self.assertRaises(Exception, resolve_instance, "B", 48, 2, "constant", 3, 7,
                          ModelType.NORMAL)

        self.assertRaises(Exception, resolve_instance, "B", 48, 2, "constant", 2, 13,
                          ModelType.NORMAL)

    def test_resolve_distances_based_energy_results_stats(self):
        self.assertTrue(resolve_distances_based_energy_results_stats().exists())
