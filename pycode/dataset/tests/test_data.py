from unittest import TestCase

from pycode.dataset.data import ModelType


class TestModelType(TestCase):
    def test_split_type(self):
        self.assertEqual("EQUALS", ModelType.HEURISTIC_SPLIT_12_EQUALS.split_type)

        self.assertEqual("LAD", ModelType.HEURISTIC_SPLIT_24_LAD.split_type)

        self.assertEqual("LSD", ModelType.HEURISTIC_SPLIT_48_LSD.split_type)

        self.assertEqual(None, ModelType.NORMAL.split_type)

    def test_n_sub_instances(self):
        self.assertEqual(12, ModelType.HEURISTIC_SPLIT_12_EQUALS.n_splits)

        self.assertEqual(24, ModelType.HEURISTIC_SPLIT_24_LAD.n_splits)

        self.assertEqual(48, ModelType.HEURISTIC_SPLIT_48_LSD.n_splits)

        self.assertEqual(None, ModelType.NORMAL.n_splits)
