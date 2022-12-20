from pathlib import Path
from unittest import TestCase

import numpy as np

from pycode.dataset import instances
from pycode.src.instance import BaseInstance, EnergyInstance

resources = Path(__file__).parent / "resources"


# noinspection PyPep8Naming
class Test(TestCase):
    def setUp(self) -> None:
        self.instance_path = resources / "instance.dat"
        self.C = np.array([6000, 6000])
        self.d = np.array([
            [10., 20, 30],
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33]
        ]).T
        self.l = np.array([[0., 1], [1, 0]])
        self.m = np.array([
            [40., 44],
            [41, 45],
            [42, 46],
            [43, 47]
        ])
        self.instance = BaseInstance(n_ap=4,
                                     n_facility=2,
                                     time_slots=3,
                                     alpha=0.4,
                                     beta=0.6, C=self.C,
                                     d=self.d,
                                     l=self.l,
                                     m=self.m)

    def test_load_base_data(self):
        self.assertEqual(self.instance, instances.load_base(self.instance_path))

    def test_load_energy_data_profile_1(self):
        expected = EnergyInstance.from_base_instance(
            base_instance=self.instance,
            alpha=1 / 2,
            beta=1 / 4,
            gamma=1 / 4,
            G=np.array([3000., 3000]),
            c=np.array([[1., 1], [1, 1], [1, 1]]),
            e=np.array([[3000., 3000], [3000, 3000], [3000, 3000]]),
            p=np.array([1500., 1500])
        )

        self.assertEqual(expected,
                         instances.load_with_energy_profile_1(
                             filepath=self.instance_path,
                             alpha=1 / 2,
                             beta=1 / 4,
                             gamma=1 / 4,
                             initial_battery_percent=0.5)
                         )

    def test_load_energy_data_profile_2(self):
        expected = EnergyInstance.from_base_instance(
            base_instance=self.instance,
            alpha=1 / 2,
            beta=1 / 4,
            gamma=1 / 4,
            G=np.array([43., 43]),
            c=np.array([[1., 1], [1, 1], [1, 1]]),
            e=np.array([[43., 43], [43, 43], [43, 43]]),
            p=np.array([21.5, 21.5]))

        self.assertEqual(expected,
                         instances.load_with_energy_profile_2(
                             filepath=self.instance_path,
                             alpha=1 / 2,
                             beta=1 / 4,
                             gamma=1 / 4,
                             initial_battery_percent=0.5)
                         )

    def test_load_distances_based_energy_data(self):
        pass
