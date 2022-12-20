from copy import copy
from unittest import TestCase

import numpy as np
import numpy.testing

from pycode.src.instance import BaseInstance, EnergyInstance


# noinspection DuplicatedCode
class Test(TestCase):
    def setUp(self) -> None:
        self.instance = EnergyInstance(n_ap=4,
                                       n_facility=2,
                                       time_slots=3,
                                       alpha=0.2,
                                       beta=0.3,
                                       gamma=0.5,
                                       C=np.zeros(2),
                                       G=np.ones(2),
                                       d=np.arange(12).reshape((3, 4)),
                                       l=np.zeros((2, 2)),
                                       m=np.arange(8).reshape((4, 2)),
                                       c=np.arange(10, 61, 10).reshape((3, 2)),
                                       e=np.arange(100, 601, 100).reshape((3, 2)))

    def test_from_base_instance(self):
        base_instance = BaseInstance(n_ap=4,
                                     n_facility=2,
                                     time_slots=3,
                                     alpha=0.3,
                                     beta=0.7,
                                     C=np.zeros(2),
                                     d=np.arange(12).reshape((3, 4)),
                                     l=np.zeros((2, 2)),
                                     m=np.arange(8).reshape((4, 2)))

        local_instance = EnergyInstance.from_base_instance(
            base_instance=base_instance,
            alpha=0.2,
            beta=0.3,
            gamma=0.5,
            G=np.ones(2),
            c=np.arange(10, 61, 10).reshape((3, 2)),
            e=np.arange(100, 601, 100).reshape((3, 2))
        )

        self.assertEqual(self.instance, local_instance)

    def test_correct_init(self):
        self.assertEqual(0.5, self.instance.gamma)

        np.testing.assert_array_equal(np.ones(2), self.instance.G)

        np.testing.assert_array_equal(np.arange(10, 61, 10).reshape((3, 2)),
                                      self.instance.c)

        np.testing.assert_array_equal(np.arange(100, 601, 100).reshape((3, 2)),
                                      self.instance.e)

        np.testing.assert_array_equal(np.zeros(2), self.instance.p)

        self.instance.p = np.array([0.5, 0.6])

        np.testing.assert_array_equal(np.array([0.5, 0.6]), self.instance.p)

        self.assertRaises(Exception, self.instance.__setattr__, "p", np.array([2, 3]))

        self.assertEqual(self.instance, copy(self.instance))

        self.assertEqual("Energy Instance composed of 3 time slots with 4 access "
                         "points and 2 facilities",
                         str(self.instance))

    def test_wrong_init(self):
        # Wrong 'gamma'.
        self.assertRaises(Exception, EnergyInstance, 4, 2, 3, 0.2, 0.3, 2.5,
                          np.zeros(2), np.ones(2), np.arange(12).reshape((3, 4)),
                          np.zeros((2, 2)), np.arange(8).reshape((4, 2)),
                          np.arange(10, 61, 10).reshape((3, 2)),
                          np.arange(100, 601, 100).reshape((3, 2)))

        # Wrong sum 'alpha', 'beta' and 'gamma'.
        self.assertRaises(Exception, EnergyInstance, 4, 2, 3, 0.2, 0.35, 0.5,
                          np.zeros(2), np.ones(2), np.arange(12).reshape((3, 4)),
                          np.zeros((2, 2)), np.arange(8).reshape((4, 2)),
                          np.arange(10, 61, 10).reshape((3, 2)),
                          np.arange(100, 601, 100).reshape((3, 2)))

        # Wrong 'G' shape.
        self.assertRaises(Exception, EnergyInstance, 4, 2, 3, 0.2, 0.3, 0.5,
                          np.zeros(2), np.ones(3), np.arange(12).reshape((3, 4)),
                          np.zeros((2, 2)), np.arange(8).reshape((4, 2)),
                          np.arange(10, 61, 10).reshape((3, 2)),
                          np.arange(100, 601, 100).reshape((3, 2)))

        # Wrong 'c' shape.
        self.assertRaises(Exception, EnergyInstance, 4, 2, 3, 0.2, 0.3, 0.5, np.zeros(2),
                          np.ones(2), np.arange(12).reshape((3, 4)),
                          np.zeros((2, 2)), np.arange(8).reshape((4, 2)),
                          np.arange(10, 61, 10),
                          np.arange(100, 601, 100).reshape((3, 2)))

        # Wrong 'e' shape.
        self.assertRaises(Exception, EnergyInstance, 4, 2, 3, 0.2, 0.3, 0.5, np.zeros(2),
                          np.ones(2), np.arange(12).reshape((3, 4)),
                          np.zeros((2, 2)), np.arange(8).reshape((4, 2)),
                          np.arange(10, 61, 10).reshape((3, 2)),
                          np.arange(100, 601, 100))

    def test_shift(self):
        expected_s1 = EnergyInstance(n_ap=4,
                                     n_facility=2,
                                     time_slots=3,
                                     alpha=0.2,
                                     beta=0.3,
                                     gamma=0.5,
                                     C=np.zeros(2),
                                     G=np.ones(2),
                                     d=np.array([[4, 5, 6, 7], [8, 9, 10, 11],
                                                 [0, 1, 2, 3]]),
                                     l=np.zeros((2, 2)),
                                     m=np.arange(8).reshape((4, 2)),
                                     c=np.array([[30, 40], [50, 60], [10, 20]]),
                                     e=np.array([[300, 400], [500, 600], [100, 200]]))

        self.assertEqual(expected_s1, self.instance.shift(1))

        expected_s2 = EnergyInstance(n_ap=4,
                                     n_facility=2,
                                     time_slots=3,
                                     alpha=0.2,
                                     beta=0.3,
                                     gamma=0.5,
                                     C=np.zeros(2),
                                     G=np.ones(2),
                                     d=np.array([[8, 9, 10, 11], [0, 1, 2, 3],
                                                 [4, 5, 6, 7]]),
                                     l=np.zeros((2, 2)),
                                     m=np.arange(8).reshape((4, 2)),
                                     c=np.array([[50, 60], [10, 20], [30, 40]]),
                                     e=np.array([[500, 600], [100, 200], [300, 400]]))

        self.assertEqual(expected_s2, self.instance.shift(2))

        self.assertEqual(self.instance, self.instance.shift(0))

    def test_get_item(self):
        # Single slot time.

        self.assertEqual(
            EnergyInstance(n_ap=4,
                           n_facility=2,
                           time_slots=1,
                           alpha=0.2,
                           beta=0.3,
                           gamma=0.5,
                           C=np.zeros(2),
                           G=np.ones(2),
                           d=np.array([[4, 5, 6, 7]]),
                           l=np.zeros((2, 2)),
                           m=np.arange(8).reshape((4, 2)),
                           c=np.array([[30, 40]]),
                           e=np.array([[300, 400]])),
            self.instance[1]
        )

        self.assertRaises(Exception, self.instance.__getitem__, -1)

        self.assertRaises(Exception, self.instance.__getitem__, 5)

        # Slice.

        self.assertEqual(
            EnergyInstance(n_ap=4,
                           n_facility=2,
                           time_slots=2,
                           alpha=0.2,
                           beta=0.3,
                           gamma=0.5,
                           C=np.zeros(2),
                           G=np.ones(2),
                           d=np.array([[4, 5, 6, 7], [8, 9, 10, 11]]),
                           l=np.zeros((2, 2)),
                           m=np.arange(8).reshape((4, 2)),
                           c=np.array([[30, 40], [50, 60]]),
                           e=np.array([[300, 400], [500, 600]])),
            self.instance[1:3]
        )

        self.assertEqual(
            EnergyInstance(n_ap=4,
                           n_facility=2,
                           time_slots=2,
                           alpha=0.2,
                           beta=0.3,
                           gamma=0.5,
                           C=np.zeros(2),
                           G=np.ones(2),
                           d=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                           l=np.zeros((2, 2)),
                           m=np.arange(8).reshape((4, 2)),
                           c=np.array([[10, 20], [30, 40]]),
                           e=np.array([[100, 200], [300, 400]])),
            self.instance[0:-1]
        )

        self.assertEqual(self.instance[0:1], self.instance[0])

        self.assertEqual(self.instance[:100], self.instance)

        self.assertRaises(Exception, self.instance.__getitem__, slice(0, 1, 2))

        self.assertRaises(Exception, self.instance.__getitem__, slice(0, 0, 1))

        # Array-like.

        self.assertEqual(self.instance[0], self.instance[[0]])

        self.assertEqual(
            EnergyInstance(n_ap=4,
                           n_facility=2,
                           time_slots=2,
                           alpha=0.2,
                           beta=0.3,
                           gamma=0.5,
                           C=np.zeros(2),
                           G=np.ones(2),
                           d=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                           l=np.zeros((2, 2)),
                           m=np.arange(8).reshape((4, 2)),
                           c=np.array([[10, 20], [30, 40]]),
                           e=np.array([[100, 200], [300, 400]])),
            self.instance[[0, 1]]
        )

        self.assertEqual(
            EnergyInstance(n_ap=4,
                           n_facility=2,
                           time_slots=2,
                           alpha=0.2,
                           beta=0.3,
                           gamma=0.5,
                           C=np.zeros(2),
                           G=np.ones(2),
                           d=np.array([[8, 9, 10, 11], [4, 5, 6, 7]]),
                           l=np.zeros((2, 2)),
                           m=np.arange(8).reshape((4, 2)),
                           c=np.array([[50, 60], [30, 40]]),
                           e=np.array([[500, 600], [300, 400]])),
            self.instance[[2, 1]]
        )

        self.assertRaises(Exception, self.instance.__getitem__, [-1, 2])

        self.assertRaises(Exception, self.instance.__getitem__, [2, 3])

    def test_equals(self):
        other_instance = EnergyInstance(n_ap=4,
                                        n_facility=2,
                                        time_slots=3,
                                        alpha=1 / 3,
                                        beta=1 / 3,
                                        gamma=1 / 3,
                                        C=np.zeros(2),
                                        G=np.ones(2),
                                        d=np.arange(12).reshape((3, 4)),
                                        l=np.zeros((2, 2)),
                                        m=np.arange(8).reshape((4, 2)),
                                        c=np.arange(10, 61, 10).reshape((3, 2)),
                                        e=np.arange(100, 601, 100).reshape((3, 2)))

        self.assertFalse(self.instance == other_instance)

        self.assertFalse(self.instance == "string")

        self.assertTrue(self.instance != other_instance)
