from copy import copy
from unittest import TestCase

import numpy as np
import numpy.testing

from pycode.src.instance import BaseInstance


# noinspection PyStatementEffect,DuplicatedCode
class TestBaseInstance(TestCase):
    def setUp(self) -> None:
        self.instance = BaseInstance(n_ap=4,
                                     n_facility=2,
                                     time_slots=3,
                                     alpha=0.3,
                                     beta=0.7,
                                     C=np.zeros(2),
                                     d=np.arange(12).reshape((3, 4)),
                                     l=np.zeros((2, 2)),
                                     m=np.arange(8).reshape((4, 2)))

    def test_correct_init(self):
        self.assertEqual(4, self.instance.n_ap)

        self.assertEqual(2, self.instance.n_facility)

        self.assertEqual(3, self.instance.time_slots)

        self.assertEqual(0.3, self.instance.alpha)

        self.assertEqual(0.7, self.instance.beta)

        np.testing.assert_array_equal(np.zeros(2), self.instance.C)

        np.testing.assert_array_equal(np.arange(12).reshape((3, 4)), self.instance.d)

        np.testing.assert_array_equal(np.zeros((2, 2)), self.instance.l)

        np.testing.assert_array_equal(np.arange(8).reshape((4, 2)), self.instance.m)

        self.assertEqual(self.instance, copy(self.instance))

        self.assertEqual("Base Instance composed of 3 time slots with 4 access "
                         "points and 2 facilities",
                         str(self.instance))

    def test_wrong_init(self):
        # Wrong 'n_ap'.
        self.assertRaises(Exception, BaseInstance, 0, 2, 3, 0.3, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'n_facility'.
        self.assertRaises(Exception, BaseInstance, 4, -1, 3, 0.3, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'time_slots'.
        self.assertRaises(Exception, BaseInstance, 4, 2, -2, 0.3, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'alpha'
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, -0.5, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'beta'
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, 0.3, 1.1, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong sum 'alpha' and 'beta'
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, 0.5, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'C' shape.
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, 0.3, 0.7, np.zeros(1),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'd' shape.
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, 0.3, 0.7, np.zeros(2),
                          np.arange(15).reshape((3, 5)), np.zeros((2, 2)),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'l' shape.
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, 0.3, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros(4),
                          np.arange(8).reshape((4, 2)))

        # Wrong 'm' shape.
        self.assertRaises(Exception, BaseInstance, 4, 2, 3, 0.3, 0.7, np.zeros(2),
                          np.arange(12).reshape((3, 4)), np.zeros((2, 2)), np.arange(8))

    def test_shift(self):
        expected_s1 = BaseInstance(n_ap=4,
                                   n_facility=2,
                                   time_slots=3,
                                   alpha=0.3,
                                   beta=0.7,
                                   C=np.zeros(2),
                                   d=np.array([[4, 5, 6, 7], [8, 9, 10, 11],
                                               [0, 1, 2, 3]]),
                                   l=np.zeros((2, 2)),
                                   m=np.arange(8).reshape((4, 2)))

        self.assertEqual(expected_s1, self.instance.shift(1))

        expected_s2 = BaseInstance(n_ap=4,
                                   n_facility=2,
                                   time_slots=3,
                                   alpha=0.3,
                                   beta=0.7,
                                   C=np.zeros(2),
                                   d=np.array([[8, 9, 10, 11], [0, 1, 2, 3],
                                               [4, 5, 6, 7]]),
                                   l=np.zeros((2, 2)),
                                   m=np.arange(8).reshape((4, 2)))

        self.assertEqual(expected_s2, self.instance.shift(2))

        self.assertEqual(self.instance, self.instance.shift(0))

        self.assertRaises(Exception, self.instance.shift, -1)

        self.assertRaises(Exception, self.instance.shift, 3)

    def test_split(self):
        instance = BaseInstance(n_ap=4,
                                n_facility=2,
                                time_slots=5,
                                alpha=0.3,
                                beta=0.7,
                                C=np.zeros(2),
                                d=np.arange(20).reshape((5, 4)),
                                l=np.zeros((2, 2)),
                                m=np.arange(8).reshape((4, 2)))

        self.assertEqual(
            [BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=3,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(12).reshape((3, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=2,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(12, 20).reshape((2, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2)))],
            instance.split([2])
        )

        self.assertEqual(
            [BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(4).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(4, 8).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(8, 12).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=2,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(12, 20).reshape((2, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2)))],
            instance.split([0, 1, 2])
        )

        self.assertEqual(
            [BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(4).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(4, 8).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(8, 12).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(12, 16).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(16, 20).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2)))],
            instance.split([0, 1, 2, 3])
        )

        self.assertEqual(
            [BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(4).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(4, 8).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(8, 12).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(12, 16).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2))),
             BaseInstance(n_ap=4,
                          n_facility=2,
                          time_slots=1,
                          alpha=0.3,
                          beta=0.7,
                          C=np.zeros(2),
                          d=np.arange(16, 20).reshape((1, 4)),
                          l=np.zeros((2, 2)),
                          m=np.arange(8).reshape((4, 2)))],
            instance.split([0, 1, 2, 3, 4])
        )

        self.assertEqual([instance], instance.split([]))

        # Non strictly monotonic.
        self.assertRaises(Exception, instance.split, [1, 2, 2, 3])

        # Non growing monotonic.
        self.assertRaises(Exception, instance.split, [1, 2, 1, 3])

    def test_get_item(self):
        # Single time slot.

        self.assertEqual(
            BaseInstance(n_ap=4,
                         n_facility=2,
                         time_slots=1,
                         alpha=0.3,
                         beta=0.7,
                         C=np.zeros(2),
                         d=np.array([[4, 5, 6, 7]]),
                         l=np.zeros((2, 2)),
                         m=np.arange(8).reshape((4, 2))),
            self.instance[1]
        )

        self.assertRaises(Exception, self.instance.__getitem__, -1)

        self.assertRaises(Exception, self.instance.__getitem__, 3)

        # Slice.

        self.assertEqual(
            BaseInstance(n_ap=4,
                         n_facility=2,
                         time_slots=2,
                         alpha=0.3,
                         beta=0.7,
                         C=np.zeros(2),
                         d=np.array([[4, 5, 6, 7], [8, 9, 10, 11]]),
                         l=np.zeros((2, 2)),
                         m=np.arange(8).reshape((4, 2))),
            self.instance[1:3]
        )

        self.assertEqual(
            BaseInstance(n_ap=4,
                         n_facility=2,
                         time_slots=2,
                         alpha=0.3,
                         beta=0.7,
                         C=np.zeros(2),
                         d=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                         l=np.zeros((2, 2)),
                         m=np.arange(8).reshape((4, 2))),
            self.instance[0:-1]
        )

        self.assertEqual(self.instance[0:1], self.instance[0])

        self.assertEqual(self.instance[:100], self.instance)

        self.assertRaises(Exception, self.instance.__getitem__, slice(0, 1, 2))

        self.assertRaises(Exception, self.instance.__getitem__, slice(0, 0, 1))

        # List.

        self.assertEqual(self.instance[1], self.instance[[1]])

        self.assertEqual(
            BaseInstance(n_ap=4,
                         n_facility=2,
                         time_slots=2,
                         alpha=0.3,
                         beta=0.7,
                         C=np.zeros(2),
                         d=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                         l=np.zeros((2, 2)),
                         m=np.arange(8).reshape((4, 2))),
            self.instance[[0, 1]]
        )

        self.assertEqual(
            BaseInstance(n_ap=4,
                         n_facility=2,
                         time_slots=2,
                         alpha=0.3,
                         beta=0.7,
                         C=np.zeros(2),
                         d=np.array([[8, 9, 10, 11], [0, 1, 2, 3]]),
                         l=np.zeros((2, 2)),
                         m=np.arange(8).reshape((4, 2))),
            self.instance[[2, 0]]
        )

        self.assertEqual(
            BaseInstance(n_ap=4,
                         n_facility=2,
                         time_slots=3,
                         alpha=0.3,
                         beta=0.7,
                         C=np.zeros(2),
                         d=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
                         l=np.zeros((2, 2)),
                         m=np.arange(8).reshape((4, 2))),
            self.instance[[0, 1, 2]]
        )

        self.assertRaises(Exception, self.instance.__getitem__, [-1, 0, 1])

        self.assertRaises(Exception, self.instance.__getitem__, [0, 1, 3])

    def test_len(self):
        self.assertEqual(3, len(self.instance))

    def test_equals(self):
        other_instance = BaseInstance(n_ap=4,
                                      n_facility=2,
                                      time_slots=3,
                                      alpha=0.5,
                                      beta=0.5,
                                      C=np.zeros(2),
                                      d=np.arange(12).reshape((3, 4)),
                                      l=np.zeros((2, 2)),
                                      m=np.arange(8).reshape((4, 2)))

        self.assertTrue(self.instance != other_instance)

        self.assertFalse(self.instance == other_instance)

        self.assertFalse(self.instance == "string")
