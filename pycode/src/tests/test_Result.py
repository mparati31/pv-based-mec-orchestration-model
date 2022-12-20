import filecmp
from unittest import TestCase

import numpy.testing

from pycode.src.result import *

resources = Path(__file__).parent / "resources"


# noinspection DuplicatedCode
class TestResult(TestCase):
    def setUp(self) -> None:
        self.n_ap = 4
        self.n_facility = 2
        self.time_slots = 3

        self.df = pd.DataFrame.from_dict(
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
        self.df.set_index(["t", "k"], inplace=True)

        self.x = np.array([
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0]
            ],
            [
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1]
            ],
            [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0]
            ]
        ])

        self.g = np.array([
            [10., 11],
            [12, 13],
            [14, 15]
        ])

        self.v = np.array([
            [20., 21],
            [22, 23],
            [24, 25]
        ])

        self.z = np.array([
            [30., 31],
            [32, 33],
            [34, 35]
        ])

        self.empty_result = EmptyResult(4, 2)

        self.result = Result(self.x, self.g, self.v, self.z)

    def test_from_file(self):
        result_file = Result.from_file(resources / "result_test_1.csv")

        self.assertEqual(self.result, result_file)

    def test_from_pd_dataframe(self):
        self.assertEqual(self.result, Result.from_pd_dataframe(self.df))

    def test_from_variables(self):
        model = mip.Model()
        model.verbose = False
        x_vars = np.array([
            [[model.add_var(lb=self.x[t, i, k], ub=self.x[t, i, k], var_type=mip.INTEGER)
              for k in range(self.n_facility)]
             for i in range(self.n_ap)]
            for t in range(self.time_slots)
        ])
        g_vars = np.array([
            [model.add_var(lb=self.g[t, k], ub=self.g[t, k], var_type=mip.CONTINUOUS)
             for k in range(self.n_facility)]
            for t in range(self.time_slots)
        ])
        v_vars = np.array([
            [model.add_var(lb=self.v[t, k], ub=self.v[t, k], var_type=mip.CONTINUOUS)
             for k in range(self.n_facility)]
            for t in range(self.time_slots)
        ])
        z_vars = np.array([
            [model.add_var(lb=self.z[t, k], ub=self.z[t, k], var_type=mip.CONTINUOUS)
             for k in range(self.n_facility)]
            for t in range(self.time_slots)
        ])
        model.optimize()

        self.assertEqual(
            self.result,
            Result.from_variables(x_vars, g_vars, v_vars, z_vars)
        )

    def test_correct_init(self):
        self.assertEqual(4, self.n_ap)

        self.assertEqual(2, self.n_facility)

        self.assertEqual(3, self.time_slots)

        np.testing.assert_array_equal(self.x, self.result.x)

        np.testing.assert_array_equal(self.g, self.result.g)

        np.testing.assert_array_equal(self.v, self.result.v)

        np.testing.assert_array_equal(self.z, self.result.z)

        self.assertEqual(3, len(self.result))

    def test_init_with_g_ub(self):
        result = Result(self.x, self.g, self.v, self.z, g_ub=np.array([12, 11]))

        expected = np.array([
            [10., 11],
            [12, 11],
            [12, 11]
        ])

        np.testing.assert_array_equal(expected, result.g)

        self.assertRaises(Exception, Result, self.x, self.g, self.v, self.z,
                          np.array([10, 12, 11]))

    def test_wrong_init(self):
        with self.assertRaisesRegex(Exception,
                                    "Parameters 'x', 'g', 'v' and 'z' have incompatible "
                                    "shapes: different number of facility."):
            wrong_g = np.array([
                [10, 11, 0],
                [12, 13, 1],
                [14, 15, 2]
            ])
            Result(self.x, wrong_g, self.v, self.z)

        with self.assertRaisesRegex(Exception,
                                    "Parameters 'x', 'g', 'v' and 'z' have incompatible "
                                    "shapes: different number of time slots."):
            wrong_v = np.array([
                [20, 21],
                [22, 23],
                [24, 25],
                [0, 1]
            ])
            Result(self.x, self.g, wrong_v, self.z)

    def test_get_all_aps_connected(self):
        self.assertEqual([], self.result.get_all_aps_connected(0, 1))

        self.assertEqual([0, 1, 3], self.result.get_all_aps_connected(1, 1))

        self.assertRaises(Exception, self.result.get_all_aps_connected, 4, 1)

        self.assertRaises(Exception, self.result.get_all_aps_connected, 1, 3)

    def test_get_facility_to_witch_is_connected(self):
        self.assertEqual(0, self.result.get_facility_to_witch_is_connected(0, 3))

        self.assertRaises(Exception, self.result.get_facility_to_witch_is_connected, 4, 1)

        self.assertRaises(Exception, self.result.get_facility_to_witch_is_connected, 1, 6)

    def test_extract_time_slot(self):
        self.assertEqual(self.result[1], self.result.extract_time_slot(1))

        expected = Result(x=np.array([[[1, 0], [1, 0], [1, 0], [1, 0]]]),
                          g=np.array([[10., 11]]),
                          v=np.array([[20., 21]]),
                          z=np.array([[30., 31]]))

        self.assertEqual(expected, self.result[0])

    # noinspection DuplicatedCode
    def test_append_single_slot_instance(self):
        x_single_slot = np.array([[
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ]])
        g_single_slot = np.array([[16., 17]])
        v_single_slot = np.array([[26., 27]])
        z_single_slot = np.array([[36., 37]])

        expected = Result(
            np.concatenate([self.x, x_single_slot]),
            np.concatenate([self.g, g_single_slot]),
            np.concatenate([self.v, v_single_slot]),
            np.concatenate([self.z, z_single_slot])
        )

        single_slot_result = Result(x_single_slot, g_single_slot, v_single_slot,
                                    z_single_slot)

        self.assertEqual(self.result + single_slot_result,
                         self.result.append(single_slot_result))

        self.assertEqual(expected, self.result.append(single_slot_result))

    # noinspection DuplicatedCode
    def test_append_multi_slot_instance(self):
        x_multi_slot = np.array([
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1]
            ],
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1]
            ]
        ])
        g_multi_slot = np.array([[16., 17], [18, 19]])
        v_multi_slot = np.array([[26., 27], [28, 29]])
        z_multi_slot = np.array([[36., 37], [38, 39]])

        expected = Result(x=np.concatenate([self.x, x_multi_slot]),
                          g=np.concatenate([self.g, g_multi_slot]),
                          v=np.concatenate([self.v, v_multi_slot]),
                          z=np.concatenate([self.z, z_multi_slot]))

        multi_slot_result = Result(x_multi_slot, g_multi_slot, v_multi_slot, z_multi_slot)

        self.assertEqual(self.result + multi_slot_result,
                         self.result.append(multi_slot_result))

        self.assertEqual(expected, self.result.append(multi_slot_result))

    def test_wrong_append(self):
        self.assertRaises(Exception, self.result.append, Result(x=np.zeros((1, 1, 2)),
                                                                g=np.zeros((1, 2)),
                                                                v=np.zeros((1, 2)),
                                                                z=np.zeros((1, 2))))

        self.assertRaises(Exception, self.result.append, Result(x=np.zeros((1, 4, 1)),
                                                                g=np.zeros((1, 1)),
                                                                v=np.zeros((1, 1)),
                                                                z=np.zeros((1, 1))))

        self.assertRaises(Exception, self.result.append, "string")

    def test_to_pd_dataframe(self):
        pd.testing.assert_frame_equal(self.df, self.result.to_pd_dataframe())

    def test_save_to_file(self):
        filepath_expected = resources / "result_test_1.csv"
        filepath_generated = resources / "result_test_2.csv"

        if filepath_generated.exists():
            os.remove(filepath_generated)
        self.result.save_to_file(filepath_generated)

        self.assertTrue(filecmp.cmp(filepath_expected, filepath_generated))

        with self.assertRaisesRegex(Exception,
                                    "The path {} indicates a folder".format(resources)):
            self.result.save_to_file(resources)

        with self.assertRaisesRegex(Exception,
                                    "The file {} already exists. Set "
                                    "overwrite_if_exists=True for overwrite "
                                    "it.".format(filepath_generated)):
            self.result.save_to_file(filepath_generated)

        self.result.save_to_file(filepath_generated, overwrite_if_exists=True)

        self.assertTrue(filecmp.cmp(filepath_expected, filepath_generated))

        os.remove(filepath_generated)

    def test_equals(self):
        self.assertTrue(Result.from_pd_dataframe(self.df) ==
                        Result.from_pd_dataframe(self.df))

        self.assertFalse(Result.from_pd_dataframe(self.df) ==
                         Result.from_pd_dataframe(self.df.loc[:1]))

        self.assertTrue(self.result == self.result)

        self.assertTrue(Result.from_pd_dataframe(self.df) !=
                        Result.from_pd_dataframe(self.df.loc[:1]))

        self.assertFalse(self.result == "string")
