"""Defines the `EmptyResult` and `Result` classes."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import List, Optional

import mip
import pandas as pd

from pycode.utility.checks import *


# ----------------------------------------------------------------------
# From dataframe to variables methods

def _from_conn_str_to_list(conn: str) -> list:
    if conn == "":
        return []
    else:
        conn_split = conn.split(" ")
        conn_map = map(lambda value: int(value), conn_split)
        return list(conn_map)


def _from_dataframe_to_x(
        df: pd.DataFrame,
        n_ap: int,
        n_facility: int,
        time_slots: int
) -> np.ndarray[np.float64]:
    x_array = np.zeros((time_slots, n_ap, n_facility))
    for t in range(time_slots):
        for k in range(n_facility):
            ap_connected = _from_conn_str_to_list(df.loc[t, k]["conn"])
            for i in ap_connected:
                x_array[t, i, k] = 1
    return x_array


def _from_dataframe_to_g(
        df: pd.DataFrame,
        n_facility: int,
        time_slots: int,
        make_copy: bool = False
) -> np.ndarray[np.float64]:
    g_array = df["g"].to_numpy(copy=make_copy)
    return g_array.reshape((time_slots, n_facility))


def _from_dataframe_to_v(
        df: pd.DataFrame,
        n_facility: int,
        time_slots: int,
        make_copy: bool = False
) -> np.ndarray[np.float64]:
    v_array = df["v"].to_numpy(copy=make_copy)
    return v_array.reshape((time_slots, n_facility))


def _from_dataframe_to_z(
        df: pd.DataFrame,
        n_facility: int,
        time_slots: int,
        make_copy: bool = False
) -> np.ndarray[np.float64]:
    z_array = df["z"].to_numpy(copy=make_copy)
    return z_array.reshape((time_slots, n_facility))


# ----------------------------------------------------------------------
# EmptyResult class

class EmptyResult:
    """Represents the empty result of a computation."""

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, n_ap: int, n_facility: int):
        # Check params.
        check_strictly_positive(n_ap, "n_ap")
        check_strictly_positive(n_facility, "n_facility")

        self._empty_constructor(n_ap, n_facility)

    def _empty_constructor(self, n_ap: int, n_facility: int) -> None:
        self._n_ap = n_ap
        self._n_facility = n_facility

    # ----------------------------------------------------------------------
    # Properties

    @property
    def n_ap(self) -> int:
        """Number of access points present in the instance of which it is
        the result."""
        return self._n_ap

    @property
    def n_facility(self) -> int:
        """Number of facilities present in the instance of which it is the
        result."""
        return self._n_facility

    @property
    def time_slots(self) -> int:
        """Number of time slots present in the instance of which it is the
        result."""
        return 0

    # ----------------------------------------------------------------------
    # Methods

    def append(self, other: Result) -> Result:
        """Returns a result representing the concatenation of this and `other`,
        in which `other` time slots come at the end of these."""
        # Return the result passed as an argument.
        if isinstance(other, Result):
            # Check params.
            if self.n_ap != other.n_ap:
                raise Exception("The result to add is incompatible: different "
                                "number of aps.")
            elif self.n_facility != other.n_facility:
                raise Exception("The result to add is incompatible: different "
                                "number of facilities.")

            return other.__copy__()

        else:
            raise Exception("Is not possible append {} to EmptyResult."
                            .format(type(other)))

    def save_to_file(
            self,
            filepath: str | Path,
            overwrite_if_exists: bool = False,
            create_path_if_not_exist: bool = True
    ) -> None:
        raise Exception("Is not possible save an empty result.")

    # ----------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of time slots in the instance of which it is
        the result."""
        return 0

    def __getitem__(self, item: int) -> Optional[Result]:
        raise Exception("Is not possible get time slots from an empty result.")

    def __add__(self, other: Result) -> Result:
        """Append `other` at this result."""
        return self.append(other)

    def __copy__(self) -> EmptyResult | Result:
        return EmptyResult(self.n_ap, self.n_facility)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EmptyResult):
            return self.n_ap == other.n_ap \
                and self.n_facility == other.n_facility \
                and self.time_slots == other.time_slots
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return "Result of an instance composed of {} time slots with {} access points " \
               "and {} facilities" \
            .format(self.time_slots, self.n_ap, self.n_facility)

    def __repr__(self) -> str:
        return self.__str__()


# ----------------------------------------------------------------------
# Result class

class Result(EmptyResult):
    """Represents the result of a computation."""

    # ----------------------------------------------------------------------
    # Static creators

    @classmethod
    def from_file(cls, filepath: str | Path) -> Result:
        """Loads a result from the file to the `filepath` and returns it."""
        path = Path(filepath)
        df = pd.read_csv(path, index_col=(0, 1))
        df.fillna("", inplace=True)
        return Result.from_pd_dataframe(df)

    @classmethod
    def from_pd_dataframe(cls, df: pd.DataFrame, make_copy: bool = True) -> Result:
        """Create a result from the given dataframe and return it.

        The `df` dataframe must have a multi-index formed by the pair
        ``time slot-facility`` and the columns ``g``, ``v``, ``z``, and ``conn``."""
        # Extracts the number of facilities and the number of time slots from
        # dataframe index.
        n_facility = len(df.index.unique("k"))
        time_slots = len(df.index.unique("t"))
        # Counts the number of ap in the 'conn' column of the first time slot.
        # It supposes that the dataframe is well-formed and all time slots have
        # the same number of distinct ap without duplications.
        n_ap = 0
        for k in range(n_facility):
            ap_connected = df.loc[0, k]["conn"].split(" ")
            # If 'ap_connected' is [""] means that the 'conn' string is empty.
            if ap_connected != [""]:
                n_ap += len(ap_connected)

        return cls(x=_from_dataframe_to_x(df, n_ap, n_facility, time_slots),
                   g=_from_dataframe_to_g(df, n_facility, time_slots),
                   v=_from_dataframe_to_v(df, n_facility, time_slots),
                   z=_from_dataframe_to_z(df, n_facility, time_slots),
                   make_copy=make_copy)

    @classmethod
    def from_variables(
            cls,
            x: np.ndarray[mip.Var],
            g: np.ndarray[mip.Var],
            v: np.ndarray[mip.Var],
            z: np.ndarray[mip.Var],
            g_ub: np.ndarray[np.float64] = None,
            make_copy: bool = True
    ) -> Result:
        """Creates a result from the variables being output from the model and
        return it."""
        # Function that extract variables value keeping same array shape.
        extract_var_value = np.vectorize(lambda var: var.x)
        return cls(x=extract_var_value(x.copy()),
                   g=extract_var_value(g.copy()),
                   v=extract_var_value(v.copy()),
                   z=extract_var_value(z.copy()),
                   g_ub=g_ub,
                   make_copy=make_copy)

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
            self,
            x: np.ndarray[np.float64],
            g: np.ndarray[np.float64],
            v: np.ndarray[np.float64],
            z: np.ndarray[np.float64],
            g_ub: np.ndarray[np.float64] = None,
            make_copy: bool = True
    ):
        """The `g_ub` parameter is the upper bound of the `g` values.

        The actual value of g is given by the minimum of `g` and `g_ub`
        calculated for each array element.
        """
        # The number of ap, facility and time slots are extracted from shape of the
        # variables.
        super().__init__(n_ap=x.shape[1], n_facility=x.shape[2])

        # Check if input arrays has compatibles shapes.
        if not (x.shape[2] == g.shape[1] == v.shape[1] == z.shape[1]):
            raise Exception("Parameters 'x', 'g', 'v' and 'z' have incompatible shapes: "
                            "different number of facility.")
        elif not (x.shape[0] == g.shape[0] == v.shape[0] == z.shape[0]):
            raise Exception("Parameters 'x', 'g', 'v' and 'z' have incompatible shapes: "
                            "different number of time slots.")
        elif g_ub is not None and g.shape[1] != g_ub.shape[0]:
            raise Exception("Parameters 'g' and 'g_ub' have incompatible shapes.")

        self._constructor(x, g, v, z, g_ub, make_copy=make_copy)

    def _constructor(
            self,
            x: np.ndarray[np.float64],
            g: np.ndarray[np.float64],
            v: np.ndarray[np.float64],
            z: np.ndarray[np.float64],
            g_ub: np.ndarray[np.float64] = None,
            make_copy: bool = True
    ) -> None:
        self._time_slots = x.shape[0]
        if g_ub is not None:
            g = np.minimum(g, g_ub)
        # The '_result' attribute is a dataframe representing this result.
        self._result = self._from_values_to_pd_dataframe(x=x.copy() if make_copy else x,
                                                         g=g.copy() if make_copy else g,
                                                         v=v.copy() if make_copy else v,
                                                         z=z.copy() if make_copy else z)

    # ----------------------------------------------------------------------
    # Properties

    @property
    def time_slots(self) -> int:
        """Number of time slots present in the instance of which it is
        the result."""
        return self._time_slots

    @functools.cached_property
    def x(self) -> np.ndarray[np.float64]:
        """The value of x variables.

        These are binary variables that indicate for each time slot which
        facility each access point is assigned to.
        """
        return _from_dataframe_to_x(self._result, self._n_ap, self._n_facility,
                                    self._time_slots)

    @functools.cached_property
    def g(self) -> np.ndarray[np.float64]:
        """The value of g variables.

        They contain for each facility and each slot time the remaining energy.
        """
        return _from_dataframe_to_g(self._result, self._n_facility, self._time_slots,
                                    make_copy=True)

    @functools.cached_property
    def v(self) -> np.ndarray[np.float64]:
        """The value of v variables.

        They contain the energy used by each facility in each time slot.
        """
        return _from_dataframe_to_v(self._result, self._n_facility, self._time_slots,
                                    make_copy=True)

    @functools.cached_property
    def z(self) -> np.ndarray[np.float64]:
        """The value of z variables.

        They contain the energy purchased by each facility in each time slot.
        """
        return _from_dataframe_to_z(self._result, self._n_facility, self._time_slots,
                                    make_copy=True)

    # ----------------------------------------------------------------------
    # Private methods

    def _from_values_to_pd_dataframe(
            self,
            x: np.ndarray[np.float64],
            g: np.ndarray[np.float64],
            v: np.ndarray[np.float64],
            z: np.ndarray[np.float64]
    ) -> pd.DataFrame:
        """Creates the respective dataframe from the value of the variables."""
        df = pd.DataFrame(columns=['t', 'k', 'g', 'v', 'z', 'conn'])
        for t, k in itertools.product(range(self._time_slots), range(self._n_facility)):
            # The variable 'conn' contains all aps connected to facility 'k' at time 't'.
            # It is filled checking all values of variable 'x' at time 't' for facility
            # 'k': if value is 1 (> 0.9 because is floating point) so the ap must be
            # added.
            conn = []
            for i in range(self._n_ap):
                if x[t, i, k] > 0.9:
                    conn.append(i)
            new_index = len(df.index)
            conn_text = " ".join(map(lambda value: str(value), conn))
            df.loc[new_index] = (t, k, g[t, k], v[t, k], z[t, k], conn_text)
        df.set_index(["t", "k"], inplace=True)
        return df

    # ----------------------------------------------------------------------
    # Methods

    def get_all_aps_connected(self, time_slot: int, facility: int) -> List[int]:
        """Returns all access points connected to the facility `facility` at
        `time_slot` time."""
        # Check params.
        check_inside(time_slot, range(self.time_slots), "time_slots")
        check_inside(time_slot, range(self.n_facility), "n_facility")

        return _from_conn_str_to_list(self._result.loc[time_slot, facility]["conn"])

    def get_facility_to_witch_is_connected(self, time_slot: int, ap: int) -> int:
        """Returns the facility to which the access point `ap` is connected at
        time `time_slot`."""
        # Check params.
        check_inside(time_slot, range(self.time_slots), "time_slots")
        check_inside(ap, range(self.n_ap), "ap")

        x = self.x[time_slot, ap]
        return np.where(x == 1)[0][0]

    def extract_time_slot(self, time_slot: int) -> Result:
        """Returns a sub-result composed of the indicated time slot."""
        # Check params.
        check_inside(time_slot, range(self.time_slots), "time_slot")

        return Result(x=np.array([self.x[time_slot]]),
                      g=np.array([self.g[time_slot]]),
                      v=np.array([self.v[time_slot]]),
                      z=np.array([self.z[time_slot]]),
                      make_copy=False)

    # noinspection GrazieInspection
    def append(self, other: Result) -> Result:
        # Is possible append only another Result.
        # The new result is the concatenation of this and 'other' in which the
        # first slot time of 'other' is the next of the last of this.
        if isinstance(other, Result):
            # Check if 'other' result has compatible shape.
            if self._n_ap != other._n_ap:
                raise Exception("The result to add is incompatible: different number "
                                "of aps.")
            elif self._n_facility != other._n_facility:
                raise Exception("The result to add is incompatible: different number of "
                                "facilities.")

            # Shifts the 'other's time slots after the last of this.
            # The facilities index remain the same.
            first_index = self._time_slots
            indexes_range = range(first_index, first_index + other._time_slots)
            indexes_changes = {old: new for old, new in enumerate(indexes_range)}
            other_df = other._result.rename(indexes_changes, level=0)

            new_df = pd.concat([self._result, other_df])
            return Result.from_pd_dataframe(new_df, make_copy=False)
        else:
            raise Exception("Is not possible append {} to Result.".format(type(other)))

    # ----------------------------------------------------------------------
    # Export methods

    def to_pd_dataframe(self) -> pd.DataFrame:
        """Returns a dataframe representing the result."""
        return self._result.copy(deep=True)

    def save_to_file(
            self,
            filepath: str | Path,
            overwrite_if_exists: bool = False,
            create_path_if_not_exist: bool = True
    ) -> None:
        """Save the result to the file at `filepath`.

        The `overwrite_if_exists` parameter indicates whether to overwrite the
        file if it already exists while `create_path_if_not_exist` indicates
        whether to create the folders in the `filepath` if they do not exist.
        """

        # noinspection PyShadowingNames
        def save(filepath: str | Path) -> None:
            self._result.to_csv(filepath, index=True)

        path = Path(filepath)
        if path.exists():
            if path.is_file():
                if overwrite_if_exists:
                    save(path)
                else:
                    raise Exception("The file {} already exists. Set "
                                    "overwrite_if_exists=True for overwrite it."
                                    .format(path))
            else:
                raise Exception("The path {} indicates a folder.".format(path))
        else:
            if create_path_if_not_exist:
                os.makedirs(path.parent, exist_ok=True)
            save(path)

    # ----------------------------------------------------------------------

    def __len__(self) -> int:
        return self.time_slots

    def __getitem__(self, item: int) -> Result:
        return self.extract_time_slot(item)

    def __copy__(self) -> EmptyResult | Result:
        return Result(self.x, self.g, self.v, self.z, make_copy=False)

    def __add__(self, other: Result | EmptyResult) -> Result:
        return self.append(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Result):
            return super(Result, self).__eq__(other) and self._result.equals(
                other._result)
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
