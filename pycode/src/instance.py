"""Defines the `BaseInstance` and `EnergyInstance` classes."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt

from pycode.utility.checks import *


# ----------------------------------------------------------------------
# BaseInstance class

# noinspection PyPep8Naming,PyUnresolvedReferences
class BaseInstance:
    """Represents an instance without energy data."""

    # ----------------------------------------------------------------------
    # Constructor

    def __init__(
            self,
            n_ap: int,
            n_facility: int,
            time_slots: int,
            alpha: float,
            beta: float,
            C: np.ndarray[np.float64],
            d: np.ndarray[np.float64],
            l: np.ndarray[np.float64],
            m: np.ndarray[np.float64],
            check_alpha_beta_sum: bool = True,
            make_copy: bool = True
    ):
        """The parameter `check_alpha_beta_sum` indicates if the sum of `alpha` and
        `beta` must be 1, and `make_copy` whether to make a copy of the arrays passed
        as parameters."""
        # Check params.
        check_all_strictly_positive([n_ap, n_facility, time_slots],
                                    ["n_ap", "n_facility", "time_slots"])
        check_all_between([alpha, beta], 0.0, 1.0, ["alpha", "beta"])
        check_sum_equal([alpha, beta], 1.0, ["alpha", "beta"],
                        active=check_alpha_beta_sum)
        check_shape(C, (n_facility,), "C")
        check_shape(d, (time_slots, n_ap), "d")
        check_shape(l, (n_facility, n_facility), "l")
        check_shape(m, (n_ap, n_facility), "m")

        self._base_constructor(n_ap, n_facility, time_slots, alpha, beta, C, d, l, m,
                               make_copy=make_copy)

    def _base_constructor(
            self,
            n_ap: int,
            n_facility: int,
            time_slots: int,
            alpha: float,
            beta: float,
            C: np.ndarray[np.float64],
            d: np.ndarray[np.float64],
            l: np.ndarray[np.float64],
            m: np.ndarray[np.float64],
            make_copy: bool = True
    ) -> None:
        self._n_ap = n_ap
        self._n_facility = n_facility
        self._time_slots = time_slots
        self._alpha = alpha
        self._beta = beta
        self._C = C.copy() if make_copy else C
        self._d = d.copy() if make_copy else d
        self._l = l.copy() if make_copy else l
        self._m = m.copy() if make_copy else m

    # ----------------------------------------------------------------------
    # Properties

    @property
    def n_ap(self) -> int:
        """Number of access points in the instance."""
        return self._n_ap

    @property
    def n_facility(self) -> int:
        """Number of MEC facilities in the instance."""
        return self._n_facility

    @property
    def time_slots(self) -> int:
        """Number of time slots in the instance."""
        return self._time_slots

    @property
    def alpha(self) -> float:
        """Weight of the first part of the objective function."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Weight of the second part of the objective function."""
        return self._beta

    @property
    def C(self) -> np.ndarray[np.float64]:
        """Capacity of the facility."""
        return self._C.copy()

    @property
    def d(self) -> np.ndarray[np.float64]:
        """Energy required to meet the energy demand of access points for each
        time slot."""
        return self._d.copy()

    @property
    def l(self) -> np.ndarray[np.float64]:
        """Network distance between each facility pair."""
        return self._l.copy()

    @property
    def m(self) -> np.ndarray[np.float64]:
        """Physical distance between each facility-access point pair."""
        return self._m.copy()

    # ----------------------------------------------------------------------
    # Methods

    def extract_time_slots(self, time_slot: int | slice | Sequence[int]):
        """Returns a sub-instance composed of the indicated slot times."""
        if isinstance(time_slot, int):
            # Single time slot.
            check_inside(time_slot, range(self.time_slots), "time_slot")
            new_time_slots = 1
            new_d = self._d[time_slot].reshape((1, self.n_ap))

        elif isinstance(time_slot, slice):
            # Slicing.
            start, stop, step = time_slot.indices(len(self))
            if step != 1:
                raise Exception("The slicing step must be 1.")
            new_time_slots = stop - start
            new_d = self._d[start:stop]

        elif isinstance(time_slot, List) or isinstance(time_slot, np.ndarray):
            # List-like.
            check_all_inside(time_slot, range(self.time_slots), ["time_slot"])
            new_time_slots = len(time_slot)
            new_d = self.d[time_slot]

        else:
            raise AssertionError()

        return BaseInstance(n_ap=self.n_ap,
                            n_facility=self.n_facility,
                            time_slots=new_time_slots,
                            alpha=self.alpha,
                            beta=self.beta,
                            C=self._C,
                            d=new_d,
                            l=self._l,
                            m=self._m,
                            check_alpha_beta_sum=False)

    def shift(self, n_time_slots: int) -> BaseInstance | EnergyInstance:
        """Returns an instance equal to this in which a shift of `n_time_slots`
        time slots to the left is made.

        It basically consists of taking the first `n_time_slots` time slots and
        placing them at the end of the instance."""
        check_inside(n_time_slots, range(self.time_slots), "n_time_slots")

        return BaseInstance(n_ap=self.n_ap,
                            n_facility=self.n_facility,
                            time_slots=self.time_slots,
                            alpha=self.alpha,
                            beta=self.beta,
                            C=self._C,
                            d=np.concatenate((self._d[n_time_slots:],
                                              self._d[:n_time_slots])),
                            l=self._l,
                            m=self._m,
                            check_alpha_beta_sum=False,
                            make_copy=False)

    def split(self, split_points: Sequence[int]) -> List[BaseInstance | EnergyInstance]:
        """Returns a list containing the sub-instances generated by dividing
        the current one using the split points passed as a parameter (the split
        points are included in the previous sub-instance).

        If the list of split points is empty, a list containing the complete
        instance is returned.
        """
        # Check that the split points are incremental and correct.
        is_strictly_monotonic = np.all(np.diff(split_points) > 0)
        if len(split_points) == 0:
            first_split_valid = True
            last_split_valid = True
        else:
            first_split_valid = split_points[0] in range(self.time_slots)
            last_split_valid = split_points[-1] in range(self.time_slots)
        if not (is_strictly_monotonic and first_split_valid and last_split_valid):
            raise Exception("The `split_points` array must be strictly increasing and "
                            "contain valid time slots.")

        return self._split(split_points)

    def _split(self, split_points: Sequence[int]) -> List[BaseInstance | EnergyInstance]:
        """Same as `split` but without parameter check."""
        if len(split_points) == 0:
            return [self.__copy__()]
        else:
            sub_instances = []
            start = 0
            for split in split_points:
                end = split + 1
                sub_instances += [self[start:end]]
                start = end
            if start < self.time_slots:
                sub_instances += [self[start:]]
            return sub_instances

    # ----------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of time slots in the instance."""
        return self.time_slots

    def __getitem__(self, item: int | slice | List[int]) -> BaseInstance:
        """Returns a sub-instance composed of the indicated slot times."""
        return self.extract_time_slots(item)

    def __copy__(self) -> BaseInstance:
        # NB: getters create a copy of the arrays.
        return BaseInstance(n_ap=self.n_ap,
                            n_facility=self.n_facility,
                            time_slots=self.time_slots,
                            alpha=self.alpha,
                            beta=self.beta,
                            C=self.C,
                            d=self.d,
                            l=self.l,
                            m=self.m,
                            check_alpha_beta_sum=False,
                            make_copy=False)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseInstance):
            return self.n_ap == other.n_ap \
                and self.n_facility == other.n_facility \
                and self.time_slots == other.time_slots \
                and self.alpha == other.alpha \
                and self.beta == other.beta \
                and np.array_equal(self._C, other._C) \
                and np.array_equal(self._d, other._d) \
                and np.array_equal(self._l, other._l) \
                and np.array_equal(self._m, other._m)
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return "Base Instance composed of {} time slots with {} access points and {} " \
               "facilities".format(self.time_slots, self.n_ap, self.n_facility)

    def __repr__(self) -> str:
        return "{}\n\n" \
               "alpha: {}\n" \
               "beta: {}\n\n" \
               "C.\n{}\n\n" \
               "d.\n{}\n\n" \
               "l.\n{}\n\n" \
               "m.\n{}\n\n".format(str(self), self.alpha, self.beta, self._C,
                                   self._d, self._l, self._m)


# ----------------------------------------------------------------------
# EnergyInstance class

# noinspection PyPep8Naming,PyArgumentList
class EnergyInstance(BaseInstance):
    """Represents an instance with energy data."""

    # ----------------------------------------------------------------------
    # Static creators

    @classmethod
    def from_base_instance(
            cls,
            base_instance: BaseInstance,
            alpha: float,
            beta: float,
            gamma: float,
            G: np.ndarray[np.float64],
            c: np.ndarray[np.float64],
            e: np.ndarray[np.float64],
            p: np.ndarray[np.float64] = None,
            check_alpha_beta_gamma_sum: bool = True,
            make_copy: bool = True
    ) -> EnergyInstance:
        """Creates an `EnergyInstance` from the `base_instance` and the energy
        data passed as a parameter.

        The parameter `check_alpha_beta_gamma_sum` indicates if the sum of `alpha`,
        `beta` and `gamma` must be 1.
        """
        return cls(n_ap=base_instance.n_ap,
                   n_facility=base_instance.n_facility,
                   time_slots=base_instance.time_slots,
                   alpha=alpha,
                   beta=beta,
                   gamma=gamma,
                   C=base_instance._C,
                   G=G,
                   d=base_instance._d,
                   l=base_instance._l,
                   m=base_instance._m,
                   c=c,
                   e=e,
                   p=p,
                   check_alpha_beta_gamma_sum=check_alpha_beta_gamma_sum,
                   make_copy=make_copy)

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
            self,
            n_ap: int,
            n_facility: int,
            time_slots: int,
            alpha: float,
            beta: float,
            gamma: float,
            C: np.ndarray[np.float64],
            G: np.ndarray[np.float64],
            d: np.ndarray[np.float64],
            l: np.ndarray[np.float64],
            m: np.ndarray[np.float64],
            c: np.ndarray[np.float64],
            e: np.ndarray[np.float64],
            p: np.ndarray[np.float64] = None,
            check_alpha_beta_gamma_sum: bool = True,
            make_copy: bool = True
    ):
        """The parameter `check_alpha_beta_gamma_sum` indicates if the sum of `alpha`,
        `beta` and `gamma` must be 1."""
        super().__init__(n_ap=n_ap,
                         n_facility=n_facility,
                         time_slots=time_slots,
                         alpha=alpha,
                         beta=beta,
                         C=C,
                         d=d,
                         l=l,
                         m=m,
                         check_alpha_beta_sum=False,
                         make_copy=make_copy)

        # Check params.
        check_between(gamma, 0.0, 1.0, "gamma")
        check_sum_equal([alpha, beta, gamma], 1.0, ["alpha", "beta", "gamma"],
                        active=check_alpha_beta_gamma_sum)
        check_shape(G, (self.n_facility,), "G")
        check_shape(c, (self.time_slots, self.n_facility), "c")
        check_shape(e, (self.time_slots, self.n_facility), "e")
        if p is not None:
            check_shape(p, (self.n_facility,), "p")

        self._energy_constructor(gamma, G, c, e, p, make_copy=make_copy)

    def _energy_constructor(
            self,
            gamma: float,
            G: np.ndarray[np.float64],
            c: np.ndarray[np.float64],
            e: np.ndarray[np.float64],
            p: np.ndarray[np.float64] = None,
            make_copy: bool = True
    ) -> None:
        self._gamma = gamma
        self._G = G.copy() if make_copy else G
        self._c = c.copy() if make_copy else c
        self._e = e.copy() if make_copy else e
        self._set_p(p if p is not None else np.zeros(self.n_facility),
                    make_copy=make_copy)

    # ----------------------------------------------------------------------
    # Properties

    @property
    def gamma(self) -> float:
        """Weight of the third part of the objective function."""
        return self._gamma

    @property
    def G(self) -> np.ndarray[np.float64]:
        """Facility battery capacity."""
        return self._G.copy()

    @property
    def c(self) -> np.ndarray[np.float64]:
        """Cost of purchasing power for each facility in each time slot."""
        return self._c.copy()

    @property
    def e(self) -> np.ndarray[np.float64]:
        """Energy produced by each facility per time slot."""
        return self._e.copy()

    @property
    def p(self) -> np.ndarray[np.float64]:
        """Energy present in the facility battery in the first time slot."""
        return self._p.copy()

    @p.setter
    def p(self, value: np.ndarray[np.float64]):
        """Energy present in the facility battery in the first time slot."""
        self._set_p(value)

    # ----------------------------------------------------------------------
    # Private methods

    def _set_p(self, value: np.ndarray[np.float64], make_copy: bool = True) -> None:
        """Set the `p` value checking the `value` parameter."""
        check_shape(value, (self._n_facility,), "p")
        if (value <= self._G).all():
            self._p = value.copy() if make_copy else value
        else:
            raise Exception("The value of 'p' must be not greater than relative "
                            "facility battery capacity.")

    # ----------------------------------------------------------------------
    # Methods

    def extract_time_slots(self, time_slot: int | slice | Sequence[int]):
        """Returns a sub-instance composed of the indicated slot times.

        In each sub-instance the `p` value is that of the current instance.
        """
        if isinstance(time_slot, int):
            # Single time slot.
            new_c = self._c[time_slot].reshape((1, self.n_facility))
            new_e = self._e[time_slot].reshape((1, self.n_facility))

        elif isinstance(time_slot, slice):
            # Slicing.
            start, stop, step = time_slot.indices(len(self))
            if step != 1:
                raise Exception("The slicing step must be 1.")
            new_c = self._c[start:stop]
            new_e = self._e[start:stop]

        elif isinstance(time_slot, List) or isinstance(time_slot, np.ndarray):
            # List-like.
            check_all_inside(time_slot, range(self.time_slots), ["time_slot"])
            new_c = self.c[time_slot]
            new_e = self.e[time_slot]

        else:
            raise AssertionError()

        return EnergyInstance.from_base_instance(
            base_instance=super(EnergyInstance, self).extract_time_slots(time_slot),
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            G=self.G,
            c=new_c,
            e=new_e,
            p=self.p,
            check_alpha_beta_gamma_sum=False
        )

    def shift(self, n_time_slots: int) -> BaseInstance | EnergyInstance:
        return EnergyInstance.from_base_instance(
            base_instance=super(EnergyInstance, self).shift(n_time_slots),
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            G=self._G,
            c=np.concatenate((self._c[n_time_slots:], self._c[:n_time_slots])),
            e=np.concatenate((self._e[n_time_slots:], self._e[:n_time_slots])),
            p=self._p,
            check_alpha_beta_gamma_sum=False,
            make_copy=False
        )

    # ----------------------------------------------------------------------

    def __getitem__(self, item: int | slice | Sequence[int]) -> EnergyInstance:
        return self.extract_time_slots(item)

    def __copy__(self) -> EnergyInstance:
        # NB: getters create a copy of the arrays.
        return EnergyInstance(
            n_ap=self.n_ap,
            n_facility=self.n_facility,
            time_slots=self.time_slots,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            C=self.C,
            G=self.G,
            d=self.d,
            l=self.l,
            m=self.m,
            c=self.c,
            e=self.e,
            p=self.p,
            check_alpha_beta_gamma_sum=False)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EnergyInstance):
            return super(EnergyInstance, self).__eq__(other) \
                and self.gamma == other.gamma \
                and np.array_equal(self._G, other._G) \
                and np.array_equal(self._c, other._c) \
                and np.array_equal(self._e, other._e) \
                and np.array_equal(self._p, other._p)
        else:
            return False

    def __str__(self) -> str:
        return "Energy Instance composed of {} time slots with {} access points " \
               "and {} facilities" \
            .format(self.time_slots, self.n_ap, self.n_facility)

    def __repr__(self) -> str:
        return "{}\n\n" \
               "alpha: {}\n" \
               "beta: {}\n" \
               "gamma: {}\n\n" \
               "C.\n{}\n\n" \
               "G.\n{}\n\n" \
               "d.\n{}\n\n" \
               "l.\n{}\n\n" \
               "m.\n{}\n\n" \
               "c.\n{}\n\n" \
               "e.\n{}\n\n" \
               "p.\n{}\n\n".format(str(self), self.alpha, self.beta, self.gamma,
                                   self._C, self._G, self._d, self._l, self._m,
                                   self._c, self._e, self._p)
