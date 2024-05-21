# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of set-valued domains for distributions."""

__author__ = ["VascoSch92"]

from typing import List, Tuple, Union

from skpro.base import BaseObject

__all__ = ["Interval", "Finite", "Product"]


class Domain(BaseObject):
    """Base class for domains."""


class Interval(Domain):
    r"""Interval of the real line.

    The class implements intervals of the real line, i.e., sets of the form
    :math:`(a, b)`, where `a, b \in \mathbb{R} \cup \{-\inf, +\inf \}`.

    Parameters
    ----------
    values: list of floats or ints of length 2, the first entry must be strictly
        smaller than the second one.
    parenthesis: string defining the boundary of the interval. Accepted values are
    "open", "closed", "left-closed", and "right-closed", optional, default = "open"

    Example
    -------
    >>> from skpro.domains import Interval

    >>> interval = Interval(values=[1, 2], parenthesis='open')
    """

    _PARENTHESIS = {
        "open": "()",
        "closed": "[]",
        "left-closed": "[)",
        "right-closed": "(]",
    }

    def __init__(self, values: List[float], parenthesis: str = "open"):
        self._left, self._right = self._validate_interval(values=values)
        self._parenthesis = self._resolve_parenthesis(parenthesis=parenthesis)

        super().__init__()

    def _validate_interval(self, values: List[float]) -> Tuple[float, float]:
        """Private method to check if a tuple of values is eligible to be an interval.

        A tuple of values is eligible to be an interval if,
            - it has length 2;
            - the first element is strictly smaller than the second element;
        """
        if len(values) != 2:
            raise ValueError(
                "Expected a tuple of length 2 for `values`, "
                f"bot got a tuple of length {len(values)}"
            )
        elif values[0] > values[1]:
            raise ValueError(
                "The left-bound must be strictly smaller then the right-bound, "
                f"but got {values[0]} and {values[1]}."
            )
        elif values[0] == values[1]:
            raise ValueError(
                "The left-bound and the right-bound coincides. "
                "Use the class `Finite` to represent this set."
            )
        else:
            return values[0], values[1]

    def _resolve_parenthesis(self, parenthesis) -> str:
        """Translate the string `parenthesis` in actual parenthesis.

        Translates the string `parenthesis` in one of {"()", "[)", "(]", "[]"},
        taking in account the case where the extremities are infinity.
        """
        if parenthesis not in {"open", "closed", "left-closed", "right-closed"}:
            raise ValueError(
                f"The parameter `parenthesis` must be on of "
                f"{['open', 'closed', 'left-closed', 'right-closed']}, \n"
                f"but got {parenthesis}."
            )
        elif (self._left, self._right) == (-float("inf"), float("inf")):
            return "()"
        elif self._left == -float("inf"):
            return "(" + self._PARENTHESIS[parenthesis][1]
        elif self._right == float("inf"):
            return self._PARENTHESIS[parenthesis][0] + ")"
        else:
            return self._PARENTHESIS[parenthesis]

    def __contains__(self, item) -> bool:
        """Implement `in` operator for the class `Interval`."""
        if isinstance(item, (float, int)) is False:
            return False
        elif item == -float("inf"):
            return item == self._left
        elif item == float("inf"):
            return item == self._right
        elif self._parenthesis == "()":
            return self._left < item < self._right
        elif self._parenthesis == "[)":
            return self._left <= item < self._right
        elif self._parenthesis == "(]":
            return self._left < item <= self._right
        else:
            return self._left <= item <= self._right

    def __str__(self) -> str:
        r"""Return a string representation of `Interval`.

        Returns a string representation of the interval in the form :math:`(a, b)`,
        where :math:`a, b \in \mathbb{R} \cup \{-\inf, +\inf \}` are the extremities
        of the interval.
        """
        return (
            f"{self._parenthesis[0]}{self._left}, {self._right}{self._parenthesis[1]}"
        )

    @property
    def boundary(self) -> Tuple[float, float]:
        """Return the boundary of the interval, i.e., the extremities."""
        return self._left, self._right

    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "values": [1, 2],
            "parenthesis": "open",
        }


class Finite(Domain):
    r"""Finite set of the real line.

    The class implements finite subsets of the real line, i.e., sets of the form
    :math:`\{a_1, ..., a_n\}`, where :math:`a_1, ..., a_n \in \mathbb{R}`.

    Parameters
    ----------
    values: list of int of float of the elements contained in the interval

    Example
    -------
    >>> from skpro.domains import Finite

    >>> finite = Finite([1, 2, 3, 4, 5])
    """

    def __init__(self, values: List[Union[int, float]]):
        self.values = self._validate_values(values=values)
        super().__init__()

    def _validate_values(self, values: List[Union[int, float]]) -> List[float]:
        """Check if a tuple of numbers is elegible to be a finite set."""
        for value in values:
            if not isinstance(value, (float, int)):
                raise TypeError(f"Expected `float`or `int`, but got {type(value)}.")
            if value in [-float("inf"), float("inf")]:
                raise ValueError(f"Value {value} not accepted in finite set.")
        return list(set(values))

    def __contains__(self, item) -> bool:
        """Implement `in` operator for the class `Interval`."""
        return item in self.values

    def __str__(self) -> str:
        r"""Return string representation of `Finite`.

        Returns a string representation of the finite set in the form
        :math:`\{a_1, ..., a_n\}`, where :math:`a_1, ..., a_n` are the elements
        of the set.
        """
        return "{" + ", ".join([str(value) for value in sorted(self.values)]) + "}"

    @property
    def boundary(self) -> Tuple[float, ...]:
        """Return the boundary of the finite set, i.e., the finite set itself."""
        return tuple(sorted(self.values))

    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "values": [1, 2, 3, 4, 5],
        }


class Product(Domain):
    r"""Direct product of sets.

    The class implements (euclidean) direct product of subsets of the real line, i.e.,
    sets of the form :math:`A_1 \times ... \times A_n\}`, where
    :math:`A_1, ..., A_n \subset \mathbb{R}` and :math:`n \in \mathbb{N}`.

    Parameters
    ----------
    elements: list of `Interval` or `Finite` sets. The list should have length at
        least 2

    Example
    -------
    >>> from skpro.domains import Product

    >>> product = Product([Interval([1, 2]), Finite([3, 4, 5, 6]), Interval([7, 9])])
    """

    def __init__(self, elements: List[Union[Interval, Finite]]):
        self.product = self._validate_elements(elements=elements)
        super().__init__()

    def _validate_elements(self, elements: List[Union[Interval, Finite]]):
        """Check that `elements` is elegible to be a product."""
        for element in elements:
            if not isinstance(element, (Interval, Finite)):
                raise TypeError(
                    f"Direct product with type {type(element)} is not supported. \n"
                    f"Supported types are: `Interval` and `Finite`."
                )
        if len(elements) == 0:
            raise ValueError("No elements provided. Expected at least 2 elements.")
        if len(elements) < 2:
            raise ValueError(
                "Not enough elements to accomplish a direct product. \n"
                "Expected at least 2 elements, but got just 1 element."
            )
        return elements

    def __contains__(self, item) -> bool:
        """Implement `in` operator for the class `Interval`."""
        if not isinstance(item, (List, Tuple)):
            raise TypeError(
                "The `in` operator support just types `List` and `Tuple`, \n"
                f"but got {type(item)}."
            )
        if len(item) != len(self.product):
            # in this case they don't live in the same vector space
            return False
        else:
            return all(x in p for x, p in zip(item, self.product))

    def __str__(self) -> str:
        r"""Return a string representation of `Product`.

        Returns a string representation of the product in the form
        :math:`A_1 \times ... \times A_n\}`, where :math:`A_1, ..., A_n` are
        `Interval`(s) or/and `Finite` set(s).
        """
        return " x ".join([str(p) for p in self.product])

    @property
    def boundary(self) -> Tuple:
        """Return the boundary of the `Product`.

        The methods return the boundary of the `Product` class, i.e., the product of
        the boundary of the elements.
        """
        return tuple(element.boundary for element in self.product)

    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "elements": [Interval([1, 2]), Finite([3, 4, 5, 6]), Interval([7, 9])],
        }
