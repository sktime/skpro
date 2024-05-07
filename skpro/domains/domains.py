from typing import Tuple, Literal, List, Union


from skpro.base import BaseObject

__all__ = ["Interval", "Finite", "Infinite", "Product"]


class Domain(BaseObject):
    """Base class for domains."""


class Discrete(Domain):
    """Base class for discrete domains."""


class Continuous(Domain):
    """Base class for continuous domains."""


class Interval(Continuous):
    r"""
    Class to represent intervals of the real line, i.e., sets of the form :math:`(a, b)`,
    where `a, b \in \mathbb{R} \cup \{-\inf, +\inf \}`
    """

    _PARENTHESIS = {
        "open": "()",
        "closed": "[]",
        "left-closed": "[)",
        "right-closed": "(]",
    }

    def __init__(
            self,
            values: Tuple[float, float],
            parenthesis: Literal["open", "closed", "left-closed", "right-closed"] = "open"
    ):
        self._left, self._right = self._validate_interval(values=values)
        self._parenthesis = self._resolve_parenthesis(parenthesis=parenthesis)

        super().__init__()

    def _validate_interval(self, values: Tuple[float, float]) -> Tuple[float, float]:
        """
        Private method to check if a tuple of values is eligible to be an interval.
        A tuple of values is eligible to be an interval if,
            - it has length 2;
            - the first element is strictly smaller than the second element;
        """
        if len(values) != 2:
            raise ValueError(
                f"Expected a tuple of length 2 for `values, bot got a tuple of length {len(values)}"
            )
        elif values[0] > values[1]:
            raise ValueError(
                f"The left-bound must be strictly smaller then the right-bound, but got {values[0]} and {values[1]}."
            )
        elif values[0] == values[1]:
            raise Exception(
                "The left-bound and the right-bound coincides. Use the class `Finite` to represent this set."
            )
        else:
            return values[0], values[1]

    def _resolve_parenthesis(self, parenthesis) -> str:
        if self._left == (-float("inf"), -float("inf")):
            return "()"
        elif self._left == -float("inf"):
            return "(" + self._PARENTHESIS[parenthesis][1]
        elif self._right == float("inf"):
            return self._PARENTHESIS[parenthesis][0] + ")"
        else:
            return self._PARENTHESIS[parenthesis]

    def __contains__(self, item) -> bool:
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
        return f"{self._parenthesis[0]}{self._left}, {self._right}{self._parenthesis[1]}"

    @property
    def boundary(self) -> Tuple[float, float]:
        return self._left, self._right


class Finite(Discrete):
    r"""
    Class to represent finite subsets of the real line, i.e., sets of the form :math:`\{a_1, ..., a_n\}`,
    where `a_1, ..., a_n \in \mathbb{R}`.
    """

    def __init__(self, values: Tuple[float, ...]):
        self.values = self._validate_values(values=values)
        super().__init__()

    def _validate_values(self, values: Tuple[float, ...]) -> List[float]:
        """The private method checks if a tuple of numbers is elegible to be a finite set."""
        for value in values:
            if not isinstance(value, (float, int)):
                raise TypeError(f"Expected `float`or `int`, but got {type(value)}.")
            if value in [-float("inf"), float("inf")]:
                raise ValueError(f"Value {value} not accepted in finite set.")
        return list(set(values))

    def __contains__(self, item) -> bool:
        return item in self.values

    def __str__(self) -> str:
        return "{" + ", ".join([str(value) for value in sorted(self.values)]) + "}"

    @property
    def boundary(self) -> Tuple[float, ...]:
        return tuple(self.values)


class Infinite(Discrete):
    pass


class Product(Domain):
    r"""
    Class to represent (euclidean) direct product of sets subsets of the real line, i.e., sets of the form
    :math:`A_1 \times ... \times A_n\}`, where `A_1, ..., A_n \subset \mathbb{R}` and :math:`n \in \mathbb{N}`.
    """

    def __init__(self, elements: List[Union[Interval, Finite, Infinite]]):
        self.product = self._validate_elements(elements=elements)
        super().__init__()

    def _validate_elements(self, elements: List[Union[Interval, Finite, Infinite]]):
        for element in elements:
            if len(elements) < 2:
                raise ValueError(
                    f"Not enough elements do accomplish a direct product. Expected at least 2, but got {len(elements)}"
                )
            if not isinstance(element, (Interval, Finite, Infinite)):
                raise TypeError(f"Direct product with type {type(element)} is not supported.")
        return elements

    def __contains__(self, item) -> bool:
        if len(item) != len(self.product):
            return False
        else:
            return all(x in p for x, p in zip(item, self.product))

    def __str__(self) -> str:
        return " x ".join([str(p) for p in self.product])

    @property
    def boundary(self):
        return tuple(element.boundary for element in self.product)
