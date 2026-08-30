import pytest

from skpro.domains import Interval


@pytest.mark.parametrize(
    "values, parenthesis, expected_value",
    [
        ([1, 2], "open", "(1, 2)"),
        ([1, 2], "closed", "[1, 2]"),
        ([1, 2], "left-closed", "[1, 2)"),
        ([1, 2], "right-closed", "(1, 2]"),
        ([-float("inf"), 2], "open", "(-inf, 2)"),
        ([-float("inf"), 2], "closed", "(-inf, 2]"),
        ([-float("inf"), 2], "left-closed", "(-inf, 2)"),
        ([-float("inf"), 2], "right-closed", "(-inf, 2]"),
        ([1, float("inf")], "open", "(1, inf)"),
        ([1, float("inf")], "closed", "[1, inf)"),
        ([1, float("inf")], "left-closed", "[1, inf)"),
        ([1, float("inf")], "right-closed", "(1, inf)"),
        ([-float("inf"), float("inf")], "open", "(-inf, inf)"),
        ([-float("inf"), float("inf")], "closed", "(-inf, inf)"),
        ([-float("inf"), float("inf")], "left-closed", "(-inf, inf)"),
        ([-float("inf"), float("inf")], "right-closed", "(-inf, inf)"),
    ],
)
def test_init(values, parenthesis, expected_value):
    if Interval(values=values, parenthesis=parenthesis).__str__() != expected_value:
        raise ValueError(
            f"Expected {expected_value}, "
            f"but got {Interval(values=values, parenthesis=parenthesis).__str__()}."
        )


@pytest.mark.parametrize(
    "values, parenthesis, error, msg",
    [
        ([1, 2, 3], "open", ValueError, "Expected a tuple of length 2 for `values`"),
        (
            [2, 1],
            "open",
            ValueError,
            "The left-bound must be strictly smaller then the right-bound",
        ),
        ([1, 1], "open", ValueError, "The left-bound and the right-bound coincides."),
        (
            [1, 2],
            "hello world",
            ValueError,
            "The parameter `parenthesis` must be on of",
        ),
    ],
)
def test_init_error(values, parenthesis, error, msg):
    with pytest.raises(error, match=msg):
        Interval(values=values, parenthesis=parenthesis)


@pytest.mark.parametrize(
    "values, parenthesis, expected_value",
    [
        ([1, 2], "open", (1, 2)),
        ([1, 2], "closed", (1, 2)),
        ([1, 2], "left-closed", (1, 2)),
        ([1, 2], "right-closed", (1, 2)),
        ([-float("inf"), 2], "open", (-float("inf"), 2)),
        ([-float("inf"), 2], "closed", (-float("inf"), 2)),
        ([-float("inf"), 2], "left-closed", (-float("inf"), 2)),
        ([-float("inf"), 2], "right-closed", (-float("inf"), 2)),
        ([1, float("inf")], "open", (1, float("inf"))),
        ([1, float("inf")], "closed", (1, float("inf"))),
        ([1, float("inf")], "left-closed", (1, float("inf"))),
        ([1, float("inf")], "right-closed", (1, float("inf"))),
        ([-float("inf"), float("inf")], "open", (-float("inf"), float("inf"))),
        ([-float("inf"), float("inf")], "closed", (-float("inf"), float("inf"))),
        ([-float("inf"), float("inf")], "left-closed", (-float("inf"), float("inf"))),
        ([-float("inf"), float("inf")], "right-closed", (-float("inf"), float("inf"))),
    ],
)
def test_boundary(values, parenthesis, expected_value):
    if Interval(values=values, parenthesis=parenthesis).boundary != expected_value:
        raise ValueError(
            f"Expected {expected_value}, "
            f"but got {Interval(values=values, parenthesis=parenthesis).boundary}."
        )


@pytest.mark.parametrize(
    "values, parenthesis, element, expected_value",
    [
        ([1, 3], "open", "abc", False),
        ([1, 3], "open", 1.5, True),
        ([1, 3], "open", 2, True),
        ([1, 3], "open", 1, False),
        ([1, 3], "open", 3, False),
        ([1, 3], "closed", 1, True),
        ([1, 3], "closed", 3, True),
        ([1, 3], "left-closed", 1, True),
        ([1, 3], "left-closed", 3, False),
        ([1, 3], "right-closed", 1, False),
        ([1, 3], "right-closed", 3, True),
        ([1, 3], "open", -float("inf"), False),
        ([-float("inf"), 3], "open", -float("inf"), True),
        ([1, 3], "open", float("inf"), False),
        ([1, float("inf")], "open", float("inf"), True),
    ],
)
def test_contains(values, parenthesis, element, expected_value):
    is_contained = element in Interval(values=values, parenthesis=parenthesis)
    if is_contained != expected_value:
        raise ValueError(
            f"The expression `{element} in "
            f"{Interval(values=values, parenthesis=parenthesis).__str__()}` should \n "
            f"evaluate {expected_value}, but got {is_contained}."
        )
