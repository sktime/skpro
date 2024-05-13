import pytest

from skpro.domains import Finite


@pytest.mark.parametrize(
    "values, expected_value",
    [
        ([1, 2], "{1, 2}"),
        ([1, 2, 2], "{1, 2}"),
        ([1.32, 2], "{1.32, 2}"),
        ([11, 1, 78], "{1, 11, 78}"),
    ],
)
def test_init(values, expected_value):
    if Finite(values=values).__str__() != expected_value:
        raise ValueError(
            f"Expected {expected_value}, but got {Finite(values=values).__str__()}."
        )


@pytest.mark.parametrize(
    "values, error, msg",
    [
        ([1, "hello world"], TypeError, "Expected `float`or `int`"),
        ([1, 2, -float("inf")], ValueError, "Value "),
        ([1, 2, float("inf")], ValueError, "Value "),
    ],
)
def test_init_error(values, error, msg):
    with pytest.raises(error, match=msg):
        Finite(values=values)


@pytest.mark.parametrize(
    "values, expected_value",
    [
        ([1, 2, 3], (1, 2, 3)),
        ([2, 2, 1], (1, 2)),
    ],
)
def test_boundary(values, expected_value):
    if Finite(values=values).boundary != expected_value:
        raise ValueError(
            f"Expected {expected_value}, but got {Finite(values=values).boundary}."
        )


@pytest.mark.parametrize(
    "values, element, expected_value",
    [
        ([1, 2], 2, True),
        ([1, 2, 2], 15, False),
    ],
)
def test_contains(values, element, expected_value):
    is_contained = element in Finite(values=values)
    if is_contained != expected_value:
        raise ValueError(
            f"The expression `{element} in {Finite(values=values).__str__()}` "
            f"should \n evaluate {expected_value}, but got {is_contained}."
        )
