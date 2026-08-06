import pytest

from skpro.domains import Finite, Interval, Product


@pytest.mark.parametrize(
    "elements, expected_value",
    [
        ([Interval([1, 2]), Interval([1, 2])], "(1, 2) x (1, 2)"),
        ([Interval([1, 2]), Finite([1, 2, 3])], "(1, 2) x {1, 2, 3}"),
        ([Finite([1, 2, 3]), Finite([1, 2, 3])], "{1, 2, 3} x {1, 2, 3}"),
        (
            [Interval([1, 2]), Finite([1, 2, 3]), Interval([5, 8], "left-closed")],
            "(1, 2) x {1, 2, 3} x [5, 8)",
        ),
    ],
)
def test_init(elements, expected_value):
    if Product(elements=elements).__str__() != expected_value:
        raise ValueError(
            f"Expected {expected_value}, "
            f"but got {Product(elements=elements).__str__()}."
        )


@pytest.mark.parametrize(
    "elements, error, msg",
    [
        ([Interval([1, 2]), 1], TypeError, "Direct product with type"),
        ([], ValueError, "No elements provided."),
        ([Interval([1, 2])], ValueError, "Not enough elements to accomplish"),
    ],
)
def test_init_error(elements, error, msg):
    with pytest.raises(error, match=msg):
        Product(elements=elements)


@pytest.mark.parametrize(
    "elements",
    [
        ([Interval([1, 2]), Interval([1, 2])]),
        ([Interval([1, 2]), Finite([1, 2, 3])]),
        ([Finite([1, 2, 3]), Finite([1, 2, 3])]),
        ([Interval([1, 2]), Finite([1, 2, 3]), Interval([5, 8], "left-closed")]),
    ],
)
def test_boundary(elements):
    boundary = Product(elements=elements).boundary
    expected_boundary = tuple(element.boundary for element in elements)
    if boundary != expected_boundary:
        raise ValueError(f"Expected {expected_boundary}, but got {boundary}.")


@pytest.mark.parametrize(
    "elements, item, expected_value",
    [
        ([Interval([1, 2]), Interval([1, 2])], (1.5, 1.5), True),
        ([Interval([1, 2]), Interval([1, 2])], (1, 1.5), False),
        ([Interval([1, 2]), Finite([1, 2, 3])], [1.5, 3], True),
        ([Interval([1, 2]), Finite([1, 2, 3])], [1, 13], False),
        (
            [Interval([1, 2]), Finite([1, 2, 3]), Interval([5, 8])],
            tuple((1.5, 3, 7)),
            True,
        ),
        ([Interval([1, 2]), Finite([1, 2, 3]), Interval([5, 8])], tuple((2, 3)), False),
    ],
)
def test_contains(elements, item, expected_value):
    is_contained = item in Product(elements=elements)
    if is_contained != expected_value:
        raise ValueError(
            f"The expression `{item} in {Product(elements=elements).__str__()}` "
            f"should \n evaluate {expected_value}, but got {is_contained}."
        )
