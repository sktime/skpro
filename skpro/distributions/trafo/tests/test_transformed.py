"""Test transformed distributions module."""

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from skpro.compose import DifferentiableTransformer
from skpro.distributions.trafo._transformed import _coerce_to_diff_transformer


@pytest.mark.parametrize(
    "transform, inverse_transform, expected_transformer, inverse_tag",
    [
        (MinMaxScaler(), None, MinMaxScaler, "exact"),
        (expit, logit, FunctionTransformer, "approx"),
        (
            FunctionTransformer(func=expit, inverse_func=logit),
            None,
            FunctionTransformer,
            "approx",
        ),
        (
            DifferentiableTransformer(
                transformer=FunctionTransformer(func=expit, inverse_func=logit),
                transform_func_diff=lambda x: expit(x) * (1 - expit(x)),
            ),
            None,
            FunctionTransformer,
            "exact",
        ),
        (MinMaxScaler().transform, None, FunctionTransformer, "approx"),
        (
            MinMaxScaler().transform,
            MinMaxScaler().inverse_transform,
            FunctionTransformer,
            "approx",
        ),
    ],
)
def test_coerce_differentiable(
    transform,
    inverse_transform,
    expected_transformer,
    inverse_tag,
):
    """Test that a DifferentiableTransformer is created correctly."""

    transformer = _coerce_to_diff_transformer(
        transform,
        inverse_transform=inverse_transform,
    )

    transformer.fit(np.array([[0], [1]]))

    assert isinstance(transformer, DifferentiableTransformer)
    assert isinstance(transformer.transformer, expected_transformer)
    # consider moving this assert to test_transformer.
    assert transformer._get_transform_diff_capabilities() == inverse_tag


def test_coerce_differentiable_warnings_for_bound_methods():
    """Test that bound methods from a transformer raise appropriate warning."""
    mms = MinMaxScaler()
    mms.fit(np.array([[0], [1]]))
    diff_mms = DifferentiableTransformer(transformer=mms)
    diff_mms.fit(np.array([[0], [1]]))

    with pytest.warns(UserWarning, match="consider passing the full transformer"):
        transformer = _coerce_to_diff_transformer(
            diff_mms.inverse_transform,
            inverse_transform=diff_mms.transform,
        )

    assert isinstance(transformer, DifferentiableTransformer)
