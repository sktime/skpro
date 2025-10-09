"""Test cases for the MultiIndex functionality of the BaseDistribution.

Uses the Normal distribution, but is intended to trigger the base layer.
"""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.normal import Normal


@pytest.fixture
def normal_dist():
    ix = pd.MultiIndex.from_product([(1, 2), (2, 3)])
    return Normal(np.array([[1, 2], [2, 3], [4, 5], [6, 7]]), 2, index=ix)


def test_loc_partial_level(normal_dist):
    result = normal_dist.loc[1]
    expected_index = pd.MultiIndex.from_tuples([(1, 2), (1, 3)])
    np.testing.assert_array_equal(result.index, expected_index)
    assert result.mean().shape == (2, 2)


def test_loc_full_tuple(normal_dist):
    result = normal_dist.loc[(2, 2)]
    expected_index = pd.MultiIndex.from_tuples([(2, 2)])
    np.testing.assert_array_equal(result.index, expected_index)
    assert result.mean().shape == (1, 2)


def test_loc_list_of_keys(normal_dist):
    result = normal_dist.loc[[(1, 2), (2, 3)]]
    expected_index = pd.MultiIndex.from_tuples([(1, 2), (2, 3)])
    np.testing.assert_array_equal(result.index, expected_index)
    assert result.mean().shape == (2, 2)


def test_iloc_single_row(normal_dist):
    result = normal_dist.iloc[0]
    expected_index = pd.MultiIndex.from_tuples([(1, 2)])
    np.testing.assert_array_equal(result.index, expected_index)
    assert result.mean().shape == (1, 2)


def test_iloc_multiple_rows(normal_dist):
    result = normal_dist.iloc[[0, 3]]
    expected_index = pd.MultiIndex.from_tuples([(1, 2), (2, 3)])
    np.testing.assert_array_equal(result.index, expected_index)
    assert result.mean().shape == (2, 2)


def test_iloc_column_slice(normal_dist):
    result = normal_dist.iloc[:, 1]
    expected_index = normal_dist.index
    assert result.mean().shape == (4, 1)
    np.testing.assert_array_equal(result.index, expected_index)


def test_loc_row_col(normal_dist):
    result = normal_dist.loc[(1, 2), :]
    expected_index = pd.MultiIndex.from_tuples([(1, 2)])
    assert result.mean().shape == (1, 2)
    np.testing.assert_array_equal(result.index, expected_index)
