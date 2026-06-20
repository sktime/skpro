# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for symbolic set representations."""

__author__ = ["khushmagrawal"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.base._set import (
    BaseSet,
    EmptySet,
    FiniteSet,
    IntegerSet,
    IntersectionSet,
    IntervalSet,
    RealSet,
    UnionSet,
)


@pytest.fixture
def sample_indices():
    """Sample index and columns for tabular tests."""
    return pd.Index([0, 1, 2], name="rows"), pd.Index(["a", "b"], name="cols")


@pytest.fixture
def multiindex_data():
    """MultiIndex and DataFrame for indexing tests."""
    ix = pd.MultiIndex.from_product([["A", "B"], [1, 2]], names=["L1", "L2"])
    cols = pd.Index(["v1", "v2"])
    df = pd.DataFrame(np.random.randn(4, 2), index=ix, columns=cols)
    return ix, cols, df


# Instances for generic interface testing
def get_all_set_types():
    return [
        IntervalSet(0, 1),
        FiniteSet([0, 1]),
        IntegerSet(0, 10),
        EmptySet(),
        RealSet(),
        UnionSet(FiniteSet([0]), IntervalSet(1, 2)),
        IntersectionSet(IntervalSet(0, 10), IntervalSet(5, 15)),
    ]


class TestSetInterface:
    """Generic tests for the BaseSet interface across all subclasses."""

    @pytest.mark.parametrize("s", get_all_set_types())
    def test_metadata_scalar(self, s):
        """Test shape and ndim for scalar sets."""
        assert s.shape == ()
        assert s.ndim == 0
        assert s.index is None
        assert s.columns is None

    @pytest.mark.parametrize(
        "s_cls", [IntervalSet, FiniteSet, IntegerSet, EmptySet, RealSet]
    )
    def test_index_coercion(self, s_cls):
        """Test that list inputs for index/columns are coerced to pd.Index."""
        if s_cls in (IntervalSet, IntegerSet):
            s = s_cls(0, 1, index=[0, 1], columns=["a"])
        elif s_cls == FiniteSet:
            s = s_cls([0], index=[0, 1], columns=["a"])
        else:
            s = s_cls(index=[0, 1], columns=["a"])

        assert isinstance(s.index, pd.Index)
        assert isinstance(s.columns, pd.Index)
        assert s.shape == (2, 1)

    @pytest.mark.parametrize("s", get_all_set_types())
    def test_serialization(self, s):
        """Test str and repr output something sensible."""
        assert isinstance(str(s), str)
        assert len(str(s)) > 0
        assert s.__class__.__name__ in repr(s)

    @pytest.mark.parametrize("s", get_all_set_types())
    def test_output_format_consistency(self, s):
        """Verify: scalar in -> bool out; array in -> array out; DF in -> DF out."""
        # 1. Scalar
        assert isinstance(s.contains(0.5), (bool, np.bool_))

        # 2. Numpy array
        x_np = np.array([0, 0.5, 1])
        res_np = s.contains(x_np)
        assert isinstance(res_np, np.ndarray)
        assert res_np.dtype == bool
        assert res_np.shape == (3,)

        # 3. DataFrame (scalar set)
        x_df = pd.DataFrame({"a": [0, 1], "b": [0.5, 2]})
        res_df = s.contains(x_df)
        assert isinstance(res_df, pd.DataFrame)
        assert res_df.shape == (2, 2)
        assert res_df.index.equals(x_df.index)

    def test_base_set_defaults(self):
        """Check BaseSet defaults and abstract behavior."""
        s = BaseSet()
        assert s.get_tag("is_discrete") is False
        assert s.get_tag("is_continuous") is False
        with pytest.raises(NotImplementedError):
            s.contains(0)


class TestTabularAndBroadcasting:
    """Focus on 2D sets, alignment, and MultiIndex support."""

    def test_tabular_contains_alignment(self, sample_indices):
        """Verify that contains aligns input DataFrame to set index/columns."""
        idx, cols = sample_indices
        s = IntervalSet(0, 5, index=idx, columns=cols)

        # Input has same shape but different order
        x = pd.DataFrame({"b": [1, 2, 3], "a": [6, 0, 1]}, index=[2, 0, 1])
        res = s.contains(x)

        # Result should match input's index/columns
        assert res.index.equals(x.index)
        assert res.columns.equals(x.columns)

        # Value check: x.loc[2, 'a'] is 6 (Out), s.loc[2, 'a'] is [0, 5]
        assert bool(res.loc[2, "a"]) is False
        assert bool(res.loc[0, "a"]) is True

    def test_scalar_set_broadcasting_to_df(self):
        """Scalar set should broadcast its logic to any input shape."""
        s = IntegerSet(lower=0, upper=10)
        x = pd.DataFrame({"a": [1, 11], "b": [5, -1]})
        res = s.contains(x)
        expected = pd.DataFrame({"a": [True, False], "b": [True, False]})
        pd.testing.assert_frame_equal(res, expected)

    def test_multiindex_support(self, multiindex_data):
        """Sets should handle MultiIndex in both definition and input."""
        ix, cols, df = multiindex_data
        s = IntervalSet(0, 1, index=ix, columns=cols)

        assert s.shape == (4, 2)
        res = s.contains(df)
        assert isinstance(res, pd.DataFrame)
        assert isinstance(res.index, pd.MultiIndex)
        assert res.index.equals(ix)

    def test_vectorized_boundaries(self, sample_indices):
        """IntervalSet with array-like boundaries (vectorized set)."""
        idx, cols = sample_indices
        # Row 0: [0, 5], Row 1: [10, 15], Row 2: [20, 25] (broadcast across columns)
        s = IntervalSet(lower=[0, 10, 20], upper=[5, 15, 25], index=idx, columns=cols)
        x = pd.DataFrame([[3, 3], [12, 12], [22, 100]], index=idx, columns=cols)
        res = s.contains(x)
        expected = pd.DataFrame(
            [[True, True], [True, True], [True, False]], index=idx, columns=cols
        )
        pd.testing.assert_frame_equal(res, expected)


# -----------------------------------------------------------------------------
# Subclass Specific Testing
# -----------------------------------------------------------------------------


class TestIntervalSet:
    """Interval specific logic (open/closed, boundaries)."""

    @pytest.mark.parametrize(
        "type_str, inner, left, right",
        [
            ("[]", True, True, True),
            ("()", True, False, False),
            ("[)", True, True, False),
            ("(]", True, False, True),
        ],
    )
    def test_boundary_inclusivity(self, type_str, inner, left, right):
        """Test Open/Closed math boundaries."""
        s = IntervalSet(0, 5, interval_type=type_str)
        assert s.contains(2.5) == inner
        assert s.contains(0) == left
        assert s.contains(5) == right

    def test_edge_cases(self):
        """Test Infinity and NaN boundaries."""
        # Inf
        s_inf = IntervalSet(-np.inf, np.inf)
        assert s_inf.contains(1e10) is True
        assert s_inf.boundary() == (-np.inf, np.inf)

        # NaN
        s_nan = IntervalSet(np.nan, np.nan)
        assert s_nan.contains(0) is False

        # Half-bounded
        s_half = IntervalSet(0, np.inf, "[)")
        assert s_half.contains(0) is True
        assert s_half.contains(-1) is False

    def test_interval_type_broadcast(self):
        """Test interval_type broadcasting logic."""
        # Using a scalar interval type cleanly handles bounding logics uniformly
        s = IntervalSet([0, 0], [10, 10], interval_type="()", index=[1, 2])
        res = s.contains(pd.DataFrame({"a": [0, 0]}, index=[1, 2]))
        assert bool(res.iloc[0, 0]) is False
        assert bool(res.iloc[1, 0]) is False

    def test_lazy_evaluation_shape_mismatch(self):
        """Test lazy evaluation shape mismatch initialization."""
        with pytest.raises(ValueError, match="broadcast"):
            IntervalSet([0, 1], [10, 20], index=[1, 2, 3])


class TestFiniteSet:
    """Specifics for set of discrete points."""

    def test_membership(self):
        """Test explicit floating inclusions."""
        s = FiniteSet([1, 2, 4.5])
        assert s.contains(1) is True
        assert s.contains(3) is False
        assert s.contains(4.5) is True
        assert s.get_tag("is_discrete") is True

    def test_boundary(self):
        """Test accurate reduction of set components."""
        s = FiniteSet([10, 2, 5])
        # FiniteSet boundaries are simply (min, max)
        l, u = s.boundary()
        np.testing.assert_array_equal(l, 2)
        np.testing.assert_array_equal(u, 10)

    def test_multiindex_flattening(self, multiindex_data):
        """Test MultiIndex flattening over tabular arrays."""
        ix, cols, df = multiindex_data
        s = FiniteSet(
            np.array([[1], [2], [3], [4]], dtype=object), index=ix, columns=cols
        )
        res = s.contains(df)
        assert res.index.equals(ix)

    def test_finiteset_tabular_evaluation(self):
        """Test FiniteSet values tabular broadcasting."""
        vals = np.empty((2, 1), dtype=object)
        vals[0, 0] = [1, 2]
        vals[1, 0] = [3, 4]
        s = FiniteSet(vals, index=[1, 2], columns=["a"])
        res = s.contains(pd.DataFrame({"a": [2, 2]}, index=[1, 2]))
        assert bool(res.iloc[0, 0]) is True
        assert bool(res.iloc[1, 0]) is False

        # Test boundary mathematical extraction
        l, u = s.boundary()
        np.testing.assert_array_equal(l, [[1], [3]])
        np.testing.assert_array_equal(u, [[2], [4]])


class TestIntegerSet:
    """Specifics for countably infinite/finite integer sets."""

    def test_integer_logic(self):
        """Ensure math accurately rejects precision floats."""
        s = IntegerSet(lower=0, upper=5)
        assert s.contains(3) is True
        assert s.contains(3.1) is False
        assert s.contains(-1) is False
        assert s.contains(6) is False
        assert s.get_tag("is_discrete") is True

    def test_metadata_preserved_in_contains(self):
        """Test metadata preservation across Pandas structures."""
        s = IntegerSet(0, 10)
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["p1", "p2"])
        res = s.contains(df)
        assert isinstance(res, pd.DataFrame)
        assert list(res.columns) == ["A", "B"]
        assert list(res.index) == ["p1", "p2"]


class TestEmptySet:
    """Specifics for completely unpopulated boundaries."""

    def test_emptyset_tabular_scalar_query(self):
        """Test EmptySet ignores shape via pd.Index properties."""
        s_empty = EmptySet(index=[1, 2, 3])
        res_empty = s_empty.contains(5)
        assert res_empty.shape == (3, 1)


class TestRealSet:
    """Specifics for totally unrestricted boundaries."""

    def test_realset_tabular_scalar_query(self):
        """Test RealSet ignores shape via pd.Index properties."""
        s_real = RealSet(index=[1, 2, 3])
        res_real = s_real.contains(5)
        assert res_real.shape == (3, 1)


class TestUnionSet:
    """Tests for Union logic arrays."""

    def test_union_basic(self):
        """Basic interval overlapping queries."""
        u = UnionSet(IntervalSet(0, 1), IntervalSet(5, 6))
        assert bool(u.contains(0.5)) is True
        assert bool(u.contains(3)) is False
        assert bool(u.contains(5.5)) is True
        assert u.boundary() == (0, 6)

    def test_heterogeneous_index_union(self):
        """Test heterogeneous index alignments in UnionSet."""
        s1 = IntervalSet(0, 10, index=pd.Index([1, 2, 3]))
        s2 = IntervalSet(20, 30, index=pd.Index([3, 4, 5]))
        u = UnionSet(s1, s2)

        # DataFrame explicitly captures partial mismatches
        df = pd.DataFrame({"col": [5, 5, 25, 25, 25]}, index=[1, 2, 3, 4, 5])
        res = u.contains(df)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 5

    def test_scalar_set_broadcast_union(self):
        """Test broadcasted scalar set mapped against UnionSet logic."""
        tab_set = IntervalSet([0, 0], [10, 10], index=[1, 2])
        scalar_set = FiniteSet([100])
        u = UnionSet(tab_set, scalar_set)
        res = u.contains(100)
        assert res.shape == (2, 1)

    def test_union_boundary_flattening(self):
        """Test recursive inner flattening math logic in boundaries."""
        s1 = IntervalSet([0, 10], [5, 15], index=[1, 2])
        s2 = IntervalSet([-5, 5], [0, 10], index=[1, 2])
        u = UnionSet(s1, s2)
        l, h = u.boundary()
        np.testing.assert_array_equal(l, [-5, 5])
        np.testing.assert_array_equal(h, [5, 15])

    def test_raw_bitwise_or_numpy_arrays(self):
        """Test NumPy ndarray logical collapse against DataFrames."""
        s1 = IntervalSet(0, 5)
        s2 = IntegerSet(10, 15)
        u = UnionSet(s1, s2)
        x = np.array([2, 12, 20])
        res = u.contains(x)
        assert res.shape == (3,)
        np.testing.assert_array_equal(res, [True, True, False])


class TestIntersectionSet:
    """Tests for Intersectional mapping geometry."""

    def test_intersection_basic(self):
        """Basic bound inclusion masks."""
        i = IntersectionSet(IntervalSet(0, 10), IntervalSet(5, 15))
        assert bool(i.contains(7)) is True
        assert bool(i.contains(2)) is False
        assert bool(i.contains(12)) is False
        assert i.boundary() == (5, 10)

    def test_identity_algebra(self):
        """Ensure abstract object geometry behaves cleanly when nested."""
        a = IntervalSet(0, 1)
        # A n Real = A
        assert bool(IntersectionSet(a, RealSet()).contains(0.5)) == bool(
            a.contains(0.5)
        )
        # A n Empty = Empty
        assert bool(IntersectionSet(a, EmptySet()).contains(0.5)) is False

    def test_nested_algebra(self):
        """Test mathematical overlaps combining Union arrays and Intersection limits."""
        s1 = IntersectionSet(IntervalSet(0, 10), IntervalSet(5, 15))  # [5, 10]
        s2 = FiniteSet([20])
        u = UnionSet(s1, s2)
        assert bool(u.contains(7)) is True
        assert bool(u.contains(20)) is True
        assert bool(u.contains(15)) is False

    def test_intersection_ghost_boundary(self):
        """Test array mathematical geometry across distinct limits."""
        s1 = IntervalSet([0, 10], [10, 20], index=[1, 2])
        s2 = IntervalSet([5, 5], [15, 15], index=[1, 2])
        i = IntersectionSet(s1, s2)
        l, h = i.boundary()
        np.testing.assert_array_equal(l, [5, 10])
        np.testing.assert_array_equal(h, [10, 15])

    def test_intersection_null(self):
        """Test completely disjoint geometries collapsing strictly to NaN spaces."""
        s1 = IntervalSet([0, 30], [5, 35], index=[1, 2])
        s2 = IntervalSet([10, 10], [15, 15], index=[1, 2])
        i = IntersectionSet(s1, s2)
        l, h = i.boundary()

        # Row 1 overrides null overlap because 10 > 5 -> NaN
        assert np.isnan(l[0])
        assert np.isnan(h[0])

        # Row 2 overrides null overlap because 30 > 15 -> NaN
        assert np.isnan(l[1])
        assert np.isnan(h[1])
