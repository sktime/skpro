"""Example generation for testing.

Exports dict of examples, useful for testing as fixtures.

example_dict: dict indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are data objects, considered examples for the mtype
    all examples with same index are considered "same" on scitype content
    if None, indicates that representation is not possible

example_lossy: dict of bool indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are bool, indicate whether representation has information removed
    all examples with same index are considered "same" on scitype content

example_metadata: dict of metadata dict, indexed by pair
  1st element = considered as this scitype - str
  2nd element = int - index of example
  (there is no "mtype" element, as properties are equal for all mtypes)
elements are metadata dict, as returned by check_is_mtype
    used as expected return of check_is_mtype in tests

overall, conversions from non-lossy representations to any other ones
    should yield the element exactly, identidally (given same index)
"""

import numpy as np
import pandas as pd

from skpro.datatypes._base import BaseExample

###
# example 0: univariate


class _ProbaUniv(BaseExample):
    _tags = {
        "scitype": "Proba",
        "index": 0,
        "metadata": {
            "is_univariate": True,
            "is_empty": False,
            "has_nans": False,
        },
    }


class _ProbaUnivPredQ(_ProbaUniv):
    _tags = {
        "mtype": "pred_quantiles",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        pred_q = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4]})
        pred_q.columns = pd.MultiIndex.from_product([["foo"], [0.2, 0.6]])

        return pred_q


class _ProbaUnivPredInt(_ProbaUniv):
    _tags = {
        "mtype": "pred_interval",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        # we need to use this due to numerical inaccuracies
        # from the binary based representation
        pseudo_0_2 = 2 * np.abs(0.6 - 0.5)

        pred_int = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4]})
        pred_int.columns = pd.MultiIndex.from_tuples(
            [("foo", 0.6, "lower"), ("foo", pseudo_0_2, "upper")]
        )

        return pred_int


###
# example 1: multi


class _ProbaMulti(BaseExample):
    _tags = {
        "scitype": "Proba",
        "index": 1,
        "metadata": {
            "is_univariate": False,
            "is_empty": False,
            "has_nans": False,
        },
    }


class _ProbaMultiPredQ(_ProbaMulti):
    _tags = {
        "mtype": "pred_quantiles",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        pred_q = pd.DataFrame(
            {0.2: [1, 2, 3], 0.6: [2, 3, 4], 42: [5, 3, -1], 46: [5, 3, -1]}
        )
        pred_q.columns = pd.MultiIndex.from_product([["foo", "bar"], [0.2, 0.6]])

        return pred_q


class _ProbaMultiPredInt(_ProbaMulti):
    _tags = {
        "mtype": "pred_interval",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        # we need to use this due to numerical inaccuracies
        # from the binary based representation
        pseudo_0_2 = 2 * np.abs(0.6 - 0.5)

        pred_int = pd.DataFrame(
            {0.2: [1, 2, 3], 0.6: [2, 3, 4], 42: [5, 3, -1], 46: [5, 3, -1]}
        )
        pred_int.columns = pd.MultiIndex.from_tuples(
            [
                ("foo", 0.6, "lower"),
                ("foo", pseudo_0_2, "upper"),
                ("bar", 0.6, "lower"),
                ("bar", pseudo_0_2, "upper"),
            ]
        )

        return pred_int
