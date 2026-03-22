"""Registry of mtypes for Proba scitype.

See datatypes._registry for API.
"""

import pandas as pd

__all__ = [
    "MTYPE_REGISTER_PROBA",
    "MTYPE_LIST_PROBA",
]


MTYPE_REGISTER_PROBA = [
    ("pred_interval", "Proba", "predictive intervals", None),
    ("pred_quantiles", "Proba", "quantile predictions", None),
    ("pred_var", "Proba", "variance predictions", None),
    # ("pred_dost", "Proba", "full distribution predictions, tensorflow-probability"),
]

MTYPE_LIST_PROBA = pd.DataFrame(MTYPE_REGISTER_PROBA)[0].values
