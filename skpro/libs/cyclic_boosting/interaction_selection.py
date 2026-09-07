from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from skpro.libs.cyclic_boosting.binning import BinNumberTransformer
from skpro.libs.cyclic_boosting.utils import multidim_binnos_to_lexicographic_binnos


def create_interactions(features1D: List[str], dim: int) -> List[Tuple]:
    """
    Creation of all possible dim-dimensional combinations from one-dimensional feature list.

    Parameters
    ----------
    features1D
        list of one-dimensional features
    dim : int
        maximal dimensionality of interactions terms to be created

    Returns
    -------
        list of tuples of strings (feature names) representing all dim-dimensional feature combinations
    """
    interactions = []
    for n in range(2, dim + 1):
        interactions += list(combinations(features1D, n))
    return interactions


def build_binned_interaction_features(
    X: pd.DataFrame,
    interaction_terms: List[Tuple],
    feature_properties: Dict[str, object],
    number_of_bins: Optional[int] = 100,
) -> pd.DataFrame:
    """
    Creation of multi-dimensional features from one-dimensional ones for all interaction_terms.

    Parameters
    ----------
    X : pd.DataFrame
        design matrix
    interaction_terms
        list of tuples of all multi-dimensional feature combinations to be used as interaction terms
    feature_properties : dict
        names and pre-processing flags of all one-dimensional features
    number_of_bins : int
        number of bins to be used for binning of non-categorical features

    Returns
    -------
    pd.DataFrame
        data frame with different interaction terms as flattened (binned) multi-dimensional features
    """
    binner = BinNumberTransformer(
        n_bins=number_of_bins, feature_properties=feature_properties, inplace=False
    )
    binned_1D = binner.fit_transform(X)

    features_multidim = pd.DataFrame()
    for it in interaction_terms:
        cols = []
        for i in it:
            cols.append(binned_1D[i])
        binned_multidim = np.column_stack(tuple(cols))
        features_multidim[it], _ = multidim_binnos_to_lexicographic_binnos(
            binned_multidim
        )

    return features_multidim


def select_interaction_terms_anova(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_properties: Dict[str, object],
    interaction_dim: int,
    k_best: int,
    classification: Optional[bool] = False,
) -> List[str]:
    """
    ANOVA selection of interaction terms for a given data set by means of binning.

    Parameters
    ----------
    X : pd.DataFrame
        design matrix
    y : np.ndarray
        target
    feature_properties : dict
        names and pre-processing flags of all one-dimensional features
    interaction_dim
        maximal dimensionality of interactions terms to be considered
    k_best : int
        number of interaction terms to be selected

    Returns
    -------
        list of the names of the selected interaction terms
    """
    interaction_terms = create_interactions(
        list(feature_properties.keys()), interaction_dim
    )

    interaction_term_features = build_binned_interaction_features(
        X, interaction_terms, feature_properties
    )

    if classification:
        anova_est = SelectKBest(f_classif, k=k_best)
    else:
        anova_est = SelectKBest(f_regression, k=k_best)
    anova_est.fit(interaction_term_features, y)
    return list(
        anova_est.get_feature_names_out(
            input_features=interaction_term_features.columns
        )
    )
