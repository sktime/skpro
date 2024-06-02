"""Interface adapters to scikit-survival survival forest models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.sksurv import _SksurvAdapter
from skpro.survival.base import BaseSurvReg


class SurvivalForestSkSurv(_SksurvAdapter, BaseSurvReg):
    """Random survival forest from scikit-survival.

    Direct interface to ``sksurv.ensemble.RandomSurvivalForest``, by ``sebp``.

    A random survival forest is a meta estimator that fits a number of
    survival trees on various sub-samples of the dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample
    size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    In each survival tree, the quality of a split is measured by the
    log-rank splitting rule.

    See [1]_ and [2]_ for further description.

    Parameters
    ----------
    n_estimators : integer, optional, default: 100
        The number of trees in the forest.

    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional, default: None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.

    random_state : int, RandomState instance or None, optional, default: None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    Attributes
    ----------
    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.

    References
    ----------
    .. [1] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841–860.

    .. [2] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25–31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    _tags = {
        "authors": ["sebp", "fkiraly"],  # sebp credit for interfaced estimator=
    }

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        warm_start=False,
        max_samples=None,
        verbose=0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.max_samples = max_samples
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def _get_sksurv_class(self):
        """Getter of the sksurv class to be used for the adapter."""
        from sksurv.ensemble.forest import RandomSurvivalForest

        return RandomSurvivalForest

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}

        params2 = {
            "n_estimators": 99,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "log2",
            "max_leaf_nodes": 10,
            "oob_score": True,
        }

        return [params1, params2]


class SurvivalForestXtraSkSurv(_SksurvAdapter, BaseSurvReg):
    """Survival random forest with extra randomization-averaging, from scikit-survival.

    Direct interface to ``sksurv.ensemble.ExtraSurvivalTrees``, by ``sebp``.

    This class implements a meta estimator that fits a number of randomized
    survival trees (a.k.a. extra-trees) on various sub-samples of the dataset
    and uses averaging to improve the predictive accuracy and control
    over-fitting. The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    ``bootstrap=True`` (default).

    In each randomized survival tree, the quality of a split is measured by
    the log-rank splitting rule.

    Compared to ``RandomSurvivalForest``, randomness goes one step
    further in the way splits are computed. As in
    ``RandomSurvivalForest``, a random subset of candidate features is
    used, but instead of looking for the most discriminative thresholds,
    thresholds are drawn at random for each candidate feature and the best of
    these randomly-generated thresholds is picked as the splitting rule.

    Parameters
    ----------
    n_estimators : integer, optional, default: 100
        The number of trees in the forest.

    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional, default: None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.

    random_state : int, RandomState instance or None, optional, default: None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    Attributes
    ----------
    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.
    """

    _tags = {
        "authors": ["sebp", "fkiraly"],  # sebp credit for interfaced estimator=
    }

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        warm_start=False,
        max_samples=None,
        verbose=0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.max_samples = max_samples
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def _get_sksurv_class(self):
        """Getter of the sksurv class to be used for the adapter."""
        from sksurv.ensemble.forest import ExtraSurvivalTrees

        return ExtraSurvivalTrees

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}

        params2 = {
            "n_estimators": 99,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "log2",
            "max_leaf_nodes": 10,
            "oob_score": True,
        }

        return [params1, params2]
