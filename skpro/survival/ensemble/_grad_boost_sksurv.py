"""Interface adapters to scikit-survival gradient boosting models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.sksurv import _SksurvAdapter
from skpro.survival.base import BaseSurvReg


class SurvGradBoostSkSurv(_SksurvAdapter, BaseSurvReg):
    """Gradient-boosted survival trees with proportional hazards loss, from sksurv.

    Direct interface to ``sksurv.ensemble.boosting.GradientBoostingSurvivalAnalysis``.

    In each stage, a regression tree is fit on the negative gradient
    of the loss function.

    For more details on gradient boosting see [1]_ and [2]_. If `loss='coxph'`,
    the partial likelihood of the proportional hazards model is optimized as
    described in [3]_. If `loss='ipcwls'`, the accelerated failure time model with
    inverse-probability of censoring weighted least squares error is optimized as
    described in [4]_. When using a non-zero `dropout_rate`, regularization is
    applied during training following [5]_.

    Parameters
    ----------
    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares
        error.

    learning_rate : float, optional, default: 0.1
        learning rate shrinks the contribution of each tree by ``learning_rate``.
        There is a trade-off between ``learning_rate`` and ``n_estimators``.
        Values must be in the range ``[0.0, inf)``.

    n_estimators : int, default: 100
        The number of regression trees to create. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range ``[1, inf)``.

    subsample : float, optional, default: 1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. ``subsample`` interacts with the parameter `n_estimators`.
        Choosing ``subsample < 1.0`` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range ``(0.0, 1.0]``.

    criterion : string, optional, "squared_error" or "friedman_mse" (default)
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "squared_error" for mean squared error.
        The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : int or float, optional, default: 2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range ``[2, inf)``.
        - If float, values must be in the range ``(0.0, 1.0]`` and ``min_samples_split``
          will be ``ceil(min_samples_split * n_samples)``.

    min_samples_leaf : int or float, default: 1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range ``[1, inf)``.
        - If float, values must be in the range ``(0.0, 1.0)`` and ``min_samples_leaf``
          will be ``ceil(min_samples_leaf * n_samples)``.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when `sample_weight` is not provided.
        Values must be in the range ``[0.0, 0.5]``.

    max_depth : int or None, optional, default: 3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.
        If int, values must be in the range ``[1, inf)``.

    min_impurity_decrease : float, optional, default: 0.
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range ``[1, inf)``.
        - If float, values must be in the range ``(0.0, 1.0]`` and the features
          considered at each split will
          be ``max(1, int(max_features * n_features_in_))``.
        - If 'sqrt', then ``max_features=sqrt(n_features)``.
        - If 'log2', then ``max_features=log2(n_features)``.
        - If None, then ``max_features=n_features``.

        Choosing ``max_features < n_features`` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If ``None``, then unlimited number of leaf nodes.

    validation_fraction : float, default: 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range ``(0.0, 1.0)``.
        Only used if ``n_iter_no_change`` is set to an integer.

    n_iter_no_change : int, default: None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        Values must be in the range ``[1, inf)``.

    tol : float, default: 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range ``[0.0, inf)``.

    dropout_rate : float, optional, default: 0.0
        If larger than zero, the residuals at each iteration are only computed
        from a random subset of base learners. The value corresponds to the
        percentage of base learners that are dropped. In each iteration,
        at least one base learner is dropped. This is an alternative regularization
        to shrinkage, i.e., setting ``learning_rate < 1.0``.
        Values must be in the range ``[0.0, 1.0)``.

    ccp_alpha : non-negative float, optional, default: 0.0.
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range ``[0.0, inf)``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range ``[0, inf)``.

    random_state : int seed, RandomState instance, or None, default: None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split.
        It also controls the random splitting of the training data to obtain a
        validation set if ``n_iter_no_change`` is not None.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

    feature_importances_ : ndarray, shape = (n_features,)
        The feature importances (the higher, the more important the feature).

    estimators_ : ndarray of DecisionTreeRegressor, shape = (n_estimators, 1)
        The collection of fitted sub-estimators.

    train_score_ : ndarray, shape = (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    oob_improvement_ : ndarray, shape = (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    References
    ----------
    .. [1] J. H. Friedman, "Greedy function approximation: A gradient boosting machine,"
           The Annals of Statistics, 29(5), 1189–1232, 2001.
    .. [2] J. H. Friedman, "Stochastic gradient boosting,"
           Computational Statistics & Data Analysis, 38(4), 367–378, 2002.
    .. [3] G. Ridgeway, "The state of boosting,"
           Computing Science and Statistics, 172–181, 1999.
    .. [4] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
           "Survival ensembles", Biostatistics, 7(3), 355-73, 2006.
    .. [5] K. V. Rashmi and R. Gilad-Bachrach,
           "DART: Dropouts meet multiple additive regression trees,"
           in 18th International Conference on Artificial Intelligence and Statistics,
           2015, 489–497.
    """

    _tags = {
        "authors": ["sebp", "fkiraly"],  # sebp credit for interfaced estimator=
    }

    def __init__(
        self,
        loss="coxph",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        max_features=None,
        max_leaf_nodes=None,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        dropout_rate=0.0,
        ccp_alpha=0.0,
        verbose=0,
        random_state=None,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.dropout_rate = dropout_rate
        self.ccp_alpha = ccp_alpha
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def _get_sksurv_class(self):
        """Getter of the sksurv class to be used for the adapter."""
        from sksurv.ensemble.boosting import GradientBoostingSurvivalAnalysis

        return GradientBoostingSurvivalAnalysis

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
            "learning_rate": 0.2,
            "n_estimators": 99,
            "subsample": 0.9,
            "criterion": "squared_error",
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_depth": 4,
            "max_features": "sqrt",
            "tol": 1e-3,
            "dropout_rate": 0.01,
        }

        return [params1, params2]


class SurvGradBoostCompSkSurv(_SksurvAdapter, BaseSurvReg):
    r"""Survival Gradient boosting component-wise least squares, from sksurv.

    Direct interface to ``ComponentwiseGradientBoostingSurvivalAnalysis``
    from ``sksurv.ensemble.boosting``.

    Parameters
    ----------
    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares
        error.

    learning_rate : float, optional, default: 0.1
        learning rate shrinks the contribution of each base learner by `learning_rate`.
        There is a trade-off between `learning_rate` and `n_estimators`.
        Values must be in the range ``[0.0, inf)``.

    n_estimators : int, default: 100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range ``[1, inf)``.

    subsample : float, optional, default: 1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. ``subsample`` interacts with the parameter ``n_estimators``.
        Choosing ``subsample < 1.0`` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range ``(0.0, 1.0]``.

    dropout_rate : float, optional, default: 0.0
        If larger than zero, the residuals at each iteration are only computed
        from a random subset of base learners. The value corresponds to the
        percentage of base learners that are dropped. In each iteration,
        at least one base learner is dropped. This is an alternative regularization
        to shrinkage, i.e., setting ``learning_rate < 1.0``.
        Values must be in the range ``[0.0, 1.0)``.

    random_state : int seed, RandomState instance, or None, default: None
        The seed of the pseudo random number generator to use when
        shuffling the data.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while.
        Values must be in the range ``[0, inf)``.

    Attributes
    ----------
    coef_ : array, shape = (n_features + 1,)
        The aggregated coefficients. The first element `coef\_[0]` corresponds
        to the intercept. If loss is `coxph`, the intercept will always be zero.

    estimators_ : list of base learners
        The collection of fitted sub-estimators.

    train_score_ : array, shape = (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    oob_improvement_ : array, shape = (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    References
    ----------
    .. [1] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
           "Survival ensembles", Biostatistics, 7(3), 355-73, 2006
    """

    _tags = {
        "authors": ["sebp", "fkiraly"],  # sebp credit for interfaced estimator=
    }

    def __init__(
        self,
        loss="coxph",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        dropout_rate=0,
        verbose=0,
        random_state=None,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def _get_sksurv_class(self):
        """Getter of the sksurv class to be used for the adapter."""
        from sksurv.ensemble.boosting import (
            ComponentwiseGradientBoostingSurvivalAnalysis,
        )

        return ComponentwiseGradientBoostingSurvivalAnalysis

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
            "learning_rate": 0.2,
            "n_estimators": 99,
            "subsample": 0.9,
            "dropout_rate": 0.01,
        }

        return [params1, params2]
