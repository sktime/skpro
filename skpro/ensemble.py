import numpy as np

from sklearn.ensemble import BaggingRegressor as BaseBaggingRegressor

from .base import ProbabilisticEstimator


class BaggingRegressor(BaseBaggingRegressor, ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def __init__(self, estimator, X, distributions, n_estimators):
            super().__init__(estimator, X)
            self.distributions = distributions
            self.n_estimators = n_estimators

        def point(self):
            return NotImplemented

        def std(self):
            return NotImplemented

        def pdf(self, x):
            ## the pdf(x) is used to calculate the loss
            # the bagged pdf should be calculated by
            # simple averaging the predicted pdfs
            reduced = []
            for distribution in self.distributions[0]:
                pdf = getattr(distribution, 'pdf', False)
                if not callable(pdf):
                    raise AttributeError('%s does not exist' % pdf)

                reduced.append(pdf(x))

            return np.mean(reduced, axis=0)


    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        # check_is_fitted(self, "estimators_features_")
        # Check data
        # X = check_array(X, accept_sparse=['csr', 'csc'])

        # Parallel loop
        from sklearn.ensemble.base import _partition_estimators
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        def _parallel_predict_regression(estimators, estimators_features, X):
            """Private function used to compute predictions within a job."""
            return [
                estimator.predict(X[:, features])
                for estimator, features in zip(estimators, estimators_features)
            ]

        all_y_hat = [
            _parallel_predict_regression(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X
            ) for i in range(n_jobs)
        ]

        # Reduce
        return self._distribution()(self, X, all_y_hat, n_estimators)

    def __str__(self, describer=str):
        return 'BaggingRegressor(' + describer(self.base_estimator) + ')'

    def __repr__(self):
        return self.__str__(repr)