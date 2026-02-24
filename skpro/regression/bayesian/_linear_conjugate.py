import numpy as np
import pandas as pd
from skpro.distributions import Normal
from skpro.regression.base import BaseProbaRegressor

class BayesianConjugateLinearRegressor(BaseProbaRegressor):
    def __init__(self, coefs_prior_cov, coefs_prior_mu=None, noise_precision=1.0):
        print("--- NEW MODEL VERSION INITIALIZED ---")
        self.coefs_prior_cov = coefs_prior_cov
        self.coefs_prior_mu = coefs_prior_mu
        self.noise_precision = noise_precision
        super().__init__()

    def _fit(self, X, y):
        self._y_cols = y.columns
        n_features = X.shape[1]
        mu_0 = np.array(self.coefs_prior_mu).reshape(-1, 1) if self.coefs_prior_mu is not None else np.zeros((n_features, 1))
        self._coefs_posterior_mu, self._coefs_posterior_cov = self._perform_bayesian_inference(X, y, mu_0, np.asarray(self.coefs_prior_cov))
        return self

    def _predict_proba(self, X):
        # Numeric extraction to avoid string errors
        X_arr = pd.to_numeric(pd.Series(np.asarray(X).flatten()), errors='coerce').dropna().values
        X_arr = X_arr.reshape(-1, self._coefs_posterior_mu.shape[0])
        
        pred_mu = X_arr @ self._coefs_posterior_mu
        beta = float(self.noise_precision)
        
        # 
        # Variance calculation using the CURRENT posterior covariance
        pred_var = np.sum((X_arr @ self._coefs_posterior_cov) * X_arr, axis=1) + (1.0 / beta)
        
        return Normal(mu=pred_mu.reshape(-1, 1), sigma=np.sqrt(pred_var).reshape(-1, 1), columns=self._y_cols, index=X.index)

    def _perform_bayesian_inference(self, X, y, mu_prior, cov_prior):
        def clean(data, c):
            f = pd.to_numeric(pd.Series(np.asarray(data).flatten()), errors='coerce').dropna().values
            return f.reshape(-1, c)

        X_mat, y_mat = clean(X, mu_prior.shape[0]), clean(y, 1)
        beta = float(self.noise_precision)
        
        # Bayesian Math: S_n = (S_0^-1 + beta * X.T @ X)^-1
        prec_prior = np.linalg.inv(cov_prior)
        prec_post = prec_prior + beta * (X_mat.T @ X_mat)
        cov_post = np.linalg.inv(prec_post)
        mu_post = cov_post @ (prec_prior @ mu_prior + beta * (X_mat.T @ y_mat))
        
        return mu_post, cov_post

    def _update(self, X, y):
        print("\n>>> HARD UPDATE EXECUTED <<<")
        # Overwrite the state directly
        self._coefs_posterior_mu, self._coefs_posterior_cov = self._perform_bayesian_inference(
            X, y, self._coefs_posterior_mu, self._coefs_posterior_cov
        )
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"coefs_prior_cov": np.eye(1), "noise_precision": 1.0}