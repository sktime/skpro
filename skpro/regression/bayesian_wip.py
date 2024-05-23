"""
Note: this is WIP; to be converted into an skpro class
"""

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

class BayesianLinearRegression:
    def __init__(self):
        self.model = None
        self.trace = None
        self.fitted = False

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "X must be a pd.DataFrame!"
        assert isinstance(y, pd.DataFrame), "y must be a pd.DataFrame!"
        assert len(y.columns) == 1, "y must have only one column!"
        self.X = X
        self.y = y
        self.y_vals = y.values[:,0] # we need a 1-dimensional array for compatibility with pymc 
        self.X_cols = X.columns
        self.y_cols = y.columns

        with pm.Model() as self.model:
            # Mutable data containers
            X_data = pm.MutableData("X", self.X, dims = ("obs_id", "pred_id"))
            y_data = pm.MutableData("y", self.y_vals, dims = ("obs_id"))

            # Priors for unknown model parameters
            self.intercepts = pm.Normal("intercepts", mu=0, sigma=1)
            self.slopes = pm.Normal("slopes", mu=0, sigma=1, dims=("pred_id"))
            self.sigma = pm.HalfNormal("sigma", sigma=1)

            # Expected value of outcome
            self.mu = pm.Deterministic("mu", self.intercepts + pm.math.dot(X_data, self.slopes))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal("y_obs", mu=self.mu, sigma=self.sigma, observed=y_data, dims =("obs_id"))

            # Sample from the posterior
            self.trace = pm.sample(
                draws=2000,            
                tune=1500,             
                chains=1,              
                random_seed=42,             
                target_accept=0.90, # Target acceptance probability; higher value leads to higher accuracy but slower sampling
                return_inferencedata=True,  # Return an InferenceData object 
                progressbar=True            
            )

        self.fitted = True

    def predict_proba(self, X_new):
        """
        Get the full posterior predictive distribution samples as xarray DataArray
        """
        if not self.fitted:
            raise RuntimeError("The model must be fitted before making predictions.")

        assert isinstance(X_new, pd.DataFrame), "X must be a pd.DataFrame"
        assert X_new.columns.equals(self.X_cols), f"The columns of X must be the same as the columns of the training data: {self.X_cols}"

        with self.model:
            # Set the X_new to be the 'X' variable and then sample
            pm.set_data({"X": X_new})
            self.trace.extend(pm.sample_posterior_predictive(self.trace, random_seed=42))
            return self.trace.posterior_predictive["y_obs"] # Note: returns y_obs as xarray.core.dataarray.DataArray containing the posterior samples

    def predict(self, X_new):
        """
        Get the mean of the posterior predictive distribution
        """
        y_pred_proba = self.predict_proba(X_new)
        y_pred = y_pred_proba.mean(dim = ["chain", "draw"]).to_dataframe()
        y_pred.columns = self.y_cols
        return y_pred

    def predict_quantile(self, X_new, alpha):
        index = X_new.index
        columns = pd.MultiIndex.from_product(
            [self.y_cols, alpha],
        )
        predict_proba = self.predict_proba(X_new)

        values = []
        for a_ in alpha:
          val_ = predict_proba.quantile(a_, dim=["chain", "draw"]).to_numpy()
          values.append(val_)
        values = np.stack(values).T
        quantiles = pd.DataFrame(values, index=index, columns=columns)
        return quantiles

    def visualize_model(self):
        """
        Use graphviz to visualize the composition of the model
        """
        if not self.fitted:
            raise RuntimeError("The model must be fitted before visualization can be done.")
        return pm.model_to_graphviz(self.model)

    def plot_posterior_predictive(self, X_new):
        if not self.fitted:
            raise RuntimeError("The model must be fitted before plotting predictions.")

        quantile_df = model.predict_quantile(X_new, alpha = [0.25, 0.5, 0.75])
        plt.plot(quantile_df.index, quantile_df["target"][0.50], label='0.50 Quantile', color='blue')
        plt.fill_between(quantile_df.index, quantile_df["target"][0.25], quantile_df["target"][0.75], color='blue', alpha=0.2, label='0.25-0.75 Quantile')

        plt.xlabel('Index')
        plt.ylabel('Target')
        plt.title('Quantiles with Shading')
        plt.legend()
        plt.show()
