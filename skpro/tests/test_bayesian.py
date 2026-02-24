import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from skpro.regression.bayesian import BayesianConjugateLinearRegressor

def test_bayesian_incremental_learning(show_plots=True):
    # --- 1. SETUP DATA ---
    X_init = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y_init = pd.Series([2.1, 4.0, 5.9])

    X_new = pd.DataFrame({"f": [7.0, 8.0, 9.0]})
    y_new = pd.Series([14.5, 15.8, 18.2])

    X_test_point = pd.DataFrame({"f": [5.0]})
    X_range = pd.DataFrame({"f": np.linspace(0, 10, 100)})

    # --- 2. INITIALIZE & FIT ---
    # Higher noise_precision makes the model trust data more than the prior
    model = BayesianConjugateLinearRegressor(coefs_prior_cov=[[10.0]], noise_precision=5.0)
    
    model.fit(X_init, y_init)
    
    # Predict initial state
    dist1_point = model.predict_proba(X_test_point)
    mu1_p, sig1_p = dist1_point.mean().values[0][0], dist1_point.sigma.values[0][0]
    
    dist1_range = model.predict_proba(X_range)
    mu1_r, sig1_r = dist1_range.mean().values.flatten(), dist1_range.sigma.values.flatten()

    # --- 3. INCREMENTAL UPDATE ---
    model.update(X_new, y_new)
    
    # Predict updated state
    dist2_point = model.predict_proba(X_test_point)
    mu2_p, sig2_p = dist2_point.mean().values[0][0], dist2_point.sigma.values[0][0]
    
    dist2_range = model.predict_proba(X_range)
    mu2_r, sig2_r = dist2_range.mean().values.flatten(), dist2_range.sigma.values.flatten()

    # --- 4. ASSERTIONS & REPORTING ---
    print("\n" + "="*50)
    print("BAYESIAN INCREMENTAL LEARNING REPORT")
    print("="*50)
    print(f"Test Point (f=5.0) Results:")
    print(f"  Initial -> Mean: {mu1_p:.4f}, Sigma: {sig1_p:.4f}")
    print(f"  Updated -> Mean: {mu2_p:.4f}, Sigma: {sig2_p:.4f}")
    
    reduction = ((sig1_p - sig2_p) / sig1_p) * 100
    print(f"  Uncertainty Reduction: {reduction:.2f}%")
    
    # Functional check: Sigma should decrease as we provide more data
    assert sig2_p < sig1_p, "FAILURE: Uncertainty did not reduce after update!"
    print("SUCCESS: Uncertainty reduced correctly.")
    print("-" * 50)
    print("Learned Coefficients (Posterior):")
    print(f"  Mean Weights:\n{model._coefs_posterior_mu}")
    print("="*50 + "\n")

    # --- 5. VISUALIZATION ---
    if show_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Probability Density at X=5.0
        x_axis = np.linspace(mu1_p - 4*sig1_p, mu2_p + 4*sig1_p, 500)
        ax1.plot(x_axis, norm.pdf(x_axis, mu1_p, sig1_p), label='Initial Fit', color='#FF69B4', lw=2)
        ax1.plot(x_axis, norm.pdf(x_axis, mu2_p, sig2_p), label='Updated Model', color='#4B0082', lw=3)
        ax1.fill_between(x_axis, norm.pdf(x_axis, mu2_p, sig2_p), color='#4B0082', alpha=0.2)
        ax1.set_title("Density Narrowing at X=5.0")
        ax1.set_xlabel("Predicted y")
        ax1.set_ylabel("Probability Density")
        ax1.legend()

        # Plot 2: Regression Line with 95% CI Ribbons
        # Initial Fit
        ax2.plot(X_range, mu1_r, color='#FF69B4', linestyle='--', label='Initial Mean')
        ax2.fill_between(X_range.values.flatten(), mu1_r - 1.96*sig1_r, mu1_r + 1.96*sig1_r, color='#FF69B4', alpha=0.1)
        # Updated Model
        ax2.plot(X_range, mu2_r, color='#4B0082', lw=2, label='Updated Mean')
        ax2.fill_between(X_range.values.flatten(), mu2_r - 1.96*sig2_r, mu2_r + 1.96*sig2_r, color='#4B0082', alpha=0.2)
        # Data Points
        ax2.scatter(X_init, y_init, color='#FF69B4', edgecolors='k', label='Batch 1')
        ax2.scatter(X_new, y_new, color='#4B0082', edgecolors='k', label='Batch 2 (Update)')

        ax2.set_title("Regression Confidence Ribbons (95% CI)")
        ax2.set_xlabel("Feature f")
        ax2.set_ylabel("Target y")
        ax2.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_bayesian_incremental_learning(show_plots=True)