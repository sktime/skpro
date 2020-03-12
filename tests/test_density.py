import numpy as np
import random

from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

from skpro.baselines.adapters.density_adapter import KernelDensityAdapter, EmpiricalDensityAdapter
np.random.seed(1)


def test_kernel_density_adapter():
    # Bayesian test sample
    loc, scale = 5, 10
    sample = np.random.normal(loc=loc, scale=scale, size=500)

    # Initialise adapter
    adapter = KernelDensityAdapter()
    adapter(sample)
    X = [random.uniform(-scale, scale) for _ in range(100)]

    # PDF
    pdf = adapter.pdf(X)
    assert (abs(pdf - norm.pdf(X, loc=loc, scale=scale)) < 0.05).all()
    
    #CDF
    cdf = adapter.cdf(X)
    assert (abs(cdf - norm.cdf(X, loc=loc, scale=scale)) < 0.05).all()


def test_empirical_density_adapter():
    # Bayesian test sample
    loc, scale = 5, 10
    sample = np.random.normal(loc=loc, scale=scale, size=5000)

    # Initialise adapter
    adapter = EmpiricalDensityAdapter(bandwidth = 0.2)
    adapter(sample)
    X = [random.uniform(-scale, scale) for _ in range(100)]
    
    # PDF
    pdf = adapter.pdf(X)
    assert (abs(pdf - norm.pdf(X, loc=loc, scale=scale)) < 0.05).all()
    

    #CDF
    cdf = adapter.cdf(X)
    assert (abs(cdf - norm.cdf(X, loc=loc, scale=scale)) < 0.05).all()


def test_plot():
    # plotting grid
    x_grid = np.linspace(-4.5, 3.5, 1000)
    
    # bimodal 1D distribution
    np.random.seed(0)
    x = np.concatenate([norm(-1, 1.).rvs(400), norm(1, 0.3).rvs(100)])   
    pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) + 0.2 * norm(1, 0.3).pdf(x_grid))
    
    # Plot the three kernel density estimates
    kadapter = KernelDensityAdapter()
    kadapter(x)
    
    eadapter = EmpiricalDensityAdapter()
    eadapter(x)
    
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(13, 3))
    fig.subplots_adjust(wspace=0)
    
    pdf = pdf_true
    ax[0].plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax[0].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax[0].set_title('true-pdf')
    ax[0].set_xlim(-4.5, 3.5)

    pdf = kadapter.pdf(x_grid)
    ax[1].plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax[1].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax[1].set_title('Scikit-learn - kernel')
    ax[1].set_xlim(-4.5, 3.5)
    
    pdf = eadapter.pdf(x_grid)
    ax[2].plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax[2].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
    ax[2].set_title('empirical')
    ax[2].set_xlim(-4.5, 3.5)
    
    from IPython.display import HTML
    HTML("<font color='#666666'>Gray = True underlying distribution</font><br>"
     "<font color='6666ff'>Blue = KDE model distribution (500 pts)</font>")

   
if __name__ == "__main__":
    test_kernel_density_adapter()
    test_empirical_density_adapter()
    test_plot()

    
    
    
    
    
    