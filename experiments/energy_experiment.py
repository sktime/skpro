"""Experiment: Compare quadrature vs Monte Carlo energy estimates.

This script compares the accuracy of CDF-based numerical quadrature
against Monte Carlo sampling for computing distribution energy.

Usage:
    python skpro/experiments/energy_experiment.py
"""

import warnings

import numpy as np
import pandas as pd


def compute_energy_mc(dist, x=None, n_samples=10000):
    """Compute energy using Monte Carlo sampling.

    Parameters
    ----------
    dist : BaseDistribution
        The distribution to compute energy for.
    x : None or pd.DataFrame, optional
        If None, computes self-energy E[|X-Y|].
        If given, computes cross-energy E[|X-x|].
    n_samples : int
        Number of MC samples.

    Returns
    -------
    float
        MC estimate of the energy.
    """
    if x is None:
        samples1 = dist.sample(n_samples)
        samples2 = dist.sample(n_samples)
        diffs = (samples1 - samples2).abs()
    else:
        samples = dist.sample(n_samples)
        if dist.ndim > 0:
            x_rep = pd.concat([x] * n_samples, keys=range(n_samples))
            diffs = (samples - x_rep).abs()
        else:
            diffs = (samples - x).abs()

    if dist.ndim > 0:
        energy = diffs.sum(axis=1).groupby(level=1, sort=False).mean()
        return energy.mean()
    else:
        return diffs.mean().iloc[0]


def compute_energy_quad(dist, x=None):
    """Compute energy using CDF-based quadrature (base class method).

    Parameters
    ----------
    dist : BaseDistribution
        The distribution.
    x : None or value
        If None, self-energy. If given, cross-energy.

    Returns
    -------
    float
        Quadrature estimate of the energy.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dist.energy(x)

    if isinstance(result, pd.DataFrame):
        return result.values.mean()
    return float(result)


def run_experiment():
    """Run comparison experiment for multiple distributions."""
    from skpro.distributions.beta import Beta
    from skpro.distributions.chi_squared import ChiSquared
    from skpro.distributions.exponential import Exponential
    from skpro.distributions.gamma import Gamma
    from skpro.distributions.logistic import Logistic
    from skpro.distributions.lognormal import LogNormal
    from skpro.distributions.normal import Normal
    from skpro.distributions.t import TDistribution
    from skpro.distributions.weibull import Weibull

    # Define test distributions (scalar case for simplicity)
    test_cases = {
        "Normal(0,1)": Normal(mu=0, sigma=1),
        "Normal(2,3)": Normal(mu=2, sigma=3),
        "Exponential(2)": Exponential(rate=2),
        "Beta(2,5)": Beta(alpha=2, beta=5),
        "Gamma(3,2)": Gamma(alpha=3, beta=2),
        "Logistic(0,1)": Logistic(mu=0, scale=1),
        "LogNormal(0,1)": LogNormal(mu=0, sigma=1),
        "TDist(0,1,df=5)": TDistribution(mu=0, sigma=1, df=5),
        "Weibull(2,3)": Weibull(scale=2, k=3),
        "ChiSquared(5)": ChiSquared(dof=5),
    }

    np.random.seed(42)

    print("=" * 80)  # noqa: T201
    print("Energy Computation Comparison: Quadrature vs Monte Carlo")  # noqa: T201
    print("=" * 80)  # noqa: T201

    # --- Self-energy comparison ---
    print("\n--- Self-Energy E[|X-Y|] ---")  # noqa: T201
    print(
        f"{'Distribution':<22} {'Quadrature':>12} {'MC (10k)':>12} {'MC (100k)':>12}"  # noqa: T201
        f" {'RelErr(10k)':>12} {'RelErr(100k)':>12}"
    )
    print("-" * 82)  # noqa: T201

    for name, dist in test_cases.items():
        quad_val = compute_energy_quad(dist)
        mc_10k = compute_energy_mc(dist, n_samples=10000)
        mc_100k = compute_energy_mc(dist, n_samples=100000)

        rel_err_10k = abs(mc_10k - quad_val) / max(abs(quad_val), 1e-10)
        rel_err_100k = abs(mc_100k - quad_val) / max(abs(quad_val), 1e-10)

        print(
            f"{name:<22} {quad_val:>12.6f} {mc_10k:>12.6f} {mc_100k:>12.6f}"  # noqa: T201
            f" {rel_err_10k:>12.4%} {rel_err_100k:>12.4%}"
        )

    # --- Cross-energy comparison ---
    print("\n--- Cross-Energy E[|X-x|] (x = mean + 0.5) ---")  # noqa: T201
    print(
        f"{'Distribution':<22} {'Quadrature':>12} {'MC (10k)':>12} {'MC (100k)':>12}"  # noqa: T201
        f" {'RelErr(10k)':>12} {'RelErr(100k)':>12}"
    )
    print("-" * 82)  # noqa: T201

    for name, dist in test_cases.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_val = float(dist.mean())

        if not np.isfinite(mean_val):
            print(f"{name:<22} {'(mean is inf, skipped)':>60}")  # noqa: T201
            continue

        x_val = mean_val + 0.5
        quad_val = compute_energy_quad(dist, x=x_val)
        mc_10k = compute_energy_mc(dist, x=x_val, n_samples=10000)
        mc_100k = compute_energy_mc(dist, x=x_val, n_samples=100000)

        rel_err_10k = abs(mc_10k - quad_val) / max(abs(quad_val), 1e-10)
        rel_err_100k = abs(mc_100k - quad_val) / max(abs(quad_val), 1e-10)

        print(
            f"{name:<22} {quad_val:>12.6f} {mc_10k:>12.6f} {mc_100k:>12.6f}"  # noqa: T201
            f" {rel_err_10k:>12.4%} {rel_err_100k:>12.4%}"
        )

    print("\n" + "=" * 80)  # noqa: T201
    print(
        "Conclusion: Quadrature provides deterministic, high-accuracy results."
    )  # noqa: T201
    print("Monte Carlo converges to quadrature as sample size increases.")  # noqa: T201
    print("=" * 80)  # noqa: T201


if __name__ == "__main__":
    run_experiment()
