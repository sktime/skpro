# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Mixture Density Network probabilistic regressor."""

__author__ = ["joshdunnlime"]

from skpro.distributions.normal_mixture import NormalMixture
from skpro.regression.base import BaseProbaRegressor


class MDNRegressor(BaseProbaRegressor):
    """Mixture Density Network probabilistic regressor.

    Wraps a configurable PyTorch Mixture Density Network (MDN) as an skpro
    probabilistic regressor. The MDN outputs a mixture of Gaussians distribution
    with per-sample mixing weights, means, and standard deviations.

    Inspired by `scikit-mdn <https://github.com/koaning/scikit-mdn>`_.

    For noise-based regularization motivation and schedules, see:
    `Adding Noise to the Inputs of a Model Trained with a Regularized Objective
    Function <https://arxiv.org/pdf/1907.08982>`_.

    Parameters
    ----------
    n_gaussians : int, optional, default=5
        Number of Gaussian mixture components.

    hidden_dims : list of int, optional, default=None
        Sizes of hidden layers in the backbone network. If None, defaults
        to ``[64, 32]``. Each entry creates one fully-connected hidden layer.

    activation : str, class, instance, or callable, optional, default="relu"
        Activation function for hidden layers.
        String options: ``"relu"``, ``"elu"``, ``"softplus"``, ``"tanh"``.
        Users may also pass:
        a ``torch.nn.Module`` class, a ``torch.nn.Module`` instance,
        or a zero-argument callable returning a ``torch.nn.Module``.

    epochs : int, optional, default=1000
        Number of training epochs.

    lr : float, optional, default=0.01
        Learning rate for the optimiser.

    weight_decay : float, optional, default=0.0
        L2 regularisation coefficient (weight decay) for the optimiser.

    optimizer : str or class, optional, default="SOAP"
        Optimizer used for training.

        * ``"SOAP"``: `SOAP` from ``pytorch_optimizer``
        * ``"ADAM"``: ``torch.optim.Adam``

        Alternatively, users can pass a ``torch.optim.Optimizer`` class directly.

    optimizer_kwargs : dict or None, optional, default=None
        Additional keyword arguments forwarded to the optimizer constructor.

    input_noise_std : float, optional, default=0.0
        Base standard deviation :math:`h_x` of Gaussian noise added to ``X`` during
        training only. Set to ``0.0`` to disable input noise regularization on ``X``.

    target_noise_std : float, optional, default=0.0
        Base standard deviation :math:`h_y` of Gaussian noise added to ``y`` during
        training only. Set to ``0.0`` to disable input noise regularization on ``y``.

    noise_schedule : str, optional, default="constant"
        Schedule for scaling noise intensity as a function of sample size ``n`` and
        total dimensionality ``d = d_x + d_y``.

        * ``"constant"``: no scaling, uses ``h_x`` and ``h_y`` directly
        * ``"rule_of_thumb"``: scales by :math:`n^{-1/(4+d)}`
        * ``"sqrt_decay"``: scales by :math:`n^{-1/(1+d)}`

    batch_size : int or None, optional, default=None
        Mini-batch size for training. If None, full-batch training is used
        (all data in one pass per epoch, like scikit-mdn). Set to an integer
        for mini-batch training via ``torch.utils.data.DataLoader``.

    device : str, optional, default="auto"
        Device to run the PyTorch model on.
        ``"auto"`` detects CUDA or MPS if available, otherwise falls back to CPU.
        Other options: ``"cpu"``, ``"cuda"``, ``"mps"``.

    random_state : int or None, optional, default=None
        Seed for reproducibility. Controls ``torch.manual_seed`` and
        ``numpy.random.seed`` during fitting.

    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> from skpro.regression.mdn import MDNRegressor
    >>> import pandas as pd

    >>> # Generate moons dataset
    >>> X_arr, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
    >>> X = pd.DataFrame(X_arr[:, 0].reshape(-1, 1), columns=['x'])
    >>> y = pd.DataFrame(X_arr[:, 1].reshape(-1, 1), columns=['y'])

    >>> # Train MDN
    >>> mdn = MDNRegressor(n_gaussians=2, hidden_dims=[10], epochs=50, random_state=42)
    >>> mdn.fit(X, y)
    MDNRegressor(...)
    >>> y_pred = mdn.predict_proba(X[:5])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["joshdunnlime"],
        "python_dependencies": ["torch", "pytorch_optimizer"],
        #
        # estimator tags
        # --------------
        "capability:multioutput": True,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        n_gaussians=5,
        hidden_dims=None,
        activation="relu",
        epochs=1000,
        lr=0.01,
        weight_decay=0.0,
        optimizer="SOAP",
        optimizer_kwargs=None,
        input_noise_std=0.0,
        target_noise_std=0.0,
        noise_schedule="constant",
        batch_size=None,
        device="auto",
        random_state=None,
    ):
        self.n_gaussians = n_gaussians
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.input_noise_std = input_noise_std
        self.target_noise_std = target_noise_std
        self.noise_schedule = noise_schedule
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        super().__init__()

    def _noise_scale(self, n_samples, total_dim):
        """Return schedule-based multiplier for noise regularization.

        Noise Regularization for Conditional Density Estimation by Rothfuss
        et al - https://arxiv.org/pdf/1907.08982
        """
        schedule = self.noise_schedule

        if schedule == "constant":
            return 1.0
        if schedule == "rule_of_thumb":
            return n_samples ** (-1.0 / (4.0 + total_dim))
        if schedule == "sqrt_decay":
            return n_samples ** (-1.0 / (1.0 + total_dim))
        raise ValueError(
            "Unknown noise_schedule "
            f"'{schedule}'. Valid options: ['constant', 'rule_of_thumb', 'sqrt_decay']"
        )

    def _resolve_device(self):
        """Resolve the device to use for PyTorch tensors.

        Returns
        -------
        torch.device
            Resolved device.
        """
        import torch

        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _get_activation_factory(self):
        """Return a zero-argument activation factory for hidden layers.

        Returns
        -------
        callable
            Zero-argument callable returning a ``torch.nn.Module`` instance.
        """
        from copy import deepcopy
        from inspect import isclass

        import torch.nn as nn

        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "tanh": nn.Tanh,
        }

        activation = self.activation

        if isinstance(activation, str):
            act_name = activation.lower()
            if act_name not in activations:
                raise ValueError(
                    f"Unknown activation '{activation}'. "
                    f"Valid string options: {list(activations.keys())}"
                )
            return activations[act_name]

        if isclass(activation) and issubclass(activation, nn.Module):
            return activation

        if isinstance(activation, nn.Module):
            return lambda: deepcopy(activation)

        if callable(activation):
            probe = activation()
            if not isinstance(probe, nn.Module):
                raise TypeError(
                    "activation callable must return a torch.nn.Module instance."
                )
            return activation

        raise TypeError(
            "activation must be a string, torch.nn.Module class, "
            "torch.nn.Module instance, or zero-argument callable returning "
            "torch.nn.Module."
        )

    def _get_optimizer_cls(self):
        """Resolve optimizer class from string alias or user-provided class."""
        import torch

        optimizer = self.optimizer

        if isinstance(optimizer, str):
            opt_name = optimizer.upper()

            if opt_name == "ADAM":
                return torch.optim.Adam

            if opt_name == "SOAP":
                try:
                    from pytorch_optimizer import SOAP
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "optimizer='SOAP' requires package 'pytorch_optimizer'."
                    ) from exc

                return SOAP

            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Valid options: ['SOAP', 'ADAM'] "
                "or a torch.optim.Optimizer class."
            )

        if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            return optimizer

        raise TypeError(
            "optimizer must be either one of ['SOAP', 'ADAM'] or a "
            "torch.optim.Optimizer class."
        )

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        import numpy as np
        import torch

        self._y_cols = y.columns
        self._output_dim = len(y.columns)
        self._input_dim = X.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        input_noise_std = float(self.input_noise_std)
        target_noise_std = float(self.target_noise_std)
        if input_noise_std < 0:
            raise ValueError(
                "input_noise_std must be non-negative, but found "
                f"{self.input_noise_std}"
            )

        if target_noise_std < 0:
            raise ValueError(
                "target_noise_std must be non-negative, "
                f"but found {self.target_noise_std}"
            )

        n_samples = len(X)
        total_dim = self._input_dim + self._output_dim
        noise_scale = self._noise_scale(n_samples=n_samples, total_dim=total_dim)
        input_noise_std *= noise_scale
        target_noise_std *= noise_scale

        device = self._resolve_device()

        # convert to tensors
        X_arr = X.values.astype("float32")
        y_arr = y.values.astype("float32")
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=device)

        # build the model
        hidden_dims = self.hidden_dims if self.hidden_dims is not None else [64, 32]

        model = _build_mdn_model(
            input_dim=self._input_dim,
            hidden_dims=hidden_dims,
            output_dim=self._output_dim,
            n_gaussians=self.n_gaussians,
            activation_factory=self._get_activation_factory(),
        )

        model = model.to(device)  # type: ignore[attr-defined]

        optimizer_cls = self._get_optimizer_cls()

        optimizer_kwargs = (
            {} if self.optimizer_kwargs is None else dict(self.optimizer_kwargs)
        )

        optimiser = optimizer_cls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **optimizer_kwargs,
        )

        losses = []

        if self.batch_size is None:
            # full-batch training
            for _epoch in range(self.epochs):
                optimiser.zero_grad()
                if input_noise_std > 0:
                    X_t_train = X_t + torch.randn_like(X_t) * input_noise_std
                else:
                    X_t_train = X_t
                if target_noise_std > 0:
                    y_t_train = y_t + torch.randn_like(y_t) * target_noise_std
                else:
                    y_t_train = y_t
                pi, mu, sigma = model(X_t_train)
                loss = _mdn_loss(pi, mu, sigma, y_t_train)
                loss.backward()
                optimiser.step()
                losses.append(loss.item())
        else:
            # mini-batch training
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            for _epoch in range(self.epochs):
                epoch_loss = 0.0
                n_batches = 0

                for X_batch, y_batch in loader:
                    optimiser.zero_grad()
                    if input_noise_std > 0:
                        X_batch = X_batch + torch.randn_like(X_batch) * input_noise_std
                    if target_noise_std > 0:
                        y_batch = y_batch + torch.randn_like(y_batch) * target_noise_std
                    pi, mu, sigma = model(X_batch)
                    loss = _mdn_loss(pi, mu, sigma, y_batch)
                    loss.backward()
                    optimiser.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                losses.append(epoch_loss / max(n_batches, 1))

        self.model_ = model
        self.losses_ = losses
        self._device = device

        return self

    def _predict_proba(self, X):  # type: ignore[override]
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        import torch

        device = self._device

        X_arr = X.values.astype("float32")
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)

        self.model_.eval()
        with torch.no_grad():
            pi, mu, sigma = self.model_(X_t)

        pi_np = pi.cpu().numpy()  # (n_samples, n_gaussians)
        mu_np = mu.cpu().numpy()  # (n_samples, n_gaussians, n_outputs)
        sigma_np = sigma.cpu().numpy()  # (n_samples, n_gaussians, n_outputs)

        return NormalMixture(
            pi=pi_np,
            mu=mu_np,
            sigma=sigma_np,
            index=X.index,
            columns=self._y_cols,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance. ``create_test_instance`` uses the first (or only) dictionary
            in ``params``.
        """
        params0 = {
            "n_gaussians": 3,
            "hidden_dims": [8],
            "epochs": 5,
            "random_state": 42,
        }

        params1 = {
            "n_gaussians": 2,
            "hidden_dims": [4, 4],
            "epochs": 5,
            "batch_size": 10,
            "random_state": 42,
        }

        params2 = {
            "n_gaussians": 2,
            "hidden_dims": [8],
            "epochs": 5,
            "activation": "softplus",
            "random_state": 42,
        }

        params3 = {
            "n_gaussians": 2,
            "hidden_dims": [8],
            "epochs": 5,
            "input_noise_std": 0.05,
            "random_state": 42,
        }

        params4 = {
            "n_gaussians": 2,
            "hidden_dims": [8],
            "epochs": 5,
            "input_noise_std": 0.05,
            "target_noise_std": 0.02,
            "noise_schedule": "rule_of_thumb",
            "random_state": 42,
        }

        params5 = {
            "n_gaussians": 2,
            "hidden_dims": [8],
            "epochs": 5,
            "optimizer": "ADAM",
            "random_state": 42,
        }
        return [params0, params1, params2, params3, params4, params5]


def _build_mdn_model(
    input_dim, hidden_dims, output_dim, n_gaussians, activation_factory
):
    """Build a Mixture Density Network PyTorch module.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Sizes of hidden layers.
    output_dim : int
        Number of output dimensions.
    n_gaussians : int
        Number of Gaussian mixture components.
    activation_factory : callable
        Zero-argument callable returning a ``torch.nn.Module``.

    Returns
    -------
    _MixtureDensityNetwork
        PyTorch MDN module.
    """
    import torch.nn as nn

    _ensure_mdn_class()

    # build backbone layers
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation_factory())
        prev_dim = h_dim

    backbone = nn.Sequential(*layers)

    return _MixtureDensityNetwork(
        backbone=backbone,
        backbone_out_dim=prev_dim,
        output_dim=output_dim,
        n_gaussians=n_gaussians,
    )


class _MixtureDensityNetwork:
    """PyTorch Mixture Density Network module.

    This is a private class — not part of the public API.

    Parameters
    ----------
    backbone : torch.nn.Sequential
        The hidden layers of the network.
    backbone_out_dim : int
        Output dimension of the backbone.
    output_dim : int
        Number of output dimensions.
    n_gaussians : int
        Number of Gaussian mixture components.
    """

    pass  # defined dynamically to avoid top-level torch import


# We define the actual nn.Module class inside a function to avoid
# importing torch at module level (soft dependency pattern).
def _ensure_mdn_class():
    """Ensure the _MixtureDensityNetwork class is a proper nn.Module.

    This function is called at import time only when torch is available,
    and replaces the placeholder class with the real implementation.
    """
    global _MixtureDensityNetwork

    # if already constructed, skip
    if hasattr(_MixtureDensityNetwork, "forward"):
        return

    import torch.nn as nn
    import torch.nn.functional as F

    class _MixtureDensityNetworkImpl(nn.Module):
        """PyTorch MDN module implementation.

        Parameters
        ----------
        backbone : nn.Sequential
            Hidden layers.
        backbone_out_dim : int
            Output dimension of backbone.
        output_dim : int
            Number of output dimensions.
        n_gaussians : int
            Number of Gaussian components.
        """

        def __init__(self, backbone, backbone_out_dim, output_dim, n_gaussians):
            super().__init__()
            self.backbone = backbone
            self.n_gaussians = n_gaussians
            self.output_dim = output_dim

            # mixing weights head (shared across outputs)
            self.pi_layer = nn.Linear(backbone_out_dim, n_gaussians)
            # means head
            self.mu_layer = nn.Linear(backbone_out_dim, n_gaussians * output_dim)
            # std devs head
            self.sigma_layer = nn.Linear(backbone_out_dim, n_gaussians * output_dim)

        def forward(self, x):
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, input_dim)

            Returns
            -------
            pi : torch.Tensor of shape (batch_size, n_gaussians)
                Mixing weights (softmax-normalised).
            mu : torch.Tensor of shape (batch_size, n_gaussians, output_dim)
                Component means.
            sigma : torch.Tensor of shape (batch_size, n_gaussians, output_dim)
                Component standard deviations (positive, via softplus).
            """
            h = self.backbone(x)
            pi = F.softmax(self.pi_layer(h), dim=1)
            mu = self.mu_layer(h).view(-1, self.n_gaussians, self.output_dim)
            # clamp sigma to a minimum for numerical stability
            sigma = F.softplus(self.sigma_layer(h)).view(
                -1, self.n_gaussians, self.output_dim
            )
            sigma = sigma + 1e-6
            return pi, mu, sigma

    _MixtureDensityNetwork = _MixtureDensityNetworkImpl


def _mdn_loss(pi, mu, sigma, target):
    """Compute the MDN negative log-likelihood loss.

    Parameters
    ----------
    pi : torch.Tensor of shape (batch_size, n_gaussians)
        Mixing weights.
    mu : torch.Tensor of shape (batch_size, n_gaussians, output_dim)
        Component means.
    sigma : torch.Tensor of shape (batch_size, n_gaussians, output_dim)
        Component standard deviations.
    target : torch.Tensor of shape (batch_size, output_dim)
        Target values.

    Returns
    -------
    loss : torch.Tensor (scalar)
        Negative log-likelihood loss.
    """
    import torch

    # expand target to (batch_size, 1, output_dim) for broadcasting
    target_exp = target.unsqueeze(1).expand_as(mu)

    # log probability of each component
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target_exp)  # (batch, n_gaussians, output_dim)

    # sum log probs across output dims (assuming independence across outputs)
    log_prob = log_prob.sum(dim=2)  # (batch, n_gaussians)

    # add log mixing weights
    log_pi = torch.log(pi + 1e-10)  # (batch, n_gaussians)
    weighted_log_prob = log_prob + log_pi  # (batch, n_gaussians)

    # logsumexp over components, then mean over batch
    return -torch.logsumexp(weighted_log_prob, dim=1).mean()
