**CAUTION: the skpro package is currently undergoing major rearchitecting and should not be used in deployment.**

If you find this package interesting and would like to contribute, kindly contact the `sktime` developers in the [skpro & probabilistic forecasting workstream on discord](https://discord.com/channels/1075852648688930887/1106258090568986644).

![skpro](/docs/_static/logo/skpro-banner.png)

<p align="center">
  <a href="https://badge.fury.io/py/skpro"><img src="https://badge.fury.io/py/skpro.svg" alt="PyPI version" height="18"></a>
  <a href="https://travis-ci.org/sktime/skpro"><img src="https://travis-ci.org/sktime/skpro.svg?branch=master" alt="Build Status"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License"></a>
</p>

`skpro` is a library for supervised probabilistic prediction in python.
It provides `scikit-learn`-like, `scikit-base` compatible interfaces to:

* tabular **supervised regressors with probabilistic prediction modes** - interval, quantile and distribution predictions
* **performance metrics to evaluate probabilistic predictions**, e.g., pinball loss, empirical coverage, CRPS
* **reductions** to turn non-probabilistic, `scikit-learn` regressors into probabilistic `skpro` regressors, such as bootstrap or conformal
* tools for building **pipelines and composite machine learning models**, including tuning via probabilistic performance metrics
* symbolic an lazy **probability distributions** with a value domain of `pandas.DataFrame`-s and a `pandas`-like interface

## :books: Documentation

TODO - TO BE ADDED

## :speech_balloon: Where to ask questions

Questions and feedback are extremely welcome!
We strongly believe in the value of sharing help publicly, as it allows a wider audience to benefit from it.

`skpro` is maintained by the `sktime` community, we use the same social channels.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| :bug: **Bug Reports**              | [GitHub Issue Tracker]                  |
| :sparkles: **Feature Requests & Ideas** | [GitHub Issue Tracker]                       |
| :woman_technologist: **Usage Questions**          | [GitHub Discussions] · [Stack Overflow] |
| :speech_balloon: **General Discussion**        | [GitHub Discussions] |
| :factory: **Contribution & Development** | `dev-chat` channel · [Discord] |
| :globe_with_meridians: **Community collaboration session** | [Discord] - Fridays 3 pm UTC, dev/meet-ups channel |

[github issue tracker]: https://github.com/sktime/skpro/issues
[github discussions]: https://github.com/sktime/skpro/discussions
[stack overflow]: https://stackoverflow.com/questions/tagged/sktime
[discord]: https://discord.com/invite/54ACzaFsn7

## :hourglass_flowing_sand: Installing `skpro`

To install `skpro`, use `pip`:

```bash
pip install skpro
```

or, with maximum dependencies,

```bash
pip install skpro[all_extras]
```

Releases are available as source packages and binary wheels. You can see all available wheels [here](https://pypi.org/simple/skpro/).

## :zap: Quickstart

### Making probabilistic predictions

``` python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from skpro.regression.residual import ResidualDouble

# step 1: data specification
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_new, y_train, _ = train_test_split(X, y)

# step 2: specifying the regressor - any compatible regressor is valid!
# example - "squaring residuals" regressor
# random forest for mean prediction
# linear regression for variance prediction
reg_mean = RandomForestRegressor()
reg_resid = LinearRegression()
reg_proba = ResidualDouble(reg_mean, reg_resid)

# step 3: fitting the model to training data
reg_proba.fit(X_train, y_train)

# step 4: predicting labels on new data

# probabilistic prediction modes - pick any or multiple

# full distribution prediction
y_pred_proba = reg_proba.predict_proba(X_new)

# interval prediction
y_pred_interval = reg_proba.predict_interval(X_new, coverage=0.9)

# quantile prediction
y_pred_quantiles = reg_proba.predict_quantiles(X_new, alpha=[0.05, 0.5, 0.95])

# variance prediction
y_pred_var = reg_proba.predict_var(X_new)

# mean prediction is same as "classical" sklearn predict, also available
y_pred_mean = reg_proba.predict(X_new)
```

### Evaluating predictions

``` python
# step 5: specifying evaluation metric
from skpro.metrics import CRPS

metric = CRPS()  # continuous rank probability score - any skpro metric works!

# step 6: evaluat metric, compare predictions to actuals
metric(y_test, y_pred_proba)
>>> 32.19
```

## :wave: How to get involved

We welcome contributions to the skpro project. Please read our [contribution guide](/CONTRIBUTING.md).

TODO - community pointers

## :wave: Citation

To cite `skpro`` in a scientific publication, see [citations](CITATION.rst).


