<a href="https://skpro.readthedocs.io/en/latest"><img src="https://github.com/sktime/skpro/blob/main/docs/source/images/skpro-banner.png" width="500" align="right" /></a>

:rocket: **Version 2.2.0 out now!** [Read the release notes here.](https://skpro.readthedocs.io/en/latest/changelog.html).

`skpro` is a library for supervised probabilistic prediction in python.
It provides `scikit-learn`-like, `scikit-base` compatible interfaces to:

* tabular **supervised regressors for probabilistic prediction** - interval, quantile and distribution predictions
* tabular **probabilistic time-to-event and survival prediction** - instance-individual survival distributions
* **metrics to evaluate probabilistic predictions**, e.g., pinball loss, empirical coverage, CRPS, survival losses
* **reductions** to turn `scikit-learn` regressors into probabilistic `skpro` regressors, such as bootstrap or conformal
* building **pipelines and composite models**, including tuning via probabilistic performance metrics
* symbolic **probability distributions** with value domain of `pandas.DataFrame`-s and `pandas`-like interface

| Overview | |
|---|---|
| **Open Source** |  [![BSD 3-clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/sktime/sktime/blob/main/LICENSE) |
| **Tutorials** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/skpro/main?filepath=examples) [![!youtube](https://img.shields.io/static/v1?logo=youtube&label=YouTube&message=tutorials&color=red)](https://www.youtube.com/playlist?list=PLKs3UgGjlWHqNzu0LEOeLKvnjvvest2d0) |
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/scikit-time/) |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/sktime/sktime/wheels.yml?logo=github)](https://github.com/sktime/skpro/actions/workflows/wheels.yml) [![!codecov](https://img.shields.io/codecov/c/github/sktime/skpro?label=codecov&logo=codecov)](https://codecov.io/gh/sktime/skpro) [![readthedocs](https://img.shields.io/readthedocs/skpro?logo=readthedocs)](https://skpro.readthedocs.io/en/latest/) [![platform](https://img.shields.io/conda/pn/conda-forge/skpro)](https://github.com/sktime/skpro) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/skpro?color=orange)](https://pypi.org/project/skpro/) [![!conda](https://img.shields.io/conda/vn/conda-forge/skpro)](https://anaconda.org/conda-forge/skpro) [![!python-versions](https://img.shields.io/pypi/pyversions/skpro)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| **Downloads**| [![Downloads](https://static.pepy.tech/personalized-badge/skpro?period=week&units=international_system&left_color=grey&right_color=blue&left_text=weekly%20(pypi))](https://pepy.tech/project/skpro) [![Downloads](https://static.pepy.tech/personalized-badge/skpro?period=month&units=international_system&left_color=grey&right_color=blue&left_text=monthly%20(pypi))](https://pepy.tech/project/skpro) [![Downloads](https://static.pepy.tech/personalized-badge/skpro?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/skpro) |

## :books: Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| :star: **[Tutorials]**        | New to skpro? Here's everything you need to know!              |
| :clipboard: **[Binder Notebooks]** | Example notebooks to play with in your browser.              |
| :woman_technologist: **[User Guides]**      | How to use skpro and its features.                             |
| :scissors: **[Extension Templates]** | How to build your own estimator using skpro's API.            |
| :control_knobs: **[API Reference]**      | The detailed reference for skpro's API.                        |
| :hammer_and_wrench: **[Changelog]**          | Changes and version history.                                   |
| :deciduous_tree: **[Roadmap]**          | skpro's software and community development plan.                                   |
| :pencil: **[Related Software]**          | A list of related software. |

[tutorials]: https://skpro.readthedocs.io/en/latest/tutorials.html
[binder notebooks]: https://mybinder.org/v2/gh/sktime/skpro/main?filepath=examples
[user guides]: https://skpro.readthedocs.io/en/latest/user_guide.html
[extension templates]: https://github.com/sktime/skpro/tree/main/extension_templates
[api reference]: https://skpro.readthedocs.io/en/latest/api_reference.html
[changelog]: https://skpro.readthedocs.io/en/latest/changelog.html
[roadmap]: https://skpro.readthedocs.io/en/latest/roadmap.html
[related software]: https://skpro.readthedocs.io/en/latest/related_software.html


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


## :dizzy: Features

Our objective is to enhance the interoperability and usability of the AI model ecosystem:

* ``skpro`` is compatible with [scikit-learn] and [sktime], e.g., an ``sktime`` proba forecaster can
be built with an ``skpro`` proba regressor which in an ``sklearn`` regressor with proba mode added by ``skpro``

* ``skpro`` provides a mini-package management framework for first-party implemenentations,
and for interfacing popular second- and third-party components, such as [cyclic-boosting] or [MAPIE] packages.

[scikit-learn]: https://scikit-learn.org/stable/
[sktime]: https://www.sktime.net
[MAPIE]: https://mapie.readthedocs.io/en/latest/
[cyclic-boosting]:  https://cyclic-boosting.readthedocs.io/en/latest/

``skpro`` curates libraries of components of the following types:

| Module | Status | Links |
|---|---|---|
| **[Probabilistic tabular regression]** | maturing | [Tutorial](https://github.com/sktime/skpro/blob/main/examples/01_skpro_intro.ipynb) · [API Reference](https://skpro.readthedocs.io/en/latest/api_reference/regression.html) · [Extension Template](https://github.com/sktime/skpro/blob/main/extension_templates/regression.py) |
| **[Time-to-event (survival) prediction]** | experimental | [API Reference](https://skpro.readthedocs.io/en/latest/api_reference/survival.html) · [Extension Template](https://github.com/sktime/skpro/blob/main/extension_templates/survival.py) |
| **[Performance metrics]** | maturing | [API Reference](https://skpro.readthedocs.io/en/latest/api_reference/metrics.html) |
| **[Probability distributions]** | maturing | [API Reference](https://skpro.readthedocs.io/en/latest/api_reference/distributions.html) |

[Probabilistic tabular regression]: https://github.com/sktime/skpro/tree/main/skpro/regression
[Time-to-event (survival) prediction]: https://github.com/sktime/skpro/tree/main/skpro/survival
[Performance metrics]: https://github.com/sktime/skpro/tree/main/skpro/metrics
[Probability distributions]: https://github.com/sktime/skpro/tree/main/skpro/distributions


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

There are many ways to get involved with development of `skpro`, which is
developed by the `sktime` community.
We follow the [all-contributors](https://github.com/all-contributors/all-contributors)
specification: all kinds of contributions are welcome - not just code.

| Documentation              |                                                                |
| -------------------------- | --------------------------------------------------------------        |
| :gift_heart: **[Contribute]**        | How to contribute to skpro.          |
| :school_satchel:  **[Mentoring]** | New to open source? Apply to our mentoring program! |
| :date: **[Meetings]** | Join our discussions, tutorials, workshops, and sprints! |
| :woman_mechanic:  **[Developer Guides]**      | How to further develop the skpro code base.                             |
| :medal_sports: **[Contributors]** | A list of all contributors. |
| :raising_hand: **[Roles]** | An overview of our core community roles. |
| :money_with_wings: **[Donate]** | Fund sktime and skpro maintenance and development. |
| :classical_building: **[Governance]** | How and by whom decisions are made in sktime's community.   |

[contribute]: https://skpro.readthedocs.io/en/latest/get_involved/contributing.html
[donate]: https://opencollective.com/sktime
[developer guides]: https://skpro.readthedocs.io/en/latest/developer_guide.html
[contributors]: https://github.com/sktime/skpro/blob/main/CONTRIBUTORS.md
[governance]: https://www.sktime.net/en/latest/get_involved/governance.html
[mentoring]: https://github.com/sktime/mentoring
[meetings]: https://calendar.google.com/calendar/u/0/embed?src=sktime.toolbox@gmail.com&ctz=UTC
[roles]: https://www.sktime.net/en/latest/about/team.html


## :wave: Citation

To cite `skpro` in a scientific publication, see [citations](CITATION.rst).
