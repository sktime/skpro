from typing import Optional, Union

import numpy as np

from skpro.libs.cyclic_boosting.features import Feature


def constant_learn_rate_one(
    iteration: int, maximal_iteration: int, feature: Optional[Feature] = None
) -> float:
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate returned is always 1.

    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature
    """
    return 1.0


def linear_learn_rate(
    iteration: int,
    maximal_iteration: Union[int, float],
    feature: Optional[Feature] = None,
) -> float:
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate is linear increasing each iteration until it reaches 1 in
    the last iteration.


    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature
    """
    return iteration * (1.0 / maximal_iteration)


def logistic_learn_rate(
    iteration: int, maximal_iteration: int, feature: Optional[Feature] = None
) -> float:
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate has a logistic form.

    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature
    """
    saturation_value = 0.999999999
    x_t = maximal_iteration / np.log((1 - saturation_value) / (1 + saturation_value))
    return (1.0 / (1.0 + np.exp(iteration / x_t)) - 0.5) * 2


def half_linear_learn_rate(
    iteration: int, maximal_iteration: int, feature: Optional[Feature] = None
) -> float:
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate is linear increasing each iteration until it reaches 1 in
    half of the iterations.


    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature
    """
    return np.minimum(
        linear_learn_rate(iteration, maximal_iteration * 0.5, feature), 1.0
    )
