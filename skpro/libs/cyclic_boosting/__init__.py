"""This package contains the Cyclic Boosting family of machine learning
algorithms.

If you are looking for conceptional explanations of the Cyclic Boosting
algorithm, you might have a look at the two papers
https://arxiv.org/abs/2002.03425 and https://arxiv.org/abs/2009.07052.

API reference of the different Cyclic Boosting methods:

Multiplicative Regression

- :class:`~.CBPoissonRegressor`
- :class:`~.CBNBinomRegressor`
- :class:`~.CBExponential`
- :class:`~.CBMultiplicativeQuantileRegressor`
- :class:`~.CBMultiplicativeGenericCRegressor`

Additive Regression

- :class:`~.CBLocationRegressor`
- :class:`~.CBLocPoissonRegressor`
- :class:`~.CBAdditiveQuantileRegressor`
- :class:`~.CBAdditiveGenericCRegressor`

PDF Prediction

- :class:`~.CBNBinomC`

Classification

- :class:`~.CBClassifier`
- :class:`~.CBGenericClassifier`

Background Subtraction

- :class:`~.CBGBSRegressor`
"""


from skpro.libs.cyclic_boosting.base import CyclicBoostingBase
from skpro.libs.cyclic_boosting.classification import CBClassifier
from skpro.libs.cyclic_boosting.GBSregression import CBGBSRegressor
from skpro.libs.cyclic_boosting.generic_loss import (
    CBAdditiveGenericRegressor,
    CBAdditiveQuantileRegressor,
    CBGenericClassifier,
    CBMultiplicativeGenericRegressor,
    CBMultiplicativeQuantileRegressor,
)
from skpro.libs.cyclic_boosting.location import (
    CBLocationRegressor,
    CBLocPoissonRegressor,
)
from skpro.libs.cyclic_boosting.nbinom import CBNBinomC
from skpro.libs.cyclic_boosting.pipelines import (
    pipeline_CBAdditiveGenericRegressor,
    pipeline_CBAdditiveQuantileRegressor,
    pipeline_CBClassifier,
    pipeline_CBExponential,
    pipeline_CBGBSRegressor,
    pipeline_CBGenericClassifier,
    pipeline_CBLocationRegressor,
    pipeline_CBLocPoissonRegressor,
    pipeline_CBMultiplicativeGenericRegressor,
    pipeline_CBMultiplicativeQuantileRegressor,
    pipeline_CBNBinomC,
    pipeline_CBNBinomRegressor,
    pipeline_CBPoissonRegressor,
)
from skpro.libs.cyclic_boosting.price import CBExponential
from skpro.libs.cyclic_boosting.regression import CBNBinomRegressor, CBPoissonRegressor

__all__ = [
    "CyclicBoostingBase",
    "CBPoissonRegressor",
    "CBNBinomRegressor",
    "CBExponential",
    "CBLocationRegressor",
    "CBLocPoissonRegressor",
    "CBNBinomC",
    "CBClassifier",
    "CBGBSRegressor",
    "CBMultiplicativeQuantileRegressor",
    "CBAdditiveQuantileRegressor",
    "CBMultiplicativeGenericRegressor",
    "CBAdditiveGenericRegressor",
    "CBGenericClassifier",
    "pipeline_CBPoissonRegressor",
    "pipeline_CBNBinomRegressor",
    "pipeline_CBClassifier",
    "pipeline_CBLocationRegressor",
    "pipeline_CBExponential",
    "pipeline_CBLocPoissonRegressor",
    "pipeline_CBNBinomC",
    "pipeline_CBGBSRegressor",
    "pipeline_CBMultiplicativeQuantileRegressor",
    "pipeline_CBAdditiveQuantileRegressor",
    "pipeline_CBMultiplicativeGenericRegressor",
    "pipeline_CBAdditiveGenericRegressor",
    "pipeline_CBGenericClassifier",
]

__version__ = "1.4.0"
