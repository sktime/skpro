from skpro.libs.cyclic_boosting import (
    CBLocationRegressor,
    CBExponential,
    CBNBinomRegressor,
    CBPoissonRegressor,
    CBLocPoissonRegressor,
    CBNBinomC,
    CBClassifier,
    CBGBSRegressor,
    CBMultiplicativeQuantileRegressor,
    CBAdditiveQuantileRegressor,
    CBMultiplicativeGenericRegressor,
    CBAdditiveGenericRegressor,
    CBGenericClassifier,
    binning,
)

from sklearn.pipeline import Pipeline


def pipeline_CB(
    estimator=None,
    feature_groups=None,
    hierarchical_feature_groups=None,
    feature_properties=None,
    weight_column=None,
    prior_prediction_column=None,
    minimal_loss_change=1e-3,
    minimal_factor_change=1e-3,
    maximal_iterations=10,
    observers=None,
    smoother_choice=None,
    output_column=None,
    learn_rate=None,
    number_of_bins=100,
    aggregate=True,
    a=1.0,
    c=0.0,
    external_colname=None,
    standard_feature_groups=None,
    external_feature_groups=None,
    var_prior_exponent=0.1,
    prior_exponent_colname=None,
    mean_prediction_column=None,
    gamma=0.0,
    bayes=False,
    n_steps=15,
    regalpha=0.0,
    quantile=None,
    costs=None,
    inplace=False,
):
    if estimator in [CBPoissonRegressor, CBLocPoissonRegressor, CBLocationRegressor, CBClassifier]:
        estimatorCB = estimator(
            feature_groups=feature_groups,
            hierarchical_feature_groups=hierarchical_feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
        )
    elif estimator == CBNBinomRegressor:
        estimatorCB = estimator(
            feature_groups=feature_groups,
            hierarchical_feature_groups=hierarchical_feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
            a=a,
            c=c,
        )
    elif estimator == CBExponential:
        estimatorCB = estimator(
            external_colname=external_colname,
            standard_feature_groups=standard_feature_groups,
            external_feature_groups=external_feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            var_prior_exponent=var_prior_exponent,
            prior_exponent_colname=prior_exponent_colname,
        )
    elif estimator == CBNBinomC:
        estimatorCB = estimator(
            mean_prediction_column=mean_prediction_column,
            feature_groups=feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            gamma=gamma,
            bayes=bayes,
            n_steps=n_steps,
        )
    elif estimator == CBGBSRegressor:
        estimatorCB = estimator(
            feature_groups=feature_groups,
            hierarchical_feature_groups=hierarchical_feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
            regalpha=regalpha,
        )
    elif estimator in [CBMultiplicativeQuantileRegressor, CBAdditiveQuantileRegressor]:
        estimatorCB = estimator(
            feature_groups=feature_groups,
            hierarchical_feature_groups=hierarchical_feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
            quantile=quantile,
        )
    elif estimator in [CBMultiplicativeGenericRegressor, CBAdditiveGenericRegressor, CBGenericClassifier]:
        estimatorCB = estimator(
            feature_groups=feature_groups,
            hierarchical_feature_groups=hierarchical_feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
            costs=costs,
        )
    else:
        raise Exception("No valid CB estimator.")
    binner = binning.BinNumberTransformer(n_bins=number_of_bins, feature_properties=feature_properties, inplace=inplace)

    return Pipeline([("binning", binner), ("CB", estimatorCB)])


def pipeline_CBPoissonRegressor(**kwargs):
    """
    Convenience function containing CBPoissonRegressor (estimator) + binning.
    """
    return pipeline_CB(CBPoissonRegressor, **kwargs)


def pipeline_CBNBinomRegressor(**kwargs):
    """
    Convenience function containing CBNBinomRegressor (estimator) + binning.
    """
    return pipeline_CB(CBNBinomRegressor, **kwargs)


def pipeline_CBClassifier(**kwargs):
    """
    Convenience function containing CBClassifier (estimator) + binning.
    """
    return pipeline_CB(CBClassifier, **kwargs)


def pipeline_CBLocationRegressor(**kwargs):
    """
    Convenience function containing CBLocationRegressor (estimator) + binning.
    """
    return pipeline_CB(CBLocationRegressor, **kwargs)


def pipeline_CBExponential(**kwargs):
    """
    Convenience function containing CBExponential (estimator) + binning.
    """
    return pipeline_CB(CBExponential, **kwargs)


def pipeline_CBLocPoissonRegressor(**kwargs):
    """
    Convenience function containing CBLocPoissonRegressor (estimator) + binning.
    """
    return pipeline_CB(CBLocPoissonRegressor, **kwargs)


def pipeline_CBNBinomC(**kwargs):
    """
    Convenience function containing CBNBinomC (estimator) + binning.
    """
    return pipeline_CB(CBNBinomC, **kwargs)


def pipeline_CBGBSRegressor(**kwargs):
    """
    Convenience function containing CBGBSRegressor (estimator) + binning.
    """
    return pipeline_CB(CBGBSRegressor, **kwargs)


def pipeline_CBMultiplicativeQuantileRegressor(**kwargs):
    """
    Convenience function containing CBMultiplicativeQuantileRegressor (estimator) + binning.
    """
    return pipeline_CB(CBMultiplicativeQuantileRegressor, **kwargs)


def pipeline_CBAdditiveQuantileRegressor(**kwargs):
    """
    Convenience function containing CBAdditiveQuantileRegressor (estimator) + binning.
    """
    return pipeline_CB(CBAdditiveQuantileRegressor, **kwargs)


def pipeline_CBMultiplicativeGenericRegressor(**kwargs):
    """
    Convenience function containing CBMultiplicativeGenericRegressor (estimator) + binning.
    """
    return pipeline_CB(CBMultiplicativeGenericRegressor, **kwargs)


def pipeline_CBAdditiveGenericRegressor(**kwargs):
    """
    Convenience function containing CBAdditiveGenericRegressor (estimator) + binning.
    """
    return pipeline_CB(CBAdditiveGenericRegressor, **kwargs)


def pipeline_CBGenericClassifier(**kwargs):
    """
    Convenience function containing CBGenericClassifier (estimator) + binning.
    """
    return pipeline_CB(CBGenericClassifier, **kwargs)
