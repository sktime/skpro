"""Utilities for set_output functionality."""

__author__ = ["julian-fong"]

from skpro.datatypes import convert
from skpro.utils.validation._dependencies import _check_soft_dependencies

SUPPORTED_OUTPUTS = ["pandas", "default"]

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    SUPPORTED_OUTPUTS.append("polars")


SUPPORTED_OUTPUT_MAPPINGS = {
    "pandas": ("pd_DataFrame_Table", "Table"),
    "polars": ("polars_eager_table", "Table"),
}


def check_output_config(estimator):
    """Given an estimator, verify the transform key in _config is available.

    Parameters
    ----------
    estimator : a given regression estimator

    Returns
    -------
    dense_config : a dict containing the specified mtype user wishes to convert
        corresponding dataframes to.
        - "dense": specifies the mtype data container in the transform config
            Possible values are located in SUPPORTED_OUTPUTS in
            `skpro.utils.set_output`
    """
    output_config = {}
    transform_output = estimator.get_config()["transform"]
    if transform_output not in SUPPORTED_OUTPUTS:
        raise ValueError(
            f"set_output container must be in {SUPPORTED_OUTPUTS}, "
            f"found {transform_output}."
        )
        valid = False
    elif transform_output != "default":
        valid = True
        output_config["dense"] = SUPPORTED_OUTPUT_MAPPINGS[transform_output]
    else:
        valid = False

    return valid, output_config


def transform_output(
    obj, valid, from_type, default_to_type, default_scitype, output_config, store
):
    """Return the correct specified output container."""
    if valid:
        convert_to_type = output_config["dense"][0]
        convert_to_scitype = output_config["dense"][1]
    else:
        convert_to_type = default_to_type
        convert_to_scitype = default_scitype

    obj = convert(
        obj,
        from_type=from_type,
        to_type=convert_to_type,
        as_scitype=convert_to_scitype,
        store=store,
    )

    return obj
