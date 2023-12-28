"""Testing machine type converters for scitypes - covert_to utility."""

__author__ = ["fkiraly"]

from skpro.datatypes._convert import convert_to
from skpro.datatypes._examples import get_examples
from skpro.utils import deep_equals

# hard-coded scitypes/mtypes to use in test_convert_to
#   easy to change in case the strings change
SCITYPES = ["Proba", "Table"]
MTYPES_PROBA = ["pred_interval", "pred_quantiles"]
MTYPES_TABLE = ["list_of_dict", "pd_Series_Table", "numpy2D"]


def test_convert_to_simple():
    """Testing convert_to basic call works."""
    scitype = SCITYPES[0]

    from_fixt = get_examples(mtype=MTYPES_PROBA[1], as_scitype=scitype).get(0)
    # expectation is that the conversion is to mtype MTYPES_PROBA[0]
    exp_fixt = get_examples(mtype=MTYPES_PROBA[0], as_scitype=scitype).get(0)

    # carry out the conversion using convert_to
    converted = convert_to(from_fixt, to_type=MTYPES_PROBA[0], as_scitype=scitype)

    # compare expected output with actual output of convert_to
    msg = "convert_to basic call does not seem to work."
    assert deep_equals(converted, exp_fixt), msg


def test_convert_to_without_scitype():
    """Testing convert_to call without scitype specification."""
    scitype = SCITYPES[0]

    from_fixt = get_examples(mtype=MTYPES_PROBA[1], as_scitype=scitype).get(0)
    # convert_to should recognize the correct scitype, otherwise same as above
    exp_fixt = get_examples(mtype=MTYPES_PROBA[0], as_scitype=scitype).get(0)

    # carry out the conversion using convert_to
    converted = convert_to(from_fixt, to_type=MTYPES_PROBA[0])

    # compare expected output with actual output of convert_to
    msg = "convert_to call without scitype does not seem to work."
    assert deep_equals(converted, exp_fixt), msg


def test_convert_to_mtype_list():
    """Testing convert_to call to_type being a list, of same scitype."""
    # convert_to list
    target_list = MTYPES_TABLE[:2]
    scitype = SCITYPES[0]

    # example that is on the list
    from_fixt_on = get_examples(mtype=MTYPES_TABLE[1], as_scitype=scitype).get(0)
    # example that is not on the list
    from_fixt_off = get_examples(mtype=MTYPES_TABLE[2], as_scitype=scitype).get(0)

    # if on the list, result should be equal to input
    exp_fixt_on = get_examples(mtype=MTYPES_TABLE[1], as_scitype=scitype).get(0)
    # if off the list, result should be converted to mtype that is first on the list
    exp_fixt_off = get_examples(mtype=MTYPES_TABLE[0], as_scitype=scitype).get(0)

    # carry out the conversion using convert_to
    converted_on = convert_to(from_fixt_on, to_type=target_list)
    converted_off = convert_to(from_fixt_off, to_type=target_list)

    # compare expected output with actual output of convert_to
    msg = "convert_to call does not work with list for to_type."
    assert deep_equals(converted_on, exp_fixt_on), msg
    assert deep_equals(converted_off, exp_fixt_off), msg
