
from enum import Enum
from typing import Optional, Union


class _FeatureProbBit(Enum):
    """Each bit may only be used once in the feature-property flag mask"""

    continuous = 0
    ordered = 1
    unordered = 2
    monotonic = 3
    missing = 4
    missing_not_learned = 5
    use_original = 6
    corr_to_width = 7
    is_linear = 8
    is_seasonal = 9
    magic_int_missing = 10
    increasing = 11
    decreasing = 12


IS_CONTINUOUS = 1 << _FeatureProbBit.continuous.value
IS_ORDERED = 1 << _FeatureProbBit.ordered.value
IS_UNORDERED = 1 << _FeatureProbBit.unordered.value
IS_MONOTONIC = IS_CONTINUOUS | (1 << _FeatureProbBit.monotonic.value)
HAS_MISSING = 1 << _FeatureProbBit.missing.value
MISSING_NOT_LEARNED = HAS_MISSING | (1 << _FeatureProbBit.missing_not_learned.value)
USE_ORIGINAL = 1 << _FeatureProbBit.use_original.value
CORR_TO_WIDTH = 1 << _FeatureProbBit.corr_to_width.value
IS_LINEAR = 1 << _FeatureProbBit.is_linear.value
IS_SEASONAL = 1 << _FeatureProbBit.is_seasonal.value
HAS_MAGIC_INT_MISSING = HAS_MISSING | (1 << _FeatureProbBit.magic_int_missing.value)
INCREASING = 1 << _FeatureProbBit.increasing.value
DECREASING = 1 << _FeatureProbBit.decreasing.value

_FLAG_LIST = [
    "IS_CONTINUOUS",
    "IS_ORDERED",
    "IS_UNORDERED",
    "IS_MONOTONIC",
    "HAS_MISSING",
    "MISSING_NOT_LEARNED",
    "USE_ORIGINAL",
    "CORR_TO_WIDTH",
    "IS_LINEAR",
    "IS_SEASONAL",
]


def is_continuous_set(feature_prop: int) -> bool:
    """Is IS_CONTINUOUS included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import is_continuous_set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> is_continuous_set(flags.IS_CONTINUOUS)
    True
    >>> is_continuous_set(flags.IS_CONTINUOUS | flags.HAS_MISSING)
    True
    >>> is_continuous_set(flags.IS_ORDERED)
    False

    IS_MONOTONIC implies IS_CONTINUOUS:

    >>> is_continuous_set(flags.IS_MONOTONIC)
    True
    """
    return feature_prop & IS_CONTINUOUS == IS_CONTINUOUS


def is_ordered_set(feature_prop: int) -> bool:
    """Is IS_ORDERED included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import is_ordered_set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> is_ordered_set(flags.IS_ORDERED)
    True
    >>> is_ordered_set(flags.IS_ORDERED | flags.HAS_MISSING)
    True
    >>> is_ordered_set(flags.IS_CONTINUOUS)
    False
    """
    return feature_prop & IS_ORDERED == IS_ORDERED


def is_unordered_set(feature_prop: int) -> bool:
    """Is IS_UNORDERED included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import is_unordered_set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> is_unordered_set(flags.IS_UNORDERED)
    True
    >>> is_unordered_set(flags.IS_UNORDERED | flags.HAS_MISSING)
    True
    >>> is_unordered_set(flags.IS_ORDERED)
    False
    >>> is_unordered_set(flags.IS_CONTINUOUS)
    False
    """
    return feature_prop & IS_UNORDERED == IS_UNORDERED


def is_monotonic_set(feature_prop: int) -> bool:
    """Is IS_MONOTONIC included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import is_monotonic_set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> is_monotonic_set(flags.IS_MONOTONIC)
    True
    >>> is_monotonic_set(flags.IS_MONOTONIC | flags.HAS_MISSING)
    True
    >>> is_monotonic_set(flags.IS_CONTINUOUS)
    False
    >>> is_monotonic_set(flags.IS_CONTINUOUS | flags.IS_MONOTONIC)
    True
    >>> is_monotonic_set(flags.IS_ORDERED)
    False
    >>> is_monotonic_set(flags.IS_MONOTONIC_INCREASING)
    True
    """
    return feature_prop & IS_MONOTONIC == IS_MONOTONIC


def increasing_set(feature_prop: int) -> bool:
    """Is INCREASING included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import _set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> _increasing_set(flags.INCREASING)
    True
    >>> increasing_set(flags.INCREASING | flags.HAS_MISSING)
    True
    >>> increasing_set(flags.IS_MONOTONIC)
    False
    """
    return feature_prop & INCREASING == INCREASING


def decreasing_set(feature_prop: int) -> bool:
    """Is DECREASING included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import _set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> decreasing_set(flags.DECREASING)
    True
    >>> decreasing_set(flags.DECREASING | flags.HAS_MISSING)
    True
    >>> decreasing_set(flags.IS_MONOTONIC)
    False
    """
    return feature_prop & DECREASING == DECREASING


def has_magic_missing_set(feature_prop: int) -> bool:
    return feature_prop & HAS_MAGIC_INT_MISSING == HAS_MAGIC_INT_MISSING


def has_missing_set(feature_prop: int) -> bool:
    """Is HAS_MISSING included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import has_missing_set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> has_missing_set(flags.HAS_MISSING)
    True
    >>> has_missing_set(flags.IS_CONTINUOUS | flags.HAS_MISSING)
    True
    >>> has_missing_set(flags.IS_CONTINUOUS)
    False

    MISSING_NOT_LEARNED implies HAS_MISSING:

    >>> has_missing_set(flags.MISSING_NOT_LEARNED)
    True
    """
    return feature_prop & HAS_MISSING == HAS_MISSING


def missing_not_learned_set(feature_prop: int) -> bool:
    """Is MISSING_NOT_LEARNED included in the feature properties?

    :param feature_prop: feature properties for one feature
    :type feature_prop: int

    :rtype: bool

    >>> from skpro.libs.cyclic_boosting.flags import missing_not_learned_set
    >>> from skpro.libs.cyclic_boosting import flags
    >>> missing_not_learned_set(flags.MISSING_NOT_LEARNED)
    True
    >>> missing_not_learned_set(flags.IS_CONTINUOUS | flags.MISSING_NOT_LEARNED)
    True
    >>> missing_not_learned_set(flags.HAS_MISSING)
    False
    >>> missing_not_learned_set(flags.IS_CONTINUOUS)
    False
    """
    return feature_prop & MISSING_NOT_LEARNED == MISSING_NOT_LEARNED


def is_linear_set(feature_prop: int) -> bool:
    return feature_prop & IS_LINEAR == IS_LINEAR


def is_seasonal_set(feature_prop: int) -> bool:
    return feature_prop & IS_SEASONAL == IS_SEASONAL


def check_flags_consistency(feature_prop: int) -> None:
    """Check that exactly one of ``IS_CONTINUOUS, IS_ORDERED, IS_UNORDERED``
    has been set.

    Parameters
    ----------
    feature_prop: int
        value to check for consistency

    Examples
    --------

    >>> from skpro.libs.cyclic_boosting.flags import check_flags_consistency
    >>> from skpro.libs.cyclic_boosting import flags

    The following flags will just work:

    >>> check_flags_consistency(flags.IS_CONTINUOUS)
    >>> check_flags_consistency(flags.IS_ORDERED)
    >>> check_flags_consistency(flags.IS_UNORDERED)

    Flags that fail:

    >>> check_flags_consistency(flags.HAS_MISSING)
    Traceback (most recent call last):
    ValueError: Exactly one of IS_CONTINUOUS, IS_ORDERED, IS_UNORDERED ...

    >>> check_flags_consistency(flags.IS_CONTINUOUS | flags.IS_UNORDERED)
    Traceback (most recent call last):
    ValueError: Exactly one of IS_CONTINUOUS, IS_ORDERED, IS_UNORDERED ...

    >>> check_flags_consistency(flags.IS_CONTINUOUS | flags.IS_ORDERED)
    Traceback (most recent call last):
    ValueError: Exactly one of IS_CONTINUOUS, IS_ORDERED, IS_UNORDERED ...

    >>> check_flags_consistency(flags.IS_ORDERED | flags.IS_UNORDERED)
    Traceback (most recent call last):
    ValueError: Exactly one of IS_CONTINUOUS, IS_ORDERED, IS_UNORDERED ...

    >>> check_flags_consistency(flags.DECREASING | flags.INCREASING)
    Traceback (most recent call last):
    ValueError: One feature can either be ...
    """
    if (
        int(is_continuous_set(feature_prop)) + int(is_ordered_set(feature_prop)) + int(is_unordered_set(feature_prop))
        != 1
    ):
        raise ValueError(
            "Exactly one of IS_CONTINUOUS, IS_ORDERED, "
            "IS_UNORDERED must be set in a "
            "flags value for the feature properties."
        )
    if int(increasing_set(feature_prop)) + int(decreasing_set(feature_prop)) == 2:
        raise ValueError("One feature can either be INCREASING" " or DECREASING")


def flags_to_string(flags_value: Union[int, tuple]) -> Union[tuple, str]:
    """
    This function converts the numeric flags to the corresponding strings
    that are defined in the flag list.

    Parameters
    ----------

    flags_value: int/tuple
        preprocessing flag (see :mod:`cyclic_boosting.flags`)

    Returns
    -------
    str
        Flag value

    Examples
    --------

    >>> from skpro.libs.cyclic_boosting.flags import flags_to_string
    >>> from skpro.libs.cyclic_boosting import flags
    >>> flags_to_string(flags.IS_CONTINUOUS)
    'IS_CONTINUOUS'

    >>> flags_to_string(flags.IS_UNORDERED)
    'IS_UNORDERED'

    >>> flags_to_string(flags.IS_ORDERED)
    'IS_ORDERED'

    >>> flags_to_string(flags.IS_MONOTONIC)
    'IS_CONTINUOUS | IS_MONOTONIC'

    >>> flags_to_string(flags.HAS_MISSING | flags.IS_CONTINUOUS)
    'IS_CONTINUOUS | HAS_MISSING'

    >>> flags_to_string(flags.IS_UNORDERED | flags.HAS_MISSING
    ... | flags.MISSING_NOT_LEARNED)
    'IS_UNORDERED | HAS_MISSING | MISSING_NOT_LEARNED'

    >>> flags_to_string((flags.IS_ORDERED, flags.HAS_MISSING
    ... | flags.IS_CONTINUOUS))
    ('IS_ORDERED', 'IS_CONTINUOUS | HAS_MISSING')

    >>> flags_to_string(flags.IS_CONTINUOUS | flags.IS_UNORDERED)
    'IS_CONTINUOUS | IS_UNORDERED'
    """
    if isinstance(flags_value, tuple):
        return tuple(_convert_flags_to_string(flags_val) for flags_val in flags_value)
    else:
        return _convert_flags_to_string(flags_value)


def _convert_flags_to_string(
    flags_value: int, alternative_flag_list: Optional[int] = None, alternative_flags: Optional[dict] = None
) -> str:
    """
    This function converts the numeric flags to the corresponding strings
    that are defined in the flag list.

    Parameters
    ----------

    flags_value: int
        preprocessing flag (see :mod:`cyclic_boosting.flags`)

    alternative_flag_list: int
        alternative list of possible flags

    alternative_flags: dict
        alternative flags

    Returns
    -------
    str
        string representation of flag
    """
    if alternative_flag_list is None:
        alternative_flag_list = _FLAG_LIST

    lst = []
    flags1 = flags_value
    glob_vars = globals()

    if alternative_flags is not None:
        glob_vars.update(alternative_flags)

    for flag_name in alternative_flag_list:
        flag = glob_vars[flag_name]
        if flag & flags_value == flag:
            lst.append(flag_name)
            flags1 &= ~flag
    if flags1 != 0:
        raise ValueError("Unknown flags %d found" % flags1)
    return " | ".join(lst)


def read_feature_property(
    feature_properties: dict, feature_group: Union[tuple, str], default: int
) -> Union[tuple, int]:
    """Read a feature property out of the ``feature_properties`` dict which may
    be None. If no value is found, return ``default`` as the default value.

    :param feature_properties: the `feature_properties` dict
    :type feature_properties: dict

    :param feature_group: feature group name
    :type feature_group: str or tuple of str

    :param default: the default value to return if ``feature_properties`` is
        `None` or doesn't contain ``feature_group`` as a key.
    :type default: int

    >>> from skpro.libs.cyclic_boosting.flags import read_feature_property, flags_to_string
    >>> from skpro.libs.cyclic_boosting import flags
    >>> feature_properties = {'a': flags.IS_ORDERED, 'b': flags.IS_UNORDERED}
    >>> flags_to_string(read_feature_property(feature_properties, 'b', flags.HAS_MISSING))
    'IS_UNORDERED'
    >>> flags_to_string(read_feature_property(feature_properties, 'c', flags.HAS_MISSING))
    'HAS_MISSING'
    """
    if isinstance(feature_group, tuple):
        return tuple(read_feature_property(feature_properties, col, default=default) for col in feature_group)
    else:
        feature_prop = None
        if feature_properties is not None:
            feature_prop = feature_properties.get(feature_group)
        if feature_prop is None:
            feature_prop = default
        return feature_prop


__all__ = [
    "IS_CONTINUOUS",
    "IS_LINEAR",
    "IS_ORDERED",
    "IS_UNORDERED",
    "IS_MONOTONIC",
    "HAS_MISSING",
    "HAS_MAGIC_INT_MISSING",
    "MISSING_NOT_LEARNED",
    "is_continuous_set",
    "is_unordered_set",
    "is_ordered_set",
    "is_monotonic_set",
    "has_missing_set",
    "missing_not_learned_set",
    "increasing_set",
    "decreasing_set",
    "check_flags_consistency",
    "flags_to_string",
    "read_feature_property",
    "has_magic_missing_set",
]
