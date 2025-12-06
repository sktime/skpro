"""Utilities for harmonizing distribution keyword arguments across estimators."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

# Sentinel that allows us to detect whether a kwarg was explicitly provided.
DISTRIBUTION_NOT_GIVEN: object = object()


def _provided_aliases(alias_values: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    """Yield alias/value pairs that were explicitly provided."""
    for name, value in alias_values.items():
        if value is DISTRIBUTION_NOT_GIVEN:
            continue
        if value is None:
            continue
        yield name, value


def resolve_distribution_kwarg(
    *,
    estimator_name: str,
    default: Any,
    alias_values: Dict[str, Any],
) -> Any:
    """Resolve distribution aliases to a single value.

    Parameters
    ----------
    estimator_name : str
        Name of the estimator for which the resolution happens (used in errors).
    default : Any
        Default value to use when no alias is explicitly provided.
    alias_values : dict
        Mapping from alias name to the value that was supplied for that alias.

    Returns
    -------
    Any
        Resolved distribution value.

    Raises
    ------
    ValueError
        If multiple aliases are provided with conflicting values.
    """
    provided = list(_provided_aliases(alias_values))
    if not provided:
        return default

    first_name, first_value = provided[0]
    for name, value in provided[1:]:
        if value != first_value:
            raise ValueError(
                f"{estimator_name} received conflicting distribution kwargs "
                f"('{first_name}'={first_value!r}, '{name}'={value!r}). "
                "Please specify only one of these aliases."
            )

    return first_value

