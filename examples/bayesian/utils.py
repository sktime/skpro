"""Utility function for Bayesian example notebooks."""

import pandas as pd


def style_data(data, vmax=None, subset=None, cmap="coolwarm", hide_index=False):
    """
    Apply styling to a Series or DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The data to style.
    vmax : float, optional
        The maximum numeric value for the color spectrum.
        Defaults to the max value in the data.
    subset : list, optional
        List of columns to which the formatting is to be applied (for DataFrames).
    cmap : str, optional
        The color map to apply for the gradient. Defaults to 'coolwarm'.
    hide_index : bool, optional
        If True, hide the index in the output. Defaults to False.

    Returns
    -------
    pd.io.formats.style.Styler
        The styled Series or DataFrame.
    """
    # Check if input is a Series or DataFrame
    if isinstance(data, pd.Series):
        # For Series, directly apply background gradient and format
        if vmax is None:
            vmax = data.max()
        styled_data = (
            data.to_frame()
            .style.background_gradient(cmap=cmap, axis=0, vmin=-vmax, vmax=vmax)
            .format("{:.3f}")
        )

        # Hide the index if requested
        if hide_index:
            styled_data = styled_data.hide(axis="index")

    elif isinstance(data, pd.DataFrame):
        # Determine the max value for the gradient if not provided
        if vmax is None:
            vmax = data.select_dtypes(include=["number"]).max().max()

        # If no subset provided, apply to all float columns by default
        if subset is None:
            subset = pd.IndexSlice[:, data.select_dtypes(include=["float64"]).columns]

        # Apply background gradient to numeric columns and format to 3 decimal points
        styled_data = data.style.background_gradient(
            cmap=cmap, axis=None, vmin=-vmax, vmax=vmax, subset=subset
        ).format("{:.3f}", subset=subset)

        # Color boolean columns (pink for False, lightblue for True)
        bool_columns = data.select_dtypes(include=["bool"]).columns

        def color_boolean(val):
            color = "lightblue" if val else "pink"
            return f"background-color: {color}"

        # Apply the boolean-specific styling if any boolean columns exist
        if not bool_columns.empty:
            styled_data = styled_data.applymap(color_boolean, subset=bool_columns)

        # Hide the index if requested
        if hide_index:
            styled_data = styled_data.hide(axis="index")

    else:
        raise TypeError("Input must be a pandas DataFrame or Series.")

    return styled_data
