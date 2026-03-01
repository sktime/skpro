"""Generate estimator data for the estimator overview page.

This script collects information about all estimators in skpro
and generates JavaScript data for the estimator overview page.
"""

import json
import sys


def generate_estimator_data():
    """Generate estimator data for the overview page.

    Returns
    -------
    list of dict
        List of estimator information dictionaries
    """
    try:
        from skpro.registry import all_objects
    except ImportError:
        sys.stderr.write("Warning: Could not import skpro.registry.all_objects\n")
        return []

    # Get all objects with tags
    try:
        df = all_objects(
            as_dataframe=True,
            return_tags=[
                "object_type",
                "estimator_type",
                "capability:survival",
                "handles_missing_data",
                "requires_y",
                "handles_multioutput",
            ],
            suppress_import_stdout=True,
        )
    except Exception as e:  # noqa: B902
        sys.stderr.write(f"Warning: Could not retrieve estimator data: {e}\n")
        return []

    if df is None or df.empty:
        sys.stderr.write("Warning: No estimators found\n")
        return []

    estimators = []

    for _, row in df.iterrows():
        name = row.get("name", "Unknown")
        # all_objects uses "object" column; keep fallback for older naming
        obj = row.get("object") if "object" in row.index else row.get("objects")
        obj_type = row.get("object_type", "unknown")

        # Get module path from the actual object class
        if obj is not None and hasattr(obj, "__module__"):
            module = obj.__module__
        else:
            module = "unknown"

        # Collect all tags with their values
        tags = {}
        if obj_type:
            tags["object_type"] = obj_type

        # Add capability tags with their values
        if "capability:survival" in row.index:
            tags["capability:survival"] = row.get("capability:survival")

        if "handles_missing_data" in row.index:
            tags["handles_missing_data"] = row.get("handles_missing_data")

        if "requires_y" in row.index:
            tags["requires_y"] = row.get("requires_y")

        if "handles_multioutput" in row.index:
            tags["handles_multioutput"] = row.get("handles_multioutput")

        estimators.append(
            {
                "name": name,
                "object_type": obj_type,
                "module": module,
                "tags": tags,
            }
        )

    return sorted(estimators, key=lambda x: x["name"])


def generate_javascript_code(estimators):
    """Generate JavaScript code to inject estimator data.

    Parameters
    ----------
    estimators : list of dict
        List of estimator information dictionaries

    Returns
    -------
    str
        JavaScript code that sets window.estimatorData
    """
    json_data = json.dumps(estimators, indent=2)
    return f"""
// Estimator data for the overview page
window.estimatorData = {json_data};
"""


def main():
    """Generate and output estimator data as JavaScript code."""
    estimators = generate_estimator_data()

    if not estimators:
        sys.stderr.write("Warning: No estimators could be generated\n")
        return

    js_code = generate_javascript_code(estimators)
    sys.stdout.write(js_code)


if __name__ == "__main__":
    main()
