"""Generate estimator data for the estimator overview page.

This script collects information about all estimators in skpro
and generates JavaScript data for the estimator overview page.
"""

import json
import sys
from pathlib import Path


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
        print("Warning: Could not import skpro.registry.all_objects", file=sys.stderr)
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
    except Exception as e:
        print(f"Warning: Could not retrieve estimator data: {e}", file=sys.stderr)
        return []

    if df is None or df.empty:
        print("Warning: No estimators found", file=sys.stderr)
        return []

    estimators = []

    for idx, row in df.iterrows():
        name = row.get("name", "Unknown")
        obj = row.get("objects")
        obj_type = row.get("object_type", "unknown")

        # Get module path
        if obj is not None:
            module = obj.__module__ if hasattr(obj, "__module__") else "unknown"
        else:
            module = "unknown"

        # Collect key tags
        tags = []
        if obj_type:
            tags.append(f"object_type:{obj_type}")

        # Add capability tags
        if row.get("capability:survival"):
            tags.append("capability:survival")

        if row.get("handles_missing_data"):
            tags.append("handles_missing_data")

        if row.get("requires_y") is False:
            tags.append("unsupervised")
        elif row.get("requires_y"):
            tags.append("supervised")

        if row.get("handles_multioutput"):
            tags.append("multioutput")

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
    """Main entry point for the script."""
    estimators = generate_estimator_data()

    if not estimators:
        print("Warning: No estimators could be generated", file=sys.stderr)
        # Return empty data structure
        return "window.estimatorData = [];"

    js_code = generate_javascript_code(estimators)
    print(js_code)


if __name__ == "__main__":
    main()
