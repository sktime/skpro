"""Generate estimator metadata for the dynamic estimator overview page.

This script is called during the Sphinx documentation build process to generate
JavaScript data containing estimator information for the interactive overview page.

The generated file is NOT committed to the repository - it's created fresh during
each build to ensure estimator registry changes are always reflected.
"""

import json
import logging
import os
import sys

# Add parent directory to path to import skpro
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skpro.registry import all_objects  # noqa: E402

# Set up logging for build output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO_GENERATED_DIR = os.path.join(
    REPO_ROOT, "docs", "source", "api_reference", "auto_generated"
)


def _serialize_value(value):
    """Convert a value to JSON-serializable format.

    Parameters
    ----------
    value : any
        Value to serialize

    Returns
    -------
    JSON-serializable value
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    # For non-serializable types (classes, etc.), convert to string
    return str(value)


def _infer_object_type(tags_dict, module):
    """Infer object type from tags and module path.

    Parameters
    ----------
    tags_dict : dict
        Class tag dictionary.
    module : str
        Python module path of the object.

    Returns
    -------
    str
        Inferred object type.
    """
    obj_type_val = tags_dict.get("object_type")
    if obj_type_val:
        if isinstance(obj_type_val, list):
            return str(obj_type_val[0])
        return str(obj_type_val)

    estimator_type_val = tags_dict.get("estimator_type")
    if estimator_type_val:
        return str(estimator_type_val)

    if module.startswith("skpro.distributions"):
        return "distribution"
    if module.startswith("skpro.metrics"):
        return "metric"
    if module.startswith("skpro.survival"):
        return "survival"
    if module.startswith("skpro.regression"):
        return "regressor_proba"

    return "unknown"


def _infer_doc_path(module, name):
    """Infer autosummary doc path used in api_reference/auto_generated.

    Parameters
    ----------
    module : str
        Python module path of the object.
    name : str
        Class name.

    Returns
    -------
    str
        Dotted doc path without ``.html`` suffix.
    """
    candidates = []

    if module:
        candidates.append(f"{module}.{name}")

        parts = module.split(".")
        if parts and parts[-1].startswith("_"):
            candidates.append(".".join(parts[:-1] + [name]))

    if module.startswith("skpro.distributions"):
        candidates.append(f"skpro.distributions.{name}")

    if module.startswith("skpro.metrics"):
        candidates.append(f"skpro.metrics.{name}")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        rst_path = os.path.join(AUTO_GENERATED_DIR, f"{candidate}.rst")
        if os.path.exists(rst_path):
            return candidate

    if unique_candidates:
        return unique_candidates[0]

    return f"skpro.{name}"


def _fallback_api_url(object_type):
    """Fallback API URL if no auto-generated detail page is available."""
    type_map = {
        "regressor_proba": "regression",
        "distribution": "distributions",
        "metric": "metrics",
        "survival": "survival",
    }
    section = type_map.get(object_type, "api_reference")
    return f"api_reference/{section}.html"


def generate_estimator_data(output_file):
    """Generate estimator metadata JavaScript file.

    Parameters
    ----------
    output_file : str
        Path to output JavaScript file.
    """
    try:
        # Get all registered objects (estimators, distributions, etc.)
        estimators_df = all_objects(as_dataframe=True)

        if estimators_df.empty:
            logger.warning("No estimators found in registry")
            estimators_data = []
        else:
            estimators_data = []

            # Convert dataframe to list of dicts with necessary fields
            for _idx, row in estimators_df.iterrows():
                name = row.get("name", "Unknown")
                estimator_class = row.get("object")

                # Extract module from class
                module = (
                    estimator_class.__module__
                    if hasattr(estimator_class, "__module__")
                    else "unknown"
                )

                # Initialize with object_type inferred from tags/module
                object_type = "unknown"
                tags = {}

                # Extract tags from _tags attribute
                if hasattr(estimator_class, "_tags"):
                    tags_dict = estimator_class._tags or {}

                    object_type = _infer_object_type(tags_dict=tags_dict, module=module)

                    # Extract all tags that are meaningful
                    for tag_name, tag_value in tags_dict.items():
                        # Skip internal tags and None values
                        if (
                            tag_name.startswith("_")
                            or tag_value is None
                            or tag_name == "object_type"
                        ):
                            continue

                        # Skip empty strings and empty lists
                        if tag_value == "" or tag_value == []:
                            continue

                        # Serialize the value
                        tags[tag_name] = _serialize_value(tag_value)
                else:
                    object_type = _infer_object_type(tags_dict={}, module=module)

                est_info = {
                    "name": name,
                    "object_type": object_type,
                    "module": module,
                    "doc_path": _infer_doc_path(module=module, name=name),
                    "tags": tags,
                }

                rst_path = os.path.join(
                    AUTO_GENERATED_DIR, f"{est_info['doc_path']}.rst"
                )
                if os.path.exists(rst_path):
                    est_info[
                        "doc_url"
                    ] = f"api_reference/auto_generated/{est_info['doc_path']}.html"
                else:
                    est_info["doc_url"] = _fallback_api_url(object_type)

                estimators_data.append(est_info)

        # Generate JavaScript file
        estimator_json = json.dumps(estimators_data, indent=2)
        comment_line = (
            "// Auto-generated estimator data for the dynamic estimator overview page."
        )
        js_content = (
            f"{comment_line}\n"
            "// This file is regenerated during each documentation build.\n"
            "\n"
            f"var estimatorData = {estimator_json};\n"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the file
        with open(output_file, "w") as f:
            f.write(js_content)

        logger.info(f"Generated estimator data: {output_file}")
        logger.info(f"Total estimators: {len(estimators_data)}")

        return True

    except Exception as e:  # noqa: B902
        logger.error(f"Error generating estimator data: {e}")
        logger.error("The estimator overview page may not function properly.")
        import traceback

        traceback.print_exc()
        # Don't fail the build - just warn
        return False


if __name__ == "__main__":
    # Default output location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_path = os.path.join(repo_root, "docs", "_build", "estimator_data.js")

    # Allow overriding via command-line argument
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    generate_estimator_data(output_path)
