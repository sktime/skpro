"""Common configuration for the skpro registry."""

# modules to ignore in skpro for lookup
MODULES_TO_IGNORE = (
    "conftest",
    "tests",
    "setup",
    "contrib",
    "utils",
    "all",
    "registry",
    "libs",
)

# modules to ignore in scikit-learn for lookup
MODULES_TO_IGNORE_SKLEARN = [
    "array_api_compat",
    "conftest",
    "tests",
    "experimental",
]
