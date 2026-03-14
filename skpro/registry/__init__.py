"""Registry and lookup functionality."""

from skpro.registry._base_classes import (
    get_base_class_for_str,
    get_base_class_list,
    get_base_class_lookup,
    get_base_class_register,
    get_obj_scitype_list,
    get_test_class_for_str,
)
from skpro.registry._craft import craft, deps, imports
from skpro.registry._lookup import all_objects, all_tags
from skpro.registry._scitype import scitype
from skpro.registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
    check_tag_is_valid,
)
from skpro.registry.test_class_register import (
    get_test_class_registry,
    get_test_classes_for_obj,
)

__all__ = [
    "OBJECT_TAG_LIST",
    "OBJECT_TAG_REGISTER",
    "all_objects",
    "all_tags",
    "check_tag_is_valid",
    "craft",
    "deps",
    "get_base_class_for_str",
    "get_base_class_list",
    "get_base_class_lookup",
    "get_base_class_register",
    "get_obj_scitype_list",
    "get_test_class_for_str",
    "get_test_class_registry",
    "get_test_classes_for_obj",
    "imports",
    "scitype",
]
