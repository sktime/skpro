# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Placeholder registry utilities for optional soft dependencies."""

import importlib
from functools import wraps

def _placeholder_record(dependency, import_path):
    """Decorator to mark a class as a placeholder for a soft dependency.
    
    If the soft dependency is installed, the class is transparently
    replaced with the actual class from the external package.
    If it is not installed, the stub class is returned. Any attempt
    to instantiate it will raise an ImportError explaining how to
    install the soft dependency.
    """
    def decorator(cls):
        from skbase.utils.dependencies import _check_soft_dependencies
        
        # Check if the soft dependency is installed in the environment
        if _check_soft_dependencies(dependency, severity="none"):
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                real_class = getattr(module, class_name)
                return real_class
            except (ImportError, AttributeError):
                pass
        
        # If not installed, wrap __init__ to raise a clear soft-dependency error
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            from skbase.utils.dependencies import _check_soft_dependencies
            _check_soft_dependencies(dependency, severity="error", obj=self)
            original_init(self, *args, **kwargs)
            
        cls.__init__ = new_init
        return cls
        
    return decorator
