# # Thin shim so "import kernelforge" loads the compiled extension
# from importlib import import_module as _import_module
# _mod = _import_module("_kernelforge")  # compiled module built by CMake/pybind11
#
# inverse_distance = _mod.inverse_distance
#
# __all__ = ["inverse_distance"]
from ._kernelforge import inverse_distance, simple_distance
__all__ = ["inverse_distance", "simple_distance"]
