cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX Fortran)

# Python build plumbing
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# For Fortran (object library to link into the Python extension)
add_library(kf_fortran OBJECT src/kernel.f90)

# Python extension module (pybind11) linked with Fortran objects
pybind11_add_module(kernelforge MODULE src/bindings.cpp $<TARGET_OBJECTS:kf_fortran>)
set_target_properties(kernelforge PROPERTIES OUTPUT_NAME "kernelforge")

# If you later need BLAS/LAPACK, add conditional linking here:
# if(APPLE)
#   find_library(ACCELERATE Accelerate REQUIRED)
#   target_link_libraries(kernelforge PRIVATE ${ACCELERATE})
# elseif(WIN32)
#   find_package(MKL CONFIG REQUIRED)
#   target_link_libraries(kernelforge PRIVATE MKL::MKL)
# else()
#   find_package(BLAS REQUIRED)
#   target_link_libraries(kernelforge PRIVATE BLAS::BLAS)
# endif()

# OpenMP (optional; only if kernels use it)
find_package(OpenMP)
if (OpenMP_Fortran_FOUND)
  target_link_libraries(kernelforge PRIVATE OpenMP::OpenMP_Fortran)
endif()

# Compiler flags: conservative defaults (portable wheels). Users can override via env.
if (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  target_compile_options(kf_fortran PRIVATE -O3)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(kf_fortran PRIVATE -O3)
endif()

# Install the compiled module into the Python package directory
install(TARGETS kernelforge LIBRARY DESTINATION kernelforge)
# Also install the pure-Python shim
install(FILES python/kernelforge/__init__.py DESTINATION kernelforge)

