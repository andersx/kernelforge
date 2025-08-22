cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX Fortran)

# Python + pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Fortran kernels as an object library (for linking into the Python module)
add_library(kf_fortran OBJECT src/kernel.f90)
set_property(TARGET kf_fortran PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build the Python extension module via pybind11 and link the Fortran objects
pybind11_add_module(_kernelforge MODULE
    src/bindings.cpp
    $<TARGET_OBJECTS:kf_fortran>
)
# Ensure the built filename is exactly "_kernelforge.*"
set_target_properties(_kernelforge PROPERTIES OUTPUT_NAME "_kernelforge")

# Optional: OpenMP for Fortran if you use it
find_package(OpenMP)
if (OpenMP_Fortran_FOUND)
  target_link_libraries(_kernelforge PRIVATE OpenMP::OpenMP_Fortran)
endif()

# Optional BLAS/LAPACK backends (enable later if needed)
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(_kernelforge PRIVATE ${ACCELERATE})
elseif(WIN32)
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(_kernelforge PRIVATE MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(_kernelforge PRIVATE BLAS::BLAS)
endif()

# Conservative optimization flags (portable wheels). Override via env if you want.
if (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  target_compile_options(kf_fortran PRIVATE -O3 -ipo -xHost -fp-model fast=2 -no-prec-div -fno-alias -qopenmp)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(kf_fortran PRIVATE -O3 -fopenmp -mcpu=native -mtune=native -ffast-math -ftree-vectorize)
endif()

# Install the compiled extension into the Python package and the Python shim
install(TARGETS _kernelforge
  LIBRARY DESTINATION kernelforge   # Linux/macOS
  RUNTIME DESTINATION kernelforge   # Windows (.pyd)
)
install(FILES python/kernelforge/__init__.py DESTINATION kernelforge)

