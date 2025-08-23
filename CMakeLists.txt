cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX Fortran)

# Python + pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create a common interface target for kernels that need pybind11 headers
add_library(kf_common INTERFACE)
target_link_libraries(kf_common INTERFACE pybind11::headers Python::Module)

# Fortran kernels as an object library (for linking into the Python module)
add_library(kf_fortran OBJECT src/kernel.f90)
set_property(TARGET kf_fortran PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kf_fortran PRIVATE kf_common)

# Build the Python extension module via pybind11 and link the Fortran objects
pybind11_add_module(_kernelforge MODULE
    src/bindings.cpp
    $<TARGET_OBJECTS:kf_fortran>
)

# Ensure the built filename is exactly "_kernelforge.*"
set_target_properties(_kernelforge PROPERTIES OUTPUT_NAME "_kernelforge")

# C++ kernel implementation (your new code)
add_library(kf_kernels OBJECT src/kernels.cpp)
set_property(TARGET kf_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kf_kernels PRIVATE kf_common)

# Build the Python extension module via pybind11 and link the Fortran objects
pybind11_add_module(_kernels MODULE
    src/bindings_kernels.cpp
    $<TARGET_OBJECTS:kf_kernels>
)

set_target_properties(_kernels PROPERTIES OUTPUT_NAME "_kernels")

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()
if (OpenMP_Fortran_FOUND)
  target_link_libraries(_kernelforge PRIVATE OpenMP::OpenMP_Fortran)
endif()

# Optional BLAS/LAPACK backends (enable later if needed)
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(_kernelforge PRIVATE ${ACCELERATE})
  target_link_libraries(_kernels PRIVATE ${ACCELERATE})
elseif(WIN32)
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(_kernelforge PRIVATE MKL::MKL)
  target_link_libraries(_kernels PRIVATE MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(_kernelforge PRIVATE BLAS::BLAS)
  target_link_libraries(_kernels PRIVATE BLAS::BLAS)
endif()

# Conservative optimization flags (portable wheels). Override via env if you want.
if (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  target_compile_options(kf_fortran PRIVATE -O3 -ipo -xHost -fp-model fast=2 -no-prec-div -fno-alias -qopenmp)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(kf_fortran PRIVATE -O3 -fopenmp -mcpu=native -mtune=native -ffast-math -ftree-vectorize)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kf_kernels PRIVATE -O3 -march=native -ffast-math -fopenmp -mtune=native -ftree-vectorize)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(kf_kernels PRIVATE -O3 -qopt-report=3 -qopenmp -xHost)
endif()

# Install the compiled extension into the Python package and the Python shim
install(TARGETS _kernelforge _kernels
  LIBRARY DESTINATION kernelforge   # Linux/macOS
  RUNTIME DESTINATION kernelforge   # Windows (.pyd)
)
install(FILES python/kernelforge/__init__.py DESTINATION kernelforge)

