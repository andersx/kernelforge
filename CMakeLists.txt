cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KF_BLAS_ILP64 "Use 64-bit integers for BLAS/LAPACK (ILP64)" OFF)

if(APPLE)

  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "" FORCE)
  add_compile_definitions(ACCELERATE_NEW_LAPACK)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

  # Necessary to compile with -Accelerate, homebrew clang and openmp
  # Took me way too long to figure out
  add_compile_options(-stdlib=libc++)
  add_link_options(
    -stdlib=libc++
    -L/opt/homebrew/opt/llvm/lib/c++
    -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++
  )

endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
  if (APPLE)
    # Apple/Homebrew Clang requires explicit flags
    add_compile_options(-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include)
    add_link_options(-L/opt/homebrew/opt/libomp/lib -lomp)
  else()
    add_compile_options(${OpenMP_CXX_FLAGS})
    add_link_options(${OpenMP_CXX_FLAGS})
  endif()
endif()

# ---- BLAS/LAPACK backend detection -------------------------------------------
# BLAS vendor selection: AUTO (default, tries MKL then OpenBLAS), MKL, OpenBLAS
set(KF_BLAS_VENDOR "AUTO" CACHE STRING "BLAS vendor: AUTO, MKL, OpenBLAS")
set_property(CACHE KF_BLAS_VENDOR PROPERTY STRINGS AUTO MKL OpenBLAS)

if(KF_BLAS_ILP64)
  add_compile_definitions(KF_BLAS_ILP64)
endif()

# Helper: Configure MKL threading based on compiler
function(kf_configure_mkl)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel|IntelLLVM")
    set(MKL_THREADING intel_thread PARENT_SCOPE)  # Intel OpenMP (libiomp5)
  else()
    set(MKL_THREADING gnu_thread PARENT_SCOPE)    # GNU OpenMP (libgomp)
  endif()
  set(MKL_LINK dynamic PARENT_SCOPE)
  if(KF_BLAS_ILP64)
    set(MKL_INTERFACE ilp64 PARENT_SCOPE)
  else()
    set(MKL_INTERFACE lp64 PARENT_SCOPE)
  endif()
endfunction()

# Helper: Find OpenBLAS ILP64 include directory
function(kf_find_openblas_includes)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(OPENBLAS64 QUIET openblas64)
  endif()

  if(OPENBLAS64_FOUND)
    set(KF_OPENBLAS_INCLUDES ${OPENBLAS64_INCLUDE_DIRS} PARENT_SCOPE)
    set(KF_OPENBLAS_SOURCE "pkg-config" PARENT_SCOPE)
  else()
    find_path(KF_OPENBLAS_INCLUDE cblas.h
      PATHS
        /usr/include/${CMAKE_LIBRARY_ARCHITECTURE}/openblas64-pthread
        /usr/include/${CMAKE_LIBRARY_ARCHITECTURE}/openblas64
        /usr/include/openblas64
      NO_DEFAULT_PATH)
    if(KF_OPENBLAS_INCLUDE)
      set(KF_OPENBLAS_INCLUDES ${KF_OPENBLAS_INCLUDE} PARENT_SCOPE)
      set(KF_OPENBLAS_SOURCE "fallback" PARENT_SCOPE)
    endif()
  endif()
endfunction()

# Detect BLAS backend
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  set(KF_BLAS_BACKEND "Accelerate")
  set(KF_BLAS_LIBS ${ACCELERATE})
  message(STATUS "BLAS backend: Accelerate (Apple)")

elseif(KF_BLAS_VENDOR STREQUAL "MKL" OR
       (KF_BLAS_VENDOR STREQUAL "AUTO" AND NOT KF_BLAS_VENDOR STREQUAL "OpenBLAS"))
  kf_configure_mkl()
  list(PREPEND CMAKE_PREFIX_PATH /opt/intel/oneapi/mkl/latest)
  find_package(MKL QUIET)

  if(MKL_FOUND)
    add_compile_definitions(KF_USE_MKL)
    set(KF_BLAS_BACKEND "MKL")
    set(KF_BLAS_LIBS MKL::MKL)
    message(STATUS "BLAS backend: Intel MKL (${MKL_INTERFACE}, ${MKL_THREADING})")
  elseif(KF_BLAS_VENDOR STREQUAL "MKL")
    message(FATAL_ERROR "Intel MKL explicitly requested but not found.")
  endif()
endif()

# Fallback to OpenBLAS/generic BLAS
if(NOT DEFINED KF_BLAS_BACKEND)
  if(KF_BLAS_VENDOR STREQUAL "OpenBLAS")
    set(BLA_VENDOR OpenBLAS)
  endif()
  if(KF_BLAS_ILP64)
    set(BLA_SIZEOF_INTEGER 8)
  endif()

  find_package(BLAS REQUIRED)
  set(KF_BLAS_BACKEND "OpenBLAS")
  set(KF_BLAS_LIBS BLAS::BLAS)
  message(STATUS "BLAS backend: OpenBLAS/generic BLAS")

  if(KF_BLAS_ILP64)
    kf_find_openblas_includes()
    if(DEFINED KF_OPENBLAS_INCLUDES)
      message(STATUS "ILP64 OpenBLAS include dir (${KF_OPENBLAS_SOURCE}): ${KF_OPENBLAS_INCLUDES}")
    endif()
  endif()
endif()

# Common interface libraries
add_library(kf_common INTERFACE)
target_link_libraries(kf_common INTERFACE pybind11::headers Python::Module)

add_library(kf_blas INTERFACE)
target_link_libraries(kf_blas INTERFACE ${KF_BLAS_LIBS})
if(DEFINED KF_OPENBLAS_INCLUDES)
  target_include_directories(kf_blas SYSTEM INTERFACE ${KF_OPENBLAS_INCLUDES})
endif()

# ---- Compiler optimization flags ---------------------------------------------
option(KF_USE_NATIVE "Enable -march/-mcpu=native style flags" OFF)

function(kf_apply_cxx_flags tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE
      -O3 -ffast-math -ftree-vectorize -fopenmp
      $<$<BOOL:${KF_USE_NATIVE}>:-mcpu=native -mtune=native>
    )
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel|IntelLLVM")
    # Intel classic (icc/icpc) and oneAPI (icx/icpx) compilers
    target_compile_options(${tgt} PRIVATE
      -O3 -ffast-math -qopenmp
      $<$<BOOL:${KF_USE_NATIVE}>:-xHost>
    )
  endif()
endfunction()

# ---- Module helper -----------------------------------------------------------
# Create a C++ object library + pybind11 module pair:
#   kf_add_cpp_module(<name> SOURCES src... BINDINGS bind...)
#   -> object lib: kf_<name>  (with optimization flags, BLAS, OpenMP)
#   -> module:     <name>     (pybind11 module linked to BLAS + OpenMP)
set(_KF_ALL_MODULES "")

function(kf_add_cpp_module name)
  cmake_parse_arguments(ARG "" "" "SOURCES;BINDINGS" ${ARGN})
  set(obj kf_${name})

  # Object library: compiled with optimization flags
  add_library(${obj} OBJECT ${ARG_SOURCES})
  target_link_libraries(${obj} PRIVATE kf_common kf_blas OpenMP::OpenMP_CXX)
  kf_apply_cxx_flags(${obj})

  # Pybind11 module: links object library + BLAS + OpenMP
  pybind11_add_module(${name} MODULE ${ARG_BINDINGS} $<TARGET_OBJECTS:${obj}>)
  set_target_properties(${name} PROPERTIES OUTPUT_NAME "${name}")
  target_link_libraries(${name} PRIVATE kf_blas OpenMP::OpenMP_CXX)

  list(APPEND _KF_ALL_MODULES ${name})
  set(_KF_ALL_MODULES "${_KF_ALL_MODULES}" PARENT_SCOPE)
endfunction()

# ---- C++ modules -------------------------------------------------------------
kf_add_cpp_module(global_kernels
  SOURCES  src/global_kernels.cpp
  BINDINGS src/global_kernels_bindings.cpp)

kf_add_cpp_module(local_kernels
  SOURCES  src/local_kernels.cpp
  BINDINGS src/local_kernels_bindings.cpp)

kf_add_cpp_module(fchl19_repr
  SOURCES  src/fchl19_repr.cpp
  BINDINGS src/fchl19_repr_bindings.cpp)

kf_add_cpp_module(invdist_repr
  SOURCES  src/invdist_repr.cpp
  BINDINGS src/invdist_repr_bindings.cpp)

kf_add_cpp_module(kernelmath
  SOURCES  src/math.cpp
  BINDINGS src/math_bindings.cpp)

kf_add_cpp_module(kitchen_sinks
  SOURCES  src/rff_features.cpp src/rff_elemental.cpp
  BINDINGS src/rff_features_bindings.cpp src/rff_elemental_bindings.cpp)

kf_add_cpp_module(fchl18_repr
  SOURCES  src/fchl18_repr.cpp
  BINDINGS src/fchl18_repr_bindings.cpp)

kf_add_cpp_module(fchl18_kernel
  SOURCES  src/fchl18_kernel.cpp
  BINDINGS src/fchl18_kernel_bindings.cpp)

# ---- Install ----------------------------------------------------------------
install(TARGETS ${_KF_ALL_MODULES}
  LIBRARY DESTINATION kernelforge   # Linux/macOS
  RUNTIME DESTINATION kernelforge   # Windows (.pyd)
)
install(FILES python/kernelforge/__init__.py DESTINATION kernelforge)
