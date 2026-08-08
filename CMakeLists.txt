cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KF_BLAS_ILP64 "Use 64-bit integers for BLAS/LAPACK (ILP64)" OFF)
# CUDA/Torch extensions are opt-in. Default OFF so CPU wheels and plain
# `make install-*` never accidentally pull in nvcc/torch.
option(KF_WITH_CUDA "Build CUDA + Torch extensions" OFF)
# Companion-package mode: build/install only CUDA extension modules (no CPU
# kernels, no Python package files). Used by packaging/kernelforge-cuda.
option(KF_CUDA_ONLY "Build only CUDA companion extensions" OFF)

if(KF_CUDA_ONLY AND NOT KF_WITH_CUDA)
  message(FATAL_ERROR "KF_CUDA_ONLY=ON requires KF_WITH_CUDA=ON")
endif()

if(KF_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "KF_WITH_CUDA=ON but no CUDA compiler was found. "
      "Install a CUDA toolkit / set CMAKE_CUDA_COMPILER, or build with -DKF_WITH_CUDA=OFF.")
  endif()
  enable_language(CUDA)
endif()

if(APPLE)

  # Homebrew libomp on macos-26 runners is built for macOS 26; keep the
  # deployment target at least that high so delocate accepts the vendored dylib.
  # Prefer an explicit env/cache value when present (cibuildwheel sets it).
  if(DEFINED ENV{MACOSX_DEPLOYMENT_TARGET} AND NOT "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "$ENV{MACOSX_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
  elseif(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "" FORCE)
  endif()
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
if(NOT KF_CUDA_ONLY)
  kf_add_cpp_module(global_kernels
    SOURCES  src/global_kernels.cpp
    BINDINGS src/global_kernels_bindings.cpp)

  kf_add_cpp_module(local_kernels
    SOURCES  src/local_kernels.cpp
    BINDINGS src/local_kernels_bindings.cpp)

  kf_add_cpp_module(fchl19_repr
    SOURCES  src/fchl19_repr.cpp
    BINDINGS src/fchl19_repr_bindings.cpp)

  kf_add_cpp_module(fchl19v2_repr
    SOURCES  src/fchl19v2_repr.cpp
    BINDINGS src/fchl19v2_repr_bindings.cpp)

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
    SOURCES  src/fchl18_kernel.cpp src/fchl18_repr.cpp
             src/fchl18_scalar_kernels.cpp
             src/fchl18_jacobian_kernels.cpp
             src/fchl18_hessian_kernels.cpp
             src/fchl18_full_kernels.cpp
    BINDINGS src/fchl18_kernel_bindings.cpp)
elseif(KF_WITH_CUDA)
  # cuda_local_kernels links $<TARGET_OBJECTS:kf_local_kernels>. Build the
  # object library only (no CPU Python module) for the companion package.
  add_library(kf_local_kernels OBJECT src/local_kernels.cpp)
  target_link_libraries(kf_local_kernels PRIVATE kf_common kf_blas OpenMP::OpenMP_CXX)
  kf_apply_cxx_flags(kf_local_kernels)
endif()

# ---- Optional CUDA + PyTorch extensions (KF_WITH_CUDA) ----------------------
if(KF_WITH_CUDA)
  # Locate PyTorch CMake config.  scikit-build-core builds use an isolated
  # Python that may lack torch.  Try the project venv (.venv) first, then
  # the build Python, then common conda/venv locations.
  set(_TORCH_CMAKE_PREFIX "")
  foreach(_py_candidate
      "${CMAKE_SOURCE_DIR}/.venv/bin/python"
      "${Python_EXECUTABLE}"
      "$ENV{VIRTUAL_ENV}/bin/python"
      "$ENV{CONDA_PREFIX}/bin/python")
    if(EXISTS "${_py_candidate}" AND NOT _TORCH_CMAKE_PREFIX)
      execute_process(
        COMMAND "${_py_candidate}" -c
                "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _TORCH_RESULT
        ERROR_QUIET)
      if(NOT _TORCH_RESULT EQUAL 0)
        set(_TORCH_CMAKE_PREFIX "")
      endif()
    endif()
  endforeach()

  if(_TORCH_CMAKE_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${_TORCH_CMAKE_PREFIX}")
    find_package(Torch REQUIRED)
  else()
    message(FATAL_ERROR
      "KF_WITH_CUDA=ON but PyTorch was not found. "
      "Install torch in the project venv (or set CMAKE_PREFIX_PATH), "
      "or build with -DKF_WITH_CUDA=OFF.")
  endif()

  # Find CUDAToolkit *after* Torch. Torch's Caffe2 cuda.cmake re-runs
  # FindCUDA / FindCUDAToolkit and can drop CUDA::cusolver (and friends) on
  # CUDA 13 manylinux images. Clear the prior result and re-resolve with
  # required components so CUDA::cublas / CUDA::cusolver exist for linking.
  unset(CUDAToolkit_FOUND)
  unset(CUDA_cusolver_LIBRARY CACHE)
  unset(CUDA_cublas_LIBRARY CACHE)
  unset(CUDA_cudart_LIBRARY CACHE)
  unset(CUDA_nvrtc_LIBRARY CACHE)
  if(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}")
  elseif(EXISTS "/usr/local/cuda")
    set(CUDAToolkit_ROOT "/usr/local/cuda")
  endif()
  find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas cusolver nvrtc)
  if(NOT TARGET CUDA::cusolver)
    message(FATAL_ERROR
      "find_package(CUDAToolkit) succeeded but CUDA::cusolver is missing. "
      "Install cuda-libraries-devel for this CUDA toolkit.")
  endif()

  message(STATUS
    "CUDA ${CMAKE_CUDA_COMPILER_VERSION} + Torch ${Torch_VERSION} — "
    "building CUDA kernel extensions")

  # libtorch_python.so provides the pybind11 type_caster<at::Tensor>
  # specialisations; without it the modules fail to load at runtime.
  find_library(_LIBTORCH_PYTHON torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH)

  # ---- cuda_global_kernels ----
  pybind11_add_module(cuda_global_kernels MODULE
    src/cuda_global_kernels.cu
    src/cuda_global_kernels_bindings.cpp
    src/curfp_handle.cpp
    src/curfp_ssfrk.cpp
    src/curfp_ssfr2.cpp
    src/curfp_spftrf.cpp
    src/curfp_spftrs.cpp)

      # Torch include dirs must precede the standalone pybind11 headers so that
      # torch/extension.h's bundled pybind11 wins the header-guard race.
      target_include_directories(cuda_global_kernels BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_global_kernels PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}"
        CUDA::cublas
        CUDA::cusolver)

      set_property(TARGET cuda_global_kernels PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_global_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
      # Torch injects nvcc arch flags via TORCH_CUDA_ARCH_LIST and explicitly
      # ignores CMAKE_CUDA_ARCHITECTURES. Disable CMake's own arch emission for
      # these Torch-backed modules to avoid duplicate fatbin generation.
      set_property(TARGET cuda_global_kernels PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_global_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_global_kernels)

      # ---- cuda_local_kernels ----
      pybind11_add_module(cuda_local_kernels MODULE
        src/cuda_local_kernels.cu
        src/cuda_local_kernels_bindings.cpp
        $<TARGET_OBJECTS:kf_local_kernels>)

      target_include_directories(cuda_local_kernels BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_local_kernels PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}"
        CUDA::cublas
        kf_blas
        OpenMP::OpenMP_CXX)

      set_property(TARGET cuda_local_kernels PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_local_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_local_kernels PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_local_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_local_kernels)

      # ---- cuda_fchl18_kernel ----
      pybind11_add_module(cuda_fchl18_kernel MODULE
        src/cuda_fchl18_kernel.cu
        src/cuda_fchl18_jacobian.cu
        src/cuda_fchl18_hessian.cu
        src/cuda_fchl18_full.cu
        src/cuda_fchl18_kernel_bindings.cpp)

      target_include_directories(cuda_fchl18_kernel BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_fchl18_kernel PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}")

      set_property(TARGET cuda_fchl18_kernel PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_fchl18_kernel PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_fchl18_kernel PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_fchl18_kernel PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_fchl18_kernel)

      # ---- cuda_fchl18_repr ----
      pybind11_add_module(cuda_fchl18_repr MODULE
        src/cuda_fchl18_repr.cu
        src/cuda_fchl18_repr_bindings.cpp)

      target_include_directories(cuda_fchl18_repr BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_fchl18_repr PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}")

      set_property(TARGET cuda_fchl18_repr PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_fchl18_repr PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_fchl18_repr PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_fchl18_repr PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_fchl18_repr)

      # ---- cuda_invdist_repr ----
      pybind11_add_module(cuda_invdist_repr MODULE
        src/cuda_invdist_repr.cu
        src/cuda_invdist_repr_bindings.cpp)

      target_include_directories(cuda_invdist_repr BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_invdist_repr PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}")

      set_property(TARGET cuda_invdist_repr PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_invdist_repr PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_invdist_repr PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_invdist_repr PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_invdist_repr)

      # ---- cuda_rff_features ----
      pybind11_add_module(cuda_rff_features MODULE
        src/cuda_rff_features.cu
        src/cuda_rff_features_bindings.cpp
        src/curfp_handle.cpp
        src/curfp_ssfrk.cpp
        src/curfp_spftrf.cpp
        src/curfp_spftrs.cpp)

      target_include_directories(cuda_rff_features BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_rff_features PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}"
        CUDA::cublas
        CUDA::cusolver)

      set_property(TARGET cuda_rff_features PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_rff_features PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_rff_features PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_rff_features PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_rff_features)

      # ---- cuda_fchl19_repr ----
      pybind11_add_module(cuda_fchl19_repr MODULE
        src/cuda_fchl19_repr.cu
        src/cuda_fchl19_repr_bindings.cpp)

      target_include_directories(cuda_fchl19_repr BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_fchl19_repr PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}")

      set_property(TARGET cuda_fchl19_repr PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_fchl19_repr PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_fchl19_repr PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_fchl19_repr PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_fchl19_repr)

      # ---- cuda_solvers ----
      pybind11_add_module(cuda_solvers MODULE
        src/cuda_solvers.cu
        src/cuda_solvers_bindings.cpp)

      target_include_directories(cuda_solvers BEFORE PRIVATE
        ${TORCH_INCLUDE_DIRS}
        src)

      target_link_libraries(cuda_solvers PRIVATE
        "${TORCH_LIBRARIES}"
        "${_LIBTORCH_PYTHON}"
        CUDA::cublas
        CUDA::cusolver)

      set_property(TARGET cuda_solvers PROPERTY CUDA_STANDARD 17)
      set_property(TARGET cuda_solvers PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cuda_solvers PROPERTY CUDA_ARCHITECTURES OFF)

      target_compile_options(cuda_solvers PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>)

      list(APPEND _KF_ALL_MODULES cuda_solvers)

else()
  message(STATUS "KF_WITH_CUDA=OFF — skipping CUDA kernel extensions")
endif()

# ---- Install ----------------------------------------------------------------
if(NOT _KF_ALL_MODULES)
  message(FATAL_ERROR "No extension modules were configured to build")
endif()

install(TARGETS ${_KF_ALL_MODULES}
  LIBRARY DESTINATION kernelforge   # Linux/macOS
  RUNTIME DESTINATION kernelforge   # Windows (.pyd)
)

if(KF_WITH_CUDA)
  file(GLOB _KF_CUDA_STUBS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/python/kernelforge/cuda_*.pyi")
  if(_KF_CUDA_STUBS)
    install(FILES ${_KF_CUDA_STUBS} DESTINATION kernelforge)
  endif()
endif()

# Full package installs the Python package via wheel.packages; companion mode
# only drops CUDA extension modules into an existing kernelforge/ install.
if(NOT KF_CUDA_ONLY)
  install(FILES python/kernelforge/__init__.py DESTINATION kernelforge)
endif()
