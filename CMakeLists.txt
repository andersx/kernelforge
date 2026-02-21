cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ILP64 BLAS/LAPACK support (64-bit integers)
option(KF_BLAS_ILP64 "Use 64-bit integers for BLAS/LAPACK (ILP64)" OFF)

# Platform tweaks
if(APPLE)

  # Required for "new lapack" in Accelerate
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

# Position-independent code for all targets (helps for Python extensions)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
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

# BLAS vendor selection (Linux only; macOS always uses Accelerate)
# Values: AUTO (default), MKL, OpenBLAS
set(KF_BLAS_VENDOR "AUTO" CACHE STRING "BLAS vendor: AUTO, MKL, OpenBLAS")
set_property(CACHE KF_BLAS_VENDOR PROPERTY STRINGS AUTO MKL OpenBLAS)

# ILP64 configuration (64-bit integers for BLAS/LAPACK)
if(KF_BLAS_ILP64)
  add_compile_definitions(KF_BLAS_ILP64)
endif()

# ---- BLAS/LAPACK backend detection -------------------------------------------
if(APPLE)
  # macOS: Always use Accelerate framework
  # ILP64 defines are set in blas_config.h (ACCELERATE_BLAS_ILP64, ACCELERATE_LAPACK_ILP64)
  find_library(ACCELERATE Accelerate REQUIRED)
  set(KF_BLAS_BACKEND "Accelerate")
  message(STATUS "BLAS backend: Accelerate (Apple)")

elseif(NOT KF_BLAS_VENDOR STREQUAL "OpenBLAS")
  # Linux/Windows: Try Intel MKL first (unless explicitly set to OpenBLAS)
  # MKL configuration:
  #   - MKL_THREADING=gnu_thread: Use GNU OpenMP (libgomp) instead of Intel OpenMP (libiomp5)
  #   - MKL_LINK=dynamic: Dynamic linking (shared libraries)
  #   - MKL_INTERFACE=lp64/ilp64: 32-bit or 64-bit integers
  set(MKL_THREADING gnu_thread)  # Use libgomp (GCC/Clang OpenMP), not libiomp5
  set(MKL_LINK dynamic)
  if(KF_BLAS_ILP64)
    set(MKL_INTERFACE ilp64)
  else()
    set(MKL_INTERFACE lp64)
  endif()

  # Add standard MKL install location to CMake prefix path
  list(PREPEND CMAKE_PREFIX_PATH /opt/intel/oneapi/mkl/latest)
  
  # Try to find MKL (QUIET: don't error if not found)
  find_package(MKL QUIET)

  if(MKL_FOUND)
    # MKL found: Set compile definitions
    add_compile_definitions(KF_USE_MKL)
    # MKL_ILP64 define is set in blas_config.h when KF_BLAS_ILP64 is defined
    set(KF_BLAS_BACKEND "MKL")
    message(STATUS "BLAS backend: Intel MKL (${MKL_INTERFACE}, ${MKL_THREADING})")
  elseif(KF_BLAS_VENDOR STREQUAL "MKL")
    # MKL explicitly requested but not found: error
    message(FATAL_ERROR "Intel MKL explicitly requested (KF_BLAS_VENDOR=MKL) but not found. "
                        "Install MKL or set KF_BLAS_VENDOR=OpenBLAS or KF_BLAS_VENDOR=AUTO.")
  endif()
endif()

# Fallback: Use generic BLAS (OpenBLAS, ATLAS, reference BLAS, etc.)
if(NOT DEFINED KF_BLAS_BACKEND)
  if(KF_BLAS_ILP64)
    # OpenBLAS ILP64: requires BLA_SIZEOF_INTEGER=8
    set(BLA_SIZEOF_INTEGER 8)
    # OPENBLAS_USE64BITINT define is set in blas_config.h
  endif()
  
  find_package(BLAS REQUIRED)
  set(KF_BLAS_BACKEND "OpenBLAS")
  message(STATUS "BLAS backend: OpenBLAS/generic BLAS")
endif()

# Common interface for headers from Python/pybind11
add_library(kf_common INTERFACE)
target_link_libraries(kf_common INTERFACE pybind11::headers Python::Module)

# Common interface for BLAS backend (provides include directories)
add_library(kf_blas INTERFACE)
if(KF_BLAS_BACKEND STREQUAL "Accelerate")
  # Accelerate doesn't need explicit include directories
  target_link_libraries(kf_blas INTERFACE ${ACCELERATE})
elseif(KF_BLAS_BACKEND STREQUAL "MKL")
  # MKL::MKL provides both include directories and libraries
  target_link_libraries(kf_blas INTERFACE MKL::MKL)
else()
  # BLAS::BLAS provides libraries (include paths may vary by system)
  target_link_libraries(kf_blas INTERFACE BLAS::BLAS)
endif()

# ---- Small helpers to avoid repetition --------------------------------------
# Track created modules/objlibs so we can link things in one go later
set(_KF_ALL_MODULES "")
set(_KF_ALL_OBJLIBS "")

# Create a C++ object library + pybind11 module pair:
#   kf_add_cpp_module(<base> <obj_src> <binding_src>)
#   -> object lib: kf_<base>
#   -> module target: <base> (no underscore prefix)
function(kf_add_cpp_module base obj_src bind_src)
  set(obj kf_${base})
  add_library(${obj} OBJECT ${obj_src})
  target_link_libraries(${obj} PRIVATE kf_common kf_blas OpenMP::OpenMP_CXX)

  pybind11_add_module(${base} MODULE
    ${bind_src}
    $<TARGET_OBJECTS:${obj}>
  )
  set_target_properties(${base} PROPERTIES OUTPUT_NAME "${base}")
  target_link_libraries(${base} PRIVATE kf_blas OpenMP::OpenMP_CXX)

  list(APPEND _KF_ALL_MODULES ${base})
  list(APPEND _KF_ALL_OBJLIBS ${obj})
  set(_KF_ALL_MODULES "${_KF_ALL_MODULES}" PARENT_SCOPE)
  set(_KF_ALL_OBJLIBS "${_KF_ALL_OBJLIBS}" PARENT_SCOPE)
endfunction()

# Portable optimization; native tuning is opt-in
option(KF_USE_NATIVE "Enable -march/-mcpu=native style flags" OFF)

function(kf_apply_cxx_flags tgt)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE
      -O3 -ffast-math -ftree-vectorize -fopenmp
      $<$<BOOL:${KF_USE_NATIVE}>:-mcpu=native -mtune=native>
    )
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(${tgt} PRIVATE
      -O3 -ffast-math
      $<$<BOOL:${KF_USE_NATIVE}>:-xHost -mtune=native>
    )
  endif()
endfunction()

# ---- C++ modules -------------------------------------------------------------
kf_add_cpp_module(global_kernels  src/global_kernels.cpp  src/global_kernels_bindings.cpp)
# FCHL19 is split into two modules: repr and local_kernels
kf_add_cpp_module(local_kernels   src/local_kernels.cpp   src/local_kernels_bindings.cpp)
kf_add_cpp_module(fchl19_repr     src/fchl19_repr.cpp     src/fchl19_repr_bindings.cpp)
kf_add_cpp_module(invdist_repr    src/invdist_repr.cpp    src/invdist_repr_bindings.cpp)
kf_add_cpp_module(kernelmath      src/math.cpp            src/math_bindings.cpp)

# Kitchen sinks (Random Fourier Features) — split across two obj/binding sources;
# handled manually so we can pass multiple files without extending the macro.
add_library(kf_kitchen_sinks OBJECT
  src/rff_features.cpp
  src/rff_elemental.cpp)
target_link_libraries(kf_kitchen_sinks PRIVATE kf_common kf_blas OpenMP::OpenMP_CXX)
kf_apply_cxx_flags(kf_kitchen_sinks)

pybind11_add_module(kitchen_sinks MODULE
  src/rff_features_bindings.cpp
  src/rff_elemental_bindings.cpp
  $<TARGET_OBJECTS:kf_kitchen_sinks>)
set_target_properties(kitchen_sinks PROPERTIES OUTPUT_NAME "kitchen_sinks")
target_link_libraries(kitchen_sinks PRIVATE kf_blas)

# Apply C++ flags to the object libs (not to the module targets)
foreach(obj ${_KF_ALL_OBJLIBS})
  kf_apply_cxx_flags(${obj})
endforeach()

# ---- OpenMP (C++) ------------------------------------------------------------
if (OpenMP_CXX_FOUND)
  target_link_libraries(kernelmath PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(global_kernels PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(local_kernels PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(fchl19_repr PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(invdist_repr PRIVATE OpenMP::OpenMP_CXX)
  # kitchen_sinks already links OpenMP via kf_kitchen_sinks object lib above
  target_link_libraries(kitchen_sinks PRIVATE OpenMP::OpenMP_CXX)
endif()

# Note: BLAS backend linking is handled via kf_blas interface library
# (linked in kf_add_cpp_module and kitchen_sinks setup)

# ---- Install ----------------------------------------------------------------
install(TARGETS global_kernels local_kernels fchl19_repr invdist_repr kernelmath kitchen_sinks
  LIBRARY DESTINATION kernelforge   # Linux/macOS
  RUNTIME DESTINATION kernelforge   # Windows (.pyd)
)
install(FILES python/kernelforge/__init__.py DESTINATION kernelforge)

# ---- Development targets -----------------------------------------------------
# Collect all C++ source files
file(GLOB_RECURSE KF_CXX_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

# clang-format target
find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${KF_CXX_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-format on all C++ source files"
    VERBATIM
  )
else()
  add_custom_target(format
    COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found. Please install it to use this target."
    COMMENT "clang-format not available"
  )
endif()

# clang-tidy target
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY_EXE} ${KF_CXX_SOURCES} --
      -std=c++17
      -I${CMAKE_CURRENT_SOURCE_DIR}/src
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-tidy on all C++ source files"
    VERBATIM
  )
else()
  add_custom_target(tidy
    COMMAND ${CMAKE_COMMAND} -E echo "clang-tidy not found. Please install it to use this target."
    COMMENT "clang-tidy not available"
  )
endif()
