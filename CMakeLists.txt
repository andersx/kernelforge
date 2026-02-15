cmake_minimum_required(VERSION 3.18)
project(kernelforge LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Use Accelerate on Apple, otherwise BLAS (MKL, OpenBLAS, etc)
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
else()
  find_package(BLAS REQUIRED)
endif()

# Common interface for headers from Python/pybind11
add_library(kf_common INTERFACE)
target_link_libraries(kf_common INTERFACE pybind11::headers Python::Module)

# ---- Small helpers to avoid repetition --------------------------------------
# Track created modules/objlibs so we can link things in one go later
set(_KF_ALL_MODULES "")
set(_KF_ALL_OBJLIBS "")

# Create a C++ object library + pybind11 module pair:
#   kf_add_cpp_module(<base> <obj_src> <binding_src>)
#   -> object lib: kf_<base>
#   -> module target: _<base>
function(kf_add_cpp_module base obj_src bind_src)
  set(obj kf_${base})
  add_library(${obj} OBJECT ${obj_src})
  target_link_libraries(${obj} PRIVATE kf_common)
  target_link_libraries(${obj} PRIVATE OpenMP::OpenMP_CXX) # <-- compile flags propagate

  pybind11_add_module(_${base} MODULE
    ${bind_src}
    $<TARGET_OBJECTS:${obj}>
  )
  set_target_properties(_${base} PROPERTIES OUTPUT_NAME "_${base}")
  target_link_libraries(_${base} PRIVATE OpenMP::OpenMP_CXX) # <-- link flags

  list(APPEND _KF_ALL_MODULES _${base})
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
kf_add_cpp_module(kernels  src/kernels.cpp                src/bindings_kernels.cpp)
kf_add_cpp_module(invdist  src/invdist.cpp                src/invdist_bindings.cpp)
kf_add_cpp_module(fchl19   src/fchl19_representation.cpp  src/bindings_fchl19.cpp)
kf_add_cpp_module(cholesky src/cholesky.cpp               src/bindings_cholesky.cpp)

# Apply C++ flags to the object libs (not to the module targets)
foreach(obj ${_KF_ALL_OBJLIBS})
  kf_apply_cxx_flags(${obj})
endforeach()

# ---- OpenMP (C++) ------------------------------------------------------------
if (OpenMP_CXX_FOUND)
  target_link_libraries(_cholesky PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(_fchl19 PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(_invdist PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- BLAS/LAPACK backend selection (link all modules) -----------------------
foreach(m ${_KF_ALL_MODULES})
  if(APPLE)
    target_link_libraries(${m} PRIVATE ${ACCELERATE})
  elseif(WIN32)
    target_link_libraries(${m} PRIVATE MKL::MKL)
  else()
    target_link_libraries(${m} PRIVATE BLAS::BLAS)
  endif()
endforeach()

# ---- Install ----------------------------------------------------------------
install(TARGETS _kernels _invdist _fchl19 _cholesky
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
