all: environment

NPROC := $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf 1)
CCACHE_BIN := $(shell command -v ccache 2>/dev/null)
NINJA_BIN := $(shell command -v ninja 2>/dev/null)
TORCH_CUDA_ARCH_LIST ?= $(shell uv run python -c "import torch; major, minor = torch.cuda.get_device_capability(0); print(f'{major}.{minor}')" 2>/dev/null)

CUDA_LOCAL_FAST_CMAKE_ARGS := -DKF_WITH_CUDA=ON -DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL -DKF_BLAS_ILP64=ON -DCMAKE_CUDA_ARCHITECTURES=native

ifneq ($(CCACHE_BIN),)
CUDA_LOCAL_FAST_CMAKE_ARGS += -DCMAKE_CXX_COMPILER_LAUNCHER=$(CCACHE_BIN) -DCMAKE_CUDA_COMPILER_LAUNCHER=$(CCACHE_BIN)
endif

ifneq ($(NINJA_BIN),)
CUDA_LOCAL_FAST_CMAKE_ARGS += -G Ninja
endif

install-linux:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON" uv pip install -e .[test,dev] --verbose

install-linux-ilp64:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_ILP64=ON" uv pip install -e .[test,dev] --verbose

install-linux-mkl:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL" uv pip install -e .[test,dev] --verbose

install-linux-mkl-ilp64:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL -DKF_BLAS_ILP64=ON" uv pip install -e .[test,dev] --verbose

install-linux-openblas:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=OpenBLAS" uv pip install -e .[test,dev] --verbose

install-linux-openblas-ilp64:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=OpenBLAS -DKF_BLAS_ILP64=ON" uv pip install -e .[test,dev] --verbose

install-linux-mkl-ilp64-cuda:
	TORCH_CUDA_ARCH_LIST="$${TORCH_CUDA_ARCH_LIST:-$(if $(TORCH_CUDA_ARCH_LIST),$(TORCH_CUDA_ARCH_LIST),12.0)}" CMAKE_BUILD_PARALLEL_LEVEL="$(NPROC)" CMAKE_ARGS="$(CUDA_LOCAL_FAST_CMAKE_ARGS)" uv pip install -e .[test,dev] --verbose

# Build CPU + companion CUDA wheels into dist/ (split PyPI layout).
wheel-cpu:
	CMAKE_ARGS="-DKF_WITH_CUDA=OFF" uv build --wheel -o dist

wheel-cuda:
	uv pip install 'setuptools-scm>=8' scikit-build-core pybind11 setuptools wheel
	cd packaging/kernelforge-cuda && \
	TORCH_CUDA_ARCH_LIST="$${TORCH_CUDA_ARCH_LIST:-$(if $(TORCH_CUDA_ARCH_LIST),$(TORCH_CUDA_ARCH_LIST),12.0)}" \
	CMAKE_BUILD_PARALLEL_LEVEL="$(NPROC)" \
	CMAKE_ARGS="$(CUDA_LOCAL_FAST_CMAKE_ARGS) -DKF_CUDA_ONLY=ON" \
	uv build --wheel --no-build-isolation -o ../../dist

# Fresh-venv smoke: CPU wheel + companion wheel → import cuda_* modules.
demo-cuda-wheels: wheel-cpu wheel-cuda
	rm -rf /tmp/kf-cuda-demo-venv
	uv venv /tmp/kf-cuda-demo-venv --python 3.14
	uv pip install --python /tmp/kf-cuda-demo-venv $$(ls -1 dist/kernelforge-*.whl | grep -v kernelforge_cuda | grep -v kernelforge-cuda | tail -1)
	uv pip install --python /tmp/kf-cuda-demo-venv --find-links dist 'kernelforge-cuda'
	/tmp/kf-cuda-demo-venv/bin/python packaging/kernelforge-cuda/smoke_imports.py

install-macos:
	CMAKE_ARGS="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DKF_USE_NATIVE=ON " uv pip install -e .[test,dev] --verbose

install-macos-ilp64:
	CMAKE_ARGS="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DKF_USE_NATIVE=ON -DKF_BLAS_ILP64=ON" uv pip install -e .[test,dev] --verbose

test:
	uv run pytest

environment:
	uv venv --python 3.14
	uv pip install scikit-build-core pybind11

check: format lint typecheck

lint: lint-python

lint-python:
	uv run ruff check python/ tests/

format: format-python format-cpp

format-python:
	uv run ruff format python/ tests/
	uv run ruff check --select I --fix python/ tests/

format-cpp:
	clang-format -i src/*.cpp src/*.hpp src/*.h

tidy:
	clang-tidy src/*.cpp src/*.hpp -- -std=c++17 -Isrc

typecheck:
	uv run ty check python/ tests/

clean:
	rm -rf ./.venv/
	rm -rf ./.ruff_cache/
	rm -rf ./.pytest_cache/
