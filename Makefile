all: environment

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
