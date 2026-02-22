all: environment

install-linux:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON" uv pip install -e .[test] --verbose

install-linux-ilp64:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_ILP64=ON" uv pip install -e .[test] --verbose

install-linux-mkl:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL" uv pip install -e .[test] --verbose

install-linux-mkl-ilp64:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL -DKF_BLAS_ILP64=ON" uv pip install -e .[test] --verbose

install-linux-openblas:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=OpenBLAS" uv pip install -e .[test] --verbose

install-linux-openblas-ilp64:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=OpenBLAS -DKF_BLAS_ILP64=ON" uv pip install -e .[test] --verbose

install-macos:
	CMAKE_ARGS="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DKF_USE_NATIVE=ON " uv pip install -e .[test] --verbose

install-macos-ilp64:
	CMAKE_ARGS="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DKF_USE_NATIVE=ON -DKF_BLAS_ILP64=ON" uv pip install -e .[test] --verbose

test:
	uv run pytest

environment:
	uv venv --python 3.14
	uv pip install scikit-build-core pybind11

format:
	uv run ruff format python/ tests/
	uv run ruff check --select I --fix python/ tests/
