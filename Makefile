install-linux:
	@echo "Auto-detecting BLAS: tries MKL first, falls back to OpenBLAS"
	CMAKE_ARGS="-DKF_USE_NATIVE=ON" uv pip install -e .[test] --verbose

install-linux-ilp64:
	@echo "Auto-detecting BLAS (ILP64 mode): tries MKL first, falls back to OpenBLAS"
	@echo "Note: Requires MKL or libopenblas64-dev (apt install libopenblas64-dev)"
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_ILP64=ON" uv pip install -e .[test] --verbose

install-linux-mkl:
	@echo "Explicit Intel MKL backend (LP64, 32-bit integers)"
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL" uv pip install -e .[test] --verbose

install-linux-mkl-ilp64:
	@echo "Explicit Intel MKL backend (ILP64, 64-bit integers)"
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=MKL -DKF_BLAS_ILP64=ON" uv pip install -e .[test] --verbose

install-linux-openblas:
	@echo "Explicit OpenBLAS backend (LP64, 32-bit integers)"
	CMAKE_ARGS="-DKF_USE_NATIVE=ON -DKF_BLAS_VENDOR=OpenBLAS" uv pip install -e .[test] --verbose

install-linux-openblas-ilp64:
	@echo "Explicit OpenBLAS backend (ILP64, 64-bit integers)"
	@echo "Note: Requires libopenblas64-dev (apt install libopenblas64-dev)"
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

typecheck:
	uv run mypy python/ tests/
