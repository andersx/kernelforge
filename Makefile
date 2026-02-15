install-linux:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON" uv pip install -e .[test] --verbose

install-macos:
	CMAKE_ARGS="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DKF_USE_NATIVE=ON " uv pip install -e .[test] --verbose

test:
	pytest

environment:
	uv venv --python 3.14
	uv pip install scikit-build-core pybind11

# Code quality targets
.PHONY: format lint typecheck check

format:
	black python/ tests/ benchmark/
	isort python/ tests/ benchmark/

lint:
	ruff check python/ tests/ benchmark/

lint-fix:
	ruff check --fix python/ tests/ benchmark/

typecheck:
	mypy python/ tests/ benchmark/

check: format lint typecheck
	@echo "All code quality checks passed!"
