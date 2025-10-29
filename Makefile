install:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON" uv pip install -e .[test] --verbose

test:
	pytest

environment:
	uv venv --python 3.10
# 	conda env create -f environments/environment-dev.yaml
