install:
	CMAKE_ARGS="-DKF_USE_NATIVE=ON" pip install -e .[test] --verbose

test:
	pytest

environment:
	conda env create -f environments/environment-dev.yaml
