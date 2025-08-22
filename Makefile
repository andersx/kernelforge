install:
	pip install -e .[test] --verbose

test:
	pytest
