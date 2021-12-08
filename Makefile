format:
	black . -l 79
install:
	pip install -e .[dev]
test:
	pytest
build-package:
	rm -rf dist build
	python setup.py sdist bdist_wheel
