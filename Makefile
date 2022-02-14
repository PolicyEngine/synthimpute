all: build
format:
	autopep8 -r .
	black . -l 79
	linecheck . --fix
install:
	pip install -e .[dev]
	pip install --upgrade jupyter-book
test:
	pytest
documentation:
	jb build docs/book
build:
	python setup.py sdist bdist_wheel
