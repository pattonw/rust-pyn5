.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
DATA_DIR = tests/data
DIST_DIR = target/wheels

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr target/
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.so' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-data:
	rm -rf $(DATA_DIR)

lint: ## check style with flake8
	flake8 pyn5 tests

test: ## run tests quickly with the default Python
	maturin develop && pytest -v

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source pyn5 setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pyn5.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pyn5
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	maturin publish -i python3.8 -i python3.7 -i python3.6

dist: clean ## builds source and wheel package
	maturin build -i python3.8 -i python3.7 -i python3.6 --release && \
	ls -l $(DIST_DIR)

install-dev: clean
	pip install -r requirements.txt && maturin develop

install: clean ## install the package to the active Python's site-packages
	pip install -r requirements.txt
	pip install .  # fails with BackendUnavailable error
	# maturin build --release --no-sdist -i python && pip install $(DIST_DIR)/pyn5-*.whl

$(DATA_DIR)/JeffT1_le.tif:
	mkdir -p $(DATA_DIR) && \
	wget -P $(DATA_DIR) https://imagej.nih.gov/ij/images/t1-head-raw.zip && \
	unzip $(DATA_DIR)/t1-head-raw.zip -d $(DATA_DIR)

data: $(DATA_DIR)/JeffT1_le.tif

fmt_py:
	black tests pyn5 --check

fmt_rust:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --workspace -- -Dwarnings