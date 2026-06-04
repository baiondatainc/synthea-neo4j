.PHONY: help venv install install-dev run lint format test clean docker-build

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Makefile targets: venv install run lint format test clean docker-build"

venv:
	@if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi

install: venv
	$(PIP) install --upgrade pip setuptools wheel
	@if [ -f pyproject.toml ]; then \
		$(PIP) install -e . || $(PIP) install . ; \
	elif [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt ; \
	else \
		echo "No pyproject.toml or requirements.txt found; consider adding dependencies." ; \
	fi

install-dev: install
	$(PIP) install --upgrade black flake8 pytest || true

run: venv
	$(PY) main.py

lint: venv
	$(PIP) install --upgrade flake8 >/dev/null 2>&1 || true
	$(VENV)/bin/flake8 .

format: venv
	$(PIP) install --upgrade black >/dev/null 2>&1 || true
	$(VENV)/bin/black .

test: venv
	$(PIP) install --upgrade pytest >/dev/null 2>&1 || true
	$(VENV)/bin/pytest -q

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache

docker-build:
	docker build -t synthea-neo4j .
