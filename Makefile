.PHONY: install install-dev lint format test

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/
