.PHONY: install run debug lint lint-strict test clean

install:
	uv sync
	uv sync --extra dev

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 src/ test.py
	uv run mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/ test.py
	uv run mypy src/ --strict

test:
	uv run pytest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .mypy_cache .pytest_cache
