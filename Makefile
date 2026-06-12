.PHONY: install run debug lint lint-strict clean

install:
	uv sync
	uv sync --extra dev

run:
	uv run python -m src

debug:
	uv run python -m pdb -m student

lint:
	uv run flake8 src/
	uv run mypy src/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .mypy_cache .pytest_cache
