"""Module entry point for `uv run python -m src`.

Python executes this file when the package is run as a module. The only job
here is to delegate command-line parsing to Fire and expose the RAGSystem
methods as CLI commands.
"""

import fire

from src.cli import RAGSystem, _die


def main() -> None:
    """Start the Fire CLI for the RAG system."""
    try:
        fire.Fire(RAGSystem)
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            RuntimeError, EOFError) as exc:
        _die(str(exc) or type(exc).__name__)


if __name__ == "__main__":
    main()
