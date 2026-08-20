.PHONY: install test lint format typecheck run-baseline run-multi run-benchmark clean

install:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest tests/ -v

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	PYTHONPATH=src mypy src

run-baseline:
	PYTHONPATH=src python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art"

run-multi:
	PYTHONPATH=src python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG state-of-the-art"

run-benchmark:
	PYTHONPATH=src python -m multi_agent_research_lab.cli benchmark --query "Research GraphRAG state-of-the-art"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
