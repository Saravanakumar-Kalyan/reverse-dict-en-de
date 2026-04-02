.PHONY: help install train inference clean test lint format

help:
	@echo "Cross-Lingual Reverse Dictionary - Available Commands"
	@echo "===================================================="
	@echo ""
	@echo "install       - Install project dependencies"
	@echo "train         - Train the model"
	@echo "inference     - Run inference with pre-trained model"
	@echo "test          - Run unit tests"
	@echo "lint          - Run code linting"
	@echo "format        - Format code with black"
	@echo "clean         - Clean up generated files"
	@echo "docs          - Generate documentation"
	@echo ""

install:
	pip install -r requirements.txt

train:
	python main.py --train

inference:
	python main.py --inference

test:
	python -m pytest tests/ -v

lint:
	python -m pylint *.py

format:
	python -m black *.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f .coverage
	rm -rf htmlcov/
	rm -rf dist/ build/ *.egg-info/

docs:
	@echo "Documentation available in README.md"

.DEFAULT_GOAL := help
