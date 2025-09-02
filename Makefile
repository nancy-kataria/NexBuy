# Makefile for NexBuy project

.PHONY: help install install-dev test lint format clean run setup

# Default target
help:
	@echo "Available targets:"
	@echo "  setup        - Set up the project for development"
	@echo "  install      - Install project dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code with black"
	@echo "  typecheck    - Run type checking with mypy"
	@echo "  clean        - Clean up build artifacts"
	@echo "  run          - Run the Streamlit app"
	@echo "  build        - Build the package"

# Set up the project for development
setup: clean install-dev
	@echo "Project setup complete!"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src/nexbuy --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__

# Format code
format:
	black src/ tests/ --line-length=100

# Type checking
typecheck:
	mypy src/nexbuy/ --ignore-missing-imports

# Check all code quality
check: lint typecheck test

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run the Streamlit app
run:
	streamlit run app.py

# Build the package
build: clean
	python setup.py sdist bdist_wheel

# Install package in editable mode
develop:
	pip install -e .

# Create virtual environment
venv:
	python -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Full development setup from scratch
dev-setup: venv
	@echo "Activate virtual environment and run 'make setup'"
