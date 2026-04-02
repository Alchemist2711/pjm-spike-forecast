.PHONY: install test lint download run clean

install:
	pip install -e ".[dev]"

download:
	python -m pjm_spike_forecast.pipeline --download-only

run:
	python -m pjm_spike_forecast.pipeline --data-dir data/raw --output-dir results

demo:
	python -m pjm_spike_forecast.pipeline --demo --output-dir results

test:
	pytest tests/ -v --cov=pjm_spike_forecast --cov-report=term-missing --cov-report=html

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info htmlcov/ .pytest_cache/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
