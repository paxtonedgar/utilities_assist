.PHONY: install test run clean

install:
	pip install -r requirements.txt

test:
	python -m pytest test/

run:
	python src/data_pipeline/main.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete