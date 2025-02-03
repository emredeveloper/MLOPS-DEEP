.PHONY: install train serve test clean

install:
	pip install -r requirements.txt
	pre-commit install

train:
	python main.py

serve:
	docker-compose up -d

test:
	pytest
	mypy deepseek_mlops
	flake8 deepseek_mlops

coverage:
	coverage run -m pytest
	coverage report

stop:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .coverage
	rm -rf mlruns/*
	rm -rf logs/*
	rm -rf models/*
