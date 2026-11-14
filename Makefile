.DEFAULT_GOAL := help
PORT ?= 8000

help:
	@echo "make install | data | train | train_news | run_api | run_frontend | test"

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

data:
	python -c "from nowcast.ml_logic.data import download_fred; download_fred()"

train:
	python -m nowcast.interface.main

train_news:
	python -c "from nowcast.interface.main import train; train(use_news=True)"

run_api:
	uvicorn nowcast.api.fast:app --reload --port $(PORT)

run_frontend:
	API_URL=http://localhost:$(PORT) streamlit run app/streamlit_app.py

test:
	pytest -q
