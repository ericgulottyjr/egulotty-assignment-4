# Makefile

VENV := venv
PYTHON := python3

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

run:
	$(VENV)/bin/python app.py
	