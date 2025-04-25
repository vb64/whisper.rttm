.PHONY: all setup
# make >debug.log 2>&1
# git remote prune origin
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
else
PYTHON = ./venv/bin/python
endif

SOURCE = whisper_rttm

FLAKE8 = $(PYTHON) -m flake8
PYLINT = $(PYTHON) -m pylint
PIP = $(PYTHON) -m pip install


all:
	$(PYTHON) -m pydocstyle $(SOURCE)
	$(FLAKE8) $(SOURCE)
	$(PYLINT) $(SOURCE)

srt:
	$(PYTHON) $(SOURCE)/to_srt.py --whisper_batch 8 --torch_batch 4 fixtures/short.mp3 fixtures/short.rttm build/short.srt

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) -r requirements.txt
	$(PIP) -r dev.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
