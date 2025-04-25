.PHONY: all setup
# make >debug.log 2>&1
# git remote prune origin
ifeq ($(OS),Windows_NT)
PYTHON = venv/Scripts/python.exe
PTEST = venv/Scripts/pytest.exe
COVERAGE = venv/Scripts/coverage.exe
else
PYTHON = ./venv/bin/python
PTEST = ./venv/bin/pytest
COVERAGE = ./venv/bin/coverage
endif

SOURCE = whisper_rttm
TESTS = tests

FLAKE8 = $(PYTHON) -m flake8
PYLINT = $(PYTHON) -m pylint
PYTEST = $(PTEST) --cov=$(SOURCE) --cov-report term:skip-covered
PIP = $(PYTHON) -m pip install

MP3 = \
xxx.mp3 \

SRT = $(addprefix build/,$(subst .mp3,.srt,$(MP3)))

all: tests

build/%.srt: build/%.mp3
	$(PYTHON) $(SOURCE)/to_srt.py --whisper_batch 8 --torch_batch 4 $< $(basename $<).rttm $@

srt: $(SRT)

tests: flake8 pep257 lint
	$(PYTEST) -m "not longrunning" --durations=5 $(TESTS)

cover: flake8 pep257 lint
	$(PYTEST) --durations=5 $(TESTS)
	$(COVERAGE) html --skip-covered

test:
	$(PTEST) -s $(TESTS)/test/$(T)

pep257:
	$(PYTHON) -m pydocstyle $(TESTS)/test
	$(PYTHON) -m pydocstyle $(SOURCE)

flake8:
	$(FLAKE8) $(TESTS)/test
	$(FLAKE8) $(SOURCE)

lint:
	$(PYLINT) $(TESTS)/test
	$(PYLINT) $(SOURCE)

setup: setup_python setup_pip

setup_pip:
	$(PIP) --upgrade pip
	$(PIP) -r requirements.txt
	$(PIP) -r dev.txt

setup_python:
	$(PYTHON_BIN) -m venv ./venv
