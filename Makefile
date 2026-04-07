PYTHON=python3
PIP=pip3

.PHONY: all build install clean

all:
	$(MAKE) build

build:
	$(PIP) install build && $(PYTHON) -m build --wheel

init:
	pip install -r requirements.txt

install:
	pip install .

clean:
	for i in `find . -name __pycache__`; do rm -rf $$i; done
	for i in `find . -name '*.egg-info'`; do rm -rf $$i; done
	-rm -rf build dist
