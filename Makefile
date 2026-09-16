PYTHON ?= python
CONFIG ?= configs/research.yaml

.PHONY: install check validation final

install:
	$(PYTHON) -m pip install -e ".[dev]"

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check cross_market tests data/download_binance.py
	$(PYTHON) -m mypy cross_market
	$(PYTHON) -m pytest

validation:
	$(PYTHON) -m cross_market.run --config $(CONFIG) --stage validation

final:
	$(PYTHON) -m cross_market.run --config $(CONFIG) --stage final
