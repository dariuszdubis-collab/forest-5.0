# Makefile (poetry-optional)

# If Poetry is present, run commands through "poetry run".
# Otherwise, run tools directly from the active environment (conda/venv).
POETRY := $(shell command -v poetry >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(POETRY),yes)
  RUN = poetry run
else
  RUN =
endif

.PHONY: lint test dev

# Szybka instalacja narzędzi developerskich do bieżącego env (ruff, pytest)
install-dev:
	python -m pip install -U ruff pytest

# Lint (Ruff)
lint:
	python -m ruff check .

# Autoformat (Ruff formatter)
format:
	$(RUN) ruff format .

# Testy (pytest)
test:
	python -m pytest -q

# „CI” lokalnie (lint + test)
ci: lint test

# Sprzątanie cache'y
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +; \
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build

dev:
	python -m pip install -e . && python -m pip install -U ruff pytest
