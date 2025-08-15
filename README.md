# Forest 5.0

Modułowy framework tradingowy łączący **analizę techniczną**, **agent AI (sentyment/fundamenty)**,  
**broker MT5 (stub)** i **paper trading**. Działa lokalnie (CLI). Opcjonalnie lekki UI (Streamlit).

## Szybki start

```bash
# 1) Klon i środowisko (wybierz conda albo poetry)
conda env create -f environment.yml && conda activate forest5
# albo
poetry install

# 2) Testy i lint
make lint && make test

# 3) Demo – generator syntetycznych danych
poetry run forest5-demo --periods 50 --out demo.csv

# 4) Backtest na CSV
poetry run forest5 backtest --data demo.csv --fast 12 --slow 26

# 5) Grid-search
poetry run forest5 grid --data demo.csv --fast 6 12 6 --slow 20 40 10

