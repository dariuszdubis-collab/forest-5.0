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
```

CLI automatycznie wykrywa separator CSV przy użyciu `csv.Sniffer` i szybkiego
parsera C. W razie potrzeby separator można podać ręcznie opcją `--sep`,
np. `--sep ';'`.

## Live trading z MetaTrader 4

1. Zainstaluj i uruchom MT4 (np. pod Wine w katalogu `~/.mt4`).
2. Wgraj EA `mt4/ForestBridge.mq4` do `MQL4/Experts` i skompiluj go w MetaEditorze.
3. Most komunikacyjny tworzy katalog `MQL4/Files/forest_bridge` – podaj tę ścieżkę w `broker.bridge_dir` lub poprzez zmienną `FOREST_MT4_BRIDGE_DIR`.
4. Sklonuj przykład konfiguracji i dostosuj go do swoich potrzeb:
   ```bash
   cp config/live.example.yaml config/live.yaml
   ```
   W `config/live.yaml` ustaw symbol, wolumen i ścieżkę mostu.
5. Uruchom tryb live:
   ```bash
   poetry run forest5 live --config config/live.yaml
   ```
   W celu testów bez realnych zleceń dopisz `--paper`.
