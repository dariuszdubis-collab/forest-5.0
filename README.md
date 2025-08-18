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

## Live trading z MT4

1. Uruchom MT4 pod Wine (domyślnie `~/.mt4`). Hash katalogu danych znajdziesz w `Terminal/<hash>`.
2. Wgraj EA: `mt4/ForestBridge.mq4` do `MQL4/Experts` i skompiluj w MetaEditorze (Navigator -> Refresh).
3. Struktura mostu: `MQL4/Files/forest_bridge/{ticks,commands,results,state}` tworzy się automatycznie.
4. Deploy EA: `export WINEPREFIX=~/.mt4 MT4_HASH=<hash> && bash scripts/mt4_deploy.sh`.
5. Ustaw `FOREST_MT4_BRIDGE_DIR` i odpal smoke-test:
   ```bash
   export FOREST_MT4_BRIDGE_DIR="/home/<user>/.mt4/drive_c/users/<user>/AppData/Roaming/MetaQuotes/Terminal/<hash>/MQL4/Files/forest_bridge"
   python scripts/mt4_smoke_test.py
   ```
6. Start trybu live: `forest5 live --config config/live.example.yaml`.
