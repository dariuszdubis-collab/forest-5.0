# Forest 5.0

Modułowy framework tradingowy łączący **analizę techniczną**, **agent AI (sentyment/fundamenty)**,
**broker MT5 (stub)** i **paper trading**. Działa lokalnie (CLI). Opcjonalnie lekki UI (Streamlit).

## Szybki start

```bash
# 1) Klon i środowisko (wybierz conda, poetry lub pip)
conda env create -f environment.yml && conda activate forest5
# albo
poetry install
# albo
pip install -e . && pip install -r requirements-dev.txt

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

## Tryb PAPER (smoke test bez MT4)

Do szybkiego testu bez uruchamiania MetaTradera można zasymulować most
plikowy. Utwórz katalog `bridge/` z podkatalogami `ticks`, `state`,
`commands` i `results` oraz minimalnymi plikami JSON:

```bash
mkdir -p bridge/{ticks,state,commands,results}
echo '{"time": 1, "bid": 100}' > bridge/ticks/tick.json
echo '{"equity": 0}' > bridge/state/account.json
echo '{"qty": 0}' > bridge/state/position_EURUSD.json
echo '{}' > bridge/commands/noop.json
echo '{}' > bridge/results/noop.json
```

Następnie w Pythonie uruchom `run_live` z brokerem typu `paper`:

```python
from forest5.config_live import (
    LiveSettings, BrokerSettings, DecisionSettings,
    LiveTimeSettings, LiveTimeModelSettings,
)
from forest5.live.live_runner import run_live

settings = LiveSettings(
    broker=BrokerSettings(type="paper", bridge_dir="bridge", symbol="EURUSD"),
    decision=DecisionSettings(min_confluence=2),
    time=LiveTimeSettings(
        model=LiveTimeModelSettings(enabled=True, path="models/model_time.json"),
    ),
)
run_live(settings, max_steps=10)
```

## Zarządzanie ryzykiem

### `risk.on_drawdown.action`

Parametr `risk.on_drawdown.action` określa reakcję systemu po osiągnięciu
progu `risk.max_drawdown`.

- `"halt"` – natychmiast wstrzymuje handel po przekroczeniu maksymalnego
  obsunięcia kapitału (wymagany ręczny restart).
- `"soft_wait"` – pauzuje otwieranie nowych pozycji do czasu, aż kapitał
  powróci powyżej zadanego progu.

Przykład konfiguracji:

```yaml
risk:
  risk_per_trade: 0.005
  max_drawdown: 0.20
  on_drawdown:
    action: "halt"  # lub "soft_wait"
```

## Backtest + TimeOnly

Sekcja `time` pozwala łączyć sygnały strategii z modelem czasu oraz blokować wybrane przedziały:

- `time.use_time_model` – włącza model czasu podczas backtestu (`true`/`false`).
- `time.time_model_path` – ścieżka do pliku z zapisanym modelem czasu.
- `time.fusion_min_confluence` – minimalna konfluencja (0–1) wymagana do fuzji sygnału strategii z modelem.
- `time.blocked_hours` – lista godzin (0–23), w których handel jest zablokowany.
- `time.blocked_weekdays` – lista dni tygodnia (0=pon … 6=niedz), w których handel jest wyłączony.

W trybie live pamiętaj o ustawieniu `decision.min_confluence` (np. `2`) i
włączeniu modelu czasu:

```yaml
time:
  model:
    enabled: true
    path: models/model_time.json
```

Podczas backtestów podobną funkcję pełnią opcje:

```python
settings.time.use_time_model = True
settings.time.time_model_path = "models/model_time.json"
```

## Trening modelu czasu

Aby wytrenować prosty model czasu na danych z pliku CSV zawierającego kolumny
`time` i `y` użyj skryptu `scripts/time_train.py`:

```bash
python scripts/time_train.py --input demo.csv --output models/model_time.json
```

Opcje `--q-low` i `--q-high` pozwalają dostroić kwantyle decyzyjne modelu.
