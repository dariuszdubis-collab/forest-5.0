# Forest 5.0

Modułowy framework tradingowy łączący **analizę techniczną**, **agent AI (sentyment/fundamenty)**,
**broker MT5 (stub)** i **paper trading**. Działa lokalnie (CLI). Opcjonalnie lekki UI (Streamlit).

## Wymagania środowiskowe

- Python ≥3.10,<3.13
- pakiety: pandas, numpy, scipy, joblib, pydantic, pyyaml, structlog, plotly, tzdata, setuptools
- pełna lista znajduje się w plikach `environment.yml` (Conda) oraz `pyproject.toml` (Poetry)

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
poetry run forest5 backtest --csv demo.csv --symbol EURUSD --fast 12 --slow 26

# 4b) Backtest ze strategią H1 EMA/RSI/ATR
poetry run forest5 backtest --csv demo.csv --symbol EURUSD --strategy h1_ema_rsi_atr \
  --ema-fast 34 --ema-slow 89 --rsi-len 14 --atr-len 14 \
  --t-sep-atr 0.20 --pullback-atr 0.50 --entry-buffer-atr 0.10 \
  --sl-min-atr 0.90 --rr 1.8

# 5) Grid-search
poetry run forest5 grid --csv demo.csv --symbol EURUSD --fast-values 6:12:6 --slow-values 20:40:10
```

## Komendy GRID i walk-forward

Pełne uruchomienie GRID:

```bash
python scripts/optimize_grid.py \
  --csv data/EURUSD_H1_ML_ready.csv --symbol EURUSD \
  --fast "5-20" --slow "20-60:5" \
  --use-rsi --rsi-period 14 --rsi-oversold 30 --rsi-overbought 70 \
  --capital 100000 --risk 0.01 --fee-perc 0.0005 --slippage-perc 0.0 \
  --atr-period 14 --atr-multiple 2.0 \
  --start 2009-07-01 --end 2011-12-31 \
  --dd-penalty 0.5 --jobs 1 --quiet --skip-fast-ge-slow \
  --export out/grid_ema_rsi.csv
```

Walk-forward:

```bash
PYTHONWARNINGS=ignore python scripts/walkforward.py \
  --csv data/EURUSD_H1_ML_ready.csv --symbol EURUSD \
  --fast "5-20" --slow "20-60:5" \
  --use-rsi --rsi-period 14 --rsi-oversold 30 --rsi-overbought 70 \
  --capital 100000 --risk 0.01 --fee-perc 0.0005 --slippage-perc 0.0 \
  --atr-period 14 --atr-multiple 2.0 \
  --start 2009-07-01 --end 2011-12-31 \
  --train-months 12 --test-months 3 --step-months 3 \
  --dd-penalty 0.5 --skip-fast-ge-slow \
  --export out/walkforward.csv
```

### Kolumny `results.csv`

Plik z wynikami gridu zawiera:

- `fast`, `slow` – parametry strategii,
- `equity_end` – końcowy kapitał,
- `dd` – maksymalne obsunięcie,
- `cagr` – roczna stopa zwrotu (CAGR),
- `rar` – `cagr` podzielone przez `dd`,
- `trades` – liczba transakcji,
- `winrate` – odsetek zyskownych transakcji,
- `pnl`, `pnl_net` – zysk brutto i netto,
- `sharpe` – współczynnik Sharpe'a,
- `expectancy` – średni zysk na transakcję,
- `expectancy_by_pattern` – zysk w podziale na patterny,
- `timeonly_wait_pct` – czas oczekiwania modelu TimeOnly,
- `setups_expired_pct` – odsetek wygasłych setupów,
- `rr_avg`, `rr_median` – średni i medianowy stosunek zysku do ryzyka.

## Pre-commit

Do automatycznego sprawdzania stylu i bezpieczeństwa używamy narzędzia
[pre-commit](https://pre-commit.com).

```bash
pip install pre-commit
pre-commit install
# uruchom wszystkie hooki na całym repozytorium
pre-commit run --all-files
```

Przykładowy wynik krzywej kapitału (`equity`) znajdziesz w pliku [`docs/equity.csv`](docs/equity.csv).

Plik CSV musi zawierać kolumny: `time, open, high, low, close` (opcjonalnie `volume`).

CLI automatycznie wykrywa separator CSV przy użyciu `csv.Sniffer` i szybkiego
parsera C. W razie potrzeby separator można podać ręcznie opcją `--sep`,
np. `--sep ';'`.

Jeżeli nie podasz `--csv`, program spróbuje znaleźć plik w
`/home/daro/Fxdata/{SYMBOL}_H1.csv` na podstawie wartości `--symbol`. Własną
lokalizację wskażesz właśnie parametrem `--csv`.

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

## Validate live-config

Przed uruchomieniem handlu można zweryfikować poprawność konfiguracji:

```bash
poetry run forest5 validate live-config --yaml config/live.example.yaml
```

Na sukces komenda wypisze komunikat `OK: <symbol> @ <broker>` i zwróci kod 0.

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

Sekcja `time` pozwala łączyć sygnały strategii z modelem czasu:

- `time.model.enabled` – włącza model czasu podczas backtestu (`true`/`false`).
- `time.model.path` – ścieżka do pliku z zapisanym modelem czasu.
- `decision.min_confluence` – minimalna liczba głosów wymaganych do otwarcia pozycji.

Przykładowa konfiguracja:

```yaml
time:
  model:
    enabled: true
    path: "models/model_time.json"
decision:
  min_confluence: 2
```

Przykładowe polecenia:

```bash
python scripts/time_train.py --input data.csv --output models/model_time.json --q-low 0.25 --q-high 0.75
poetry run forest5 backtest --config config/backtest.yaml
poetry run forest5 live --config config/live.yaml --paper
```

Opcje `--q-low` i `--q-high` pozwalają dostroić kwantyle decyzyjne modelu.

## DecisionAgent i `confidence_tech`

`DecisionAgent` scala głosy z trzech źródeł: sygnału technicznego, modelu
czasu oraz opcjonalnego agenta AI.  Każdy głos posiada kierunek (`BUY/SELL` lub
`-1/0/1`) oraz wagę, która decyduje o ostatecznej konfluencji.  Waga głosu
technicznego może być modulowana przez pole `confidence_tech` zwracane przez
strategię.  Wartość jest przycinana do zakresu zdefiniowanego w
`decision.tech.conf_floor` i `decision.tech.conf_cap`.  Jeśli strategia nie
zwróci `confidence_tech`, użyta zostanie wartość
`decision.tech.default_conf_int`.

Globalne wagi dla poszczególnych źródeł ustawia się w sekcji
`decision.weights`:

```yaml
decision:
  weights:
    tech: 1.0   # sygnał techniczny
    ai: 1.0     # agent AI
    time: 1.0   # model czasu
  tech:
    default_conf_int: 1.0
    conf_floor: 0.0
    conf_cap: 1.0
```

Zmniejszenie wag lub zawężenie przedziału `confidence_tech` pozwala
kontrolować wpływ poszczególnych źródeł na końcową decyzję.
