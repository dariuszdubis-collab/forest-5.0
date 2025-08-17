# -------- Forest 5.0 Makefile --------

PY ?= python

# Domyślna ścieżka danych i outputów
CSV    ?= data/EURUSD_H1_ML_ready.csv
SYMBOL ?= EURUSD
OUTDIR ?= out

# Domyślne parametry ryzyka / kosztów / ATR
CAPITAL        ?= 100000
RISK           ?= 0.01
FEE_PERC       ?= 0.0005
SLIP_PERC      ?= 0.0
ATR_PERIOD     ?= 14
ATR_MULT       ?= 2.0

# Domyślne okno czasowe
START ?= 2009-07-01
END   ?= 2011-12-31

# Grid: zakresy i opcje
FAST   ?= 5-20
SLOW   ?= 20-60:5
USE_RSI       ?= 1
RSI_PERIOD    ?= 14
RSI_OVERSOLD  ?= 30
RSI_OVERBOUGHT?= 70
DDPEN         ?= 0.5
JOBS          ?= 1
QUIET         ?= 1
SKIP_GE       ?= 1

# Walkforward
TRN_MONTHS ?= 12
TST_MONTHS ?= 3
STEP_MONTHS?= 3

# Ścieżki wyjściowe
GRID_OUT ?= $(OUTDIR)/grid_ema_rsi.csv
SMALL_OUT?= $(OUTDIR)/grid_small.csv
WF_OUT   ?= $(OUTDIR)/walkforward.csv

.PHONY: all lint test grid grid-small walkforward diag clean

all: lint test

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

# Mały grid – szybka walidacja
grid-small:
	$(MAKE) grid FAST="5-10" SLOW="10-20" OUT="$(SMALL_OUT)" END="2010-06-30"

# Pełny grid
grid:
	@mkdir -p $(OUTDIR)
	$(PY) scripts/optimize_grid.py \
	  --csv $(CSV) --symbol $(SYMBOL) \
	  --fast "$(FAST)" --slow "$(SLOW)" \
	  $(if $(filter 1,$(USE_RSI)),--use-rsi,) --rsi-period $(RSI_PERIOD) --rsi-oversold $(RSI_OVERSOLD) --rsi-overbought $(RSI_OVERBOUGHT) \
	  --capital $(CAPITAL) --risk $(RISK) --fee-perc $(FEE_PERC) --slippage-perc $(SLIP_PERC) \
	  --atr-period $(ATR_PERIOD) --atr-multiple $(ATR_MULT) \
	  --start $(START) --end $(END) \
	  --dd-penalty $(DDPEN) --jobs $(JOBS) \
	  $(if $(filter 1,$(QUIET)),--quiet,) \
	  $(if $(filter 1,$(SKIP_GE)),--skip-fast-ge-slow,) \
	  --export $(if $(OUT),$(OUT),$(GRID_OUT))

# Walk-forward: wewnętrzny grid na każdym oknie train => pick best => test
walkforward:
	@mkdir -p $(OUTDIR)
	PYTHONWARNINGS=ignore \
	$(PY) scripts/walkforward.py \
	  --csv $(CSV) --symbol $(SYMBOL) \
	  --fast "$(FAST)" --slow "$(SLOW)" \
	  $(if $(filter 1,$(USE_RSI)),--use-rsi,) --rsi-period $(RSI_PERIOD) --rsi-oversold $(RSI_OVERSOLD) --rsi-overbought $(RSI_OVERBOUGHT) \
	  --capital $(CAPITAL) --risk $(RISK) --fee-perc $(FEE_PERC) --slippage-perc $(SLIP_PERC) \
	  --atr-period $(ATR_PERIOD) --atr-multiple $(ATR_MULT) \
	  --start $(START) --end $(END) \
	  --train-months $(TRN_MONTHS) --test-months $(TST_MONTHS) --step-months $(STEP_MONTHS) \
	  --dd-penalty $(DDPEN) \
	  $(if $(filter 1,$(SKIP_GE)),--skip-fast-ge-slow,) \
	  --export $(WF_OUT)

# Diagnostyka (opcjonalnie) – możesz podmienić parametry
diag:
	@echo "Diag placeholder – dodaj tu własny tracer lub exporter"

clean:
	rm -f $(OUTDIR)/*.csv

