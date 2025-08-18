#!/usr/bin/env bash
set -euo pipefail
WINEPREFIX="${WINEPREFIX:-$HOME/.mt4}"
USER_WIN="${USER_WIN:-$USER}"   # zmień na właściwą nazwę usera w prefixie, jeśli potrzeba
HASH="${MT4_HASH:-CHANGE_ME_HASH}"

ROOT="$(git rev-parse --show-toplevel)"
SRC="$ROOT/mt4/ForestBridge.mq4"
DATA_DIR="$WINEPREFIX/drive_c/users/$USER_WIN/AppData/Roaming/MetaQuotes/Terminal/$HASH"
EXP="$DATA_DIR/MQL4/Experts"
FILES="$DATA_DIR/MQL4/Files"
BRIDGE="$FILES/forest_bridge"

mkdir -p "$EXP" "$BRIDGE"/{ticks,commands,results,state}
cp "$SRC" "$EXP/ForestBridge.mq4"
echo "[OK] Copied EA to: $EXP/ForestBridge.mq4"
echo "[i] Compile in MetaEditor / Navigator->Refresh"
echo "[i] Bridge dir: $BRIDGE"
