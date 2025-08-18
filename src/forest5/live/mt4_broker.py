from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from .router import OrderRouter, OrderResult

log = logging.getLogger(__name__)


class MT4Broker(OrderRouter):
    """Broker współpracujący z mostem plikowym MetaTrader4.

    Komunikacja odbywa się przez katalog zawierający podkatalogi
    ``ticks/``, ``commands/``, ``results/`` oraz ``state/``.
    """

    def __init__(
        self,
        bridge_dir: Optional[str | Path] = None,
        *,
        config_path: Optional[str | Path] = None,
        timeout: float = 5.0,
    ) -> None:
        self._connected = False
        self.timeout = float(timeout)
        if bridge_dir is None:
            env_dir = os.getenv("FOREST_MT4_BRIDGE_DIR")
            if env_dir:
                bridge_dir = Path(env_dir)
            elif config_path is not None:
                bridge_dir = self._load_bridge_dir_from_yaml(config_path)
            else:
                raise ValueError("bridge directory not specified")
        else:
            bridge_dir = Path(bridge_dir)

        self.bridge_dir = Path(bridge_dir)
        self.commands_dir = self.bridge_dir / "commands"
        self.results_dir = self.bridge_dir / "results"
        self.state_dir = self.bridge_dir / "state"
        self.ticks_dir = self.bridge_dir / "ticks"

    # ------------------------------------------------------------------
    def _load_bridge_dir_from_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        val = data.get("mt4_bridge_dir") or data.get("bridge_dir")
        if not val:
            raise KeyError("bridge_dir not found in config")
        return Path(val)

    # ------------------------------------------------------------------
    def connect(self) -> None:
        for d in [
            self.commands_dir,
            self.results_dir,
            self.state_dir,
            self.ticks_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        self._connected = True
        log.info("MT4Broker connected: %s", self.bridge_dir)

    def close(self) -> None:
        self._connected = False
        log.info("MT4Broker closed")

    # ------------------------------------------------------------------
    def _command_path(self, uid: str) -> Path:
        return self.commands_dir / f"cmd_{uid}.json"

    def _result_path(self, uid: str) -> Path:
        return self.results_dir / f"res_{uid}.json"

    def _wait_for_result(self, uid: str) -> OrderResult:
        res_path = self._result_path(uid)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if res_path.exists():
                try:
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                    return OrderResult(
                        int(data.get("id", 0)),
                        data.get("status", "rejected"),
                        float(data.get("filled_qty", 0.0)),
                        float(data.get("avg_price", 0.0)),
                        data.get("error"),
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    log.exception("invalid result: %s", exc)
                    break
            time.sleep(0.1)
        log.error("timeout waiting for result %s", uid)
        return OrderResult(0, "rejected", 0.0, 0.0, "timeout")

    # ------------------------------------------------------------------
    def market_order(
        self, side: str, qty: float, price: Optional[float] = None
    ) -> OrderResult:
        if not self._connected:
            return OrderResult(0, "rejected", 0.0, 0.0, "not connected")

        uid = uuid.uuid4().hex
        cmd = {
            "id": uid,
            "type": "market_order",
            "side": side,
            "qty": qty,
        }
        if price is not None:
            cmd["price"] = price

        cmd_path = self._command_path(uid)
        cmd_path.write_text(json.dumps(cmd), encoding="utf-8")
        log.info("command written: %s", cmd_path)
        return self._wait_for_result(uid)

    # ------------------------------------------------------------------
    def position_qty(self) -> float:
        pos_file = self.state_dir / "position.json"
        try:
            data = json.loads(pos_file.read_text(encoding="utf-8"))
            return float(data.get("qty", 0.0))
        except FileNotFoundError:
            log.warning("position file missing: %s", pos_file)
            return 0.0
        except Exception:  # pragma: no cover - defensive
            log.exception("error reading position file")
            return 0.0

    def equity(self) -> float:
        acc_file = self.state_dir / "account.json"
        try:
            data = json.loads(acc_file.read_text(encoding="utf-8"))
            return float(data.get("equity", 0.0))
        except FileNotFoundError:
            log.warning("account file missing: %s", acc_file)
            return 0.0
        except Exception:  # pragma: no cover - defensive
            log.exception("error reading account file")
            return 0.0

    # ------------------------------------------------------------------
    def set_price(self, price: float) -> None:  # pragma: no cover - unused
        # interfejs wymagany przez OrderRouter, ale nieużywany w MT4 bridge
        pass
