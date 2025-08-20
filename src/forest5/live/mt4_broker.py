from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from .router import OrderRouter, OrderResult
from ..utils.log import log


class MT4Broker(OrderRouter):
    """Broker współpracujący z mostem plikowym MetaTrader4.

    Komunikacja odbywa się przez katalog zawierający podkatalogi
    ``ticks/``, ``commands/``, ``results/`` oraz ``state/``.
    """

    def __init__(
        self,
        bridge_dir: Optional[str | Path] = None,
        *,
        symbol: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        timeout_sec: float = 5.0,
        timeout: Optional[float] = None,
    ) -> None:
        self._connected = False
        self._id = 0
        if timeout is not None:
            self.timeout = float(timeout)
        else:
            self.timeout = float(timeout_sec)

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

        if symbol is None:
            raise ValueError("symbol required")

        self.symbol = str(symbol)
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
        log.info("broker_connected", bridge_dir=str(self.bridge_dir))

    def close(self) -> None:
        self._connected = False
        log.info("broker_closed")

    # ------------------------------------------------------------------
    def _command_path(self, uid: str) -> Path:
        return self.commands_dir / f"cmd_{uid}.json"

    def _result_path(self, uid: str) -> Path:
        return self.results_dir / f"res_{uid}.json"

    def _wait_for_result(self, uid: str, qty: float) -> OrderResult:
        res_path = self._result_path(uid)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if res_path.exists():
                try:
                    if res_path.stat().st_size == 0:
                        time.sleep(0.1)
                        continue
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                    status = data.get("status", "rejected")
                    price = float(data.get("price", data.get("avg_price", 0.0)))
                    ticket = data.get("ticket", 0)
                    err = data.get("error")
                    filled = qty if status == "filled" else 0.0
                    return OrderResult(int(ticket) if isinstance(ticket, int) else 0, status, filled, price, err)
                except json.JSONDecodeError:
                    log.warning("invalid_json_result", path=str(res_path))
                    time.sleep(0.1)
                    continue
                except Exception as exc:  # pragma: no cover - defensive
                    log.exception("invalid_result", path=str(res_path), error=str(exc))
                    time.sleep(0.1)
                    continue
            time.sleep(0.1)
        log.error("timeout_waiting_for_result", order_id=uid)
        return OrderResult(0, "rejected", 0.0, 0.0, "timeout")

    # ------------------------------------------------------------------
    def market_order(
        self, side: str, qty: float, price: Optional[float] = None
    ) -> OrderResult:
        if not self._connected:
            res = OrderResult(0, "rejected", 0.0, 0.0, "not connected")
            log.info(
                "order_result",
                timestamp=time.time(),
                mode="live",
                symbol=self.symbol,
                action="market_order",
                side=side.upper(),
                qty=qty,
                price=price,
                latency_ms=0.0,
                decision=side.upper(),
                votes=None,
                reason="not connected",
                error=res.error,
            )
            return res

        start_ts = time.time()
        uid = uuid.uuid4().hex
        cmd = {
            "id": uid,
            "action": side.upper(),
            "symbol": self.symbol,
            "volume": qty,
            "sl": None,
            "tp": None,
        }
        if price is not None:
            cmd["price"] = price

        cmd_path = self._command_path(uid)
        cmd_path.write_text(json.dumps(cmd), encoding="utf-8")
        send_latency_ms = (time.time() - start_ts) * 1000.0
        log.info(
            "order_sent",
            timestamp=start_ts,
            mode="live",
            symbol=self.symbol,
            action="market_order",
            side=side.upper(),
            qty=qty,
            price=price,
            latency_ms=send_latency_ms,
            decision=side.upper(),
            votes=None,
            reason=None,
            error=None,
        )

        res = self._wait_for_result(uid, qty)
        total_latency_ms = (time.time() - start_ts) * 1000.0
        log.info(
            "order_result",
            timestamp=time.time(),
            mode="live",
            symbol=self.symbol,
            action="market_order",
            side=side.upper(),
            qty=qty,
            price=res.avg_price,
            latency_ms=total_latency_ms,
            decision=side.upper(),
            votes=None,
            reason=None,
            error=res.error,
        )
        return res

    # ------------------------------------------------------------------
    def position_qty(self) -> float:
        pos_file = self.state_dir / f"position_{self.symbol}.json"
        try:
            data = json.loads(pos_file.read_text(encoding="utf-8"))
            return float(data.get("qty", 0.0))
        except FileNotFoundError:
            log.warning("position_file_missing", path=str(pos_file))
            return 0.0
        except Exception:  # pragma: no cover - defensive
            log.exception("position_file_error")
            return 0.0

    def equity(self) -> float:
        acc_file = self.state_dir / "account.json"
        try:
            data = json.loads(acc_file.read_text(encoding="utf-8"))
            return float(data.get("equity", 0.0))
        except FileNotFoundError:
            log.warning("account_file_missing", path=str(acc_file))
            return 0.0
        except Exception:  # pragma: no cover - defensive
            log.exception("account_file_error")
            return 0.0

    # ------------------------------------------------------------------
    def set_price(self, price: float) -> None:  # pragma: no cover - unused
        # interfejs wymagany przez OrderRouter, ale nieużywany w MT4 bridge
        pass
