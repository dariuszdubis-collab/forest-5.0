from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from .router import OrderRouter, OrderResult
from ..utils.log import (
    E_BROKER_ADJUSTED_STOPS,
    E_ORDER_ACK,
    E_ORDER_FILLED,
    E_ORDER_REJECTED,
    E_ORDER_RETRY,
    E_ORDER_TIMEOUT,
    TelemetryContext,
    log_event,
    setup_logger,
)


log = setup_logger()


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
    def _load_stop_level(self) -> float | None:
        """Return minimum stop distance in price units if available."""
        path = self.state_dir / f"stop_level_{self.symbol}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            val = float(data.get("stop_level"))
            return val if val > 0 else None
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            log.warning("stop_level_invalid_json", path=str(path))
            return None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            log.warning("stop_level_invalid_value", path=str(path))
            return None

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

    # ------------------------------------------------------------------
    # Handshake utilities ------------------------------------------------
    def request_specs(self) -> str:
        """Request symbol specifications from the bridge.

        Returns the unique request id which should match the corresponding
        acknowledgement file name created by the Expert Advisor.
        """

        uid = uuid.uuid4().hex
        payload = {"id": uid, "action": "GET_SPECS", "symbol": self.symbol}
        path = self.commands_dir / f"req_{uid}.json"
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, path)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        return uid

    def await_ack(self, uid: str, timeout: float | None = None) -> Dict[str, Any]:
        """Wait for ``ack_<uid>.json`` and return parsed specs.

        Parameters
        ----------
        uid:
            Identifier returned by :meth:`request_specs`.
        timeout:
            Optional timeout overriding the broker's default timeout.

        Raises
        ------
        TimeoutError
            If the acknowledgement file is not produced within ``timeout``
            seconds.
        """

        path = self.results_dir / f"ack_{uid}.json"
        deadline = time.time() + (timeout if timeout is not None else self.timeout)
        delay = 0.1
        while time.time() < deadline:
            if path.exists():
                try:
                    if path.stat().st_size == 0:
                        time.sleep(min(delay, max(0, deadline - time.time())))
                        continue
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(min(delay, max(0, deadline - time.time())))
                    continue
            time.sleep(min(delay, max(0, deadline - time.time())))
        raise TimeoutError(f"timeout waiting for ack {uid}")

    def validate_specs(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalise symbol specification dictionary.

        The bridge should provide fields like ``digits``, ``point`` and
        ``tick_size``. The function ensures required keys exist and converts
        values to ``float``/``int`` as appropriate. It returns the normalised
        dictionary on success and raises :class:`ValueError` on problems.
        """

        required = [
            "digits",
            "point",
            "tick_size",
            "min_lot",
            "lot_step",
            "contract_size",
            "stop_level",
            "freeze_level",
        ]
        missing = [k for k in required if k not in specs]
        if missing:
            raise ValueError(f"missing fields: {missing}")

        norm: Dict[str, Any] = {}
        for k in required:
            v = specs[k]
            if k == "digits":
                norm[k] = int(v)
            else:
                norm[k] = float(v)
        return norm

    def _wait_for_result(
        self,
        uid: str,
        qty: float,
        *,
        ctx: TelemetryContext | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        res_path = self._result_path(uid)
        deadline = time.time() + self.timeout
        attempt = 0
        delay = 0.1
        while time.time() < deadline:
            if res_path.exists():
                try:
                    if res_path.stat().st_size == 0:
                        time.sleep(min(delay, max(0, deadline - time.time())))
                        continue
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                    status = data.get("status", "rejected")
                    price = float(data.get("price", data.get("avg_price", 0.0)))
                    ticket = data.get("ticket", 0)
                    err = data.get("error")
                    filled = qty if status == "filled" else 0.0
                    adj_sl = data.get("sl") or data.get("new_sl")
                    adj_tp = data.get("tp") or data.get("new_tp")
                    log_event(
                        E_ORDER_ACK,
                        ctx,
                        client_order_id=client_order_id,
                        ticket=ticket,
                    )
                    if status == "filled":
                        log_event(
                            E_ORDER_FILLED,
                            ctx,
                            client_order_id=client_order_id,
                            ticket=ticket,
                            fill_price=price,
                            fill_qty=filled,
                        )
                    else:
                        log_event(
                            E_ORDER_REJECTED,
                            ctx,
                            client_order_id=client_order_id,
                            ticket=ticket,
                            reason=err,
                        )
                    res = OrderResult(
                        int(ticket) if isinstance(ticket, int) else 0, status, filled, price, err
                    )
                    if adj_sl is not None or adj_tp is not None:
                        try:
                            if adj_sl is not None:
                                setattr(res, "sl", float(adj_sl))
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            pass
                        try:
                            if adj_tp is not None:
                                setattr(res, "tp", float(adj_tp))
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            pass
                    return res
                except json.JSONDecodeError:
                    attempt += 1
                    log_event(
                        E_ORDER_RETRY,
                        ctx,
                        client_order_id=client_order_id,
                        attempt=attempt,
                        reason="invalid_json",
                    )
                    log.warning("invalid_json_result", path=str(res_path), attempt=attempt)
                    time.sleep(min(delay, max(0, deadline - time.time())))
                    delay = min(delay * 2, 1.0)
                    continue
                except (OSError, ValueError, TypeError) as exc:  # pragma: no cover - defensive
                    log_event(
                        E_ORDER_RETRY,
                        ctx,
                        client_order_id=client_order_id,
                        reason=str(exc),
                        attempt=attempt,
                    )
                    log.exception("invalid_result", path=str(res_path), error=str(exc))
                    time.sleep(min(delay, max(0, deadline - time.time())))
                    continue
            time.sleep(min(delay, max(0, deadline - time.time())))
        log_event(E_ORDER_TIMEOUT, ctx, client_order_id=client_order_id)
        log.error("timeout_waiting_for_result", order_id=uid)
        return OrderResult(0, "rejected", 0.0, 0.0, "timeout")

    # ------------------------------------------------------------------
    def market_order(
        self,
        side: str,
        qty: float,
        price: Optional[float] = None,
        *,
        entry: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        ctx: TelemetryContext | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        if not self._connected:
            res = OrderResult(0, "rejected", 0.0, 0.0, "not connected")
            log_event(
                E_ORDER_ACK,
                ctx,
                client_order_id=client_order_id,
                ticket=0,
            )
            log_event(
                E_ORDER_REJECTED,
                ctx,
                client_order_id=client_order_id,
                ticket=0,
                reason="not connected",
            )
            log.info(
                "order_result",
                timestamp=time.time(),
                symbol=self.symbol,
                action="market_order",
                side=side.upper(),
                qty=0.0,
                price=price,
                latency_ms=0.0,
                error=res.error,
                context={"status": res.status, "id": res.id},
            )
            return res

        start_ts = time.time()
        uid = uuid.uuid4().hex
        # stop-level enforcement -------------------------------------------------
        price_for_stops = entry if entry is not None else price
        if price_for_stops is None:
            tick_path = self.ticks_dir / "tick.json"
            try:
                tick = json.loads(tick_path.read_text(encoding="utf-8"))
                if side.upper() == "BUY":
                    price_for_stops = float(tick.get("ask"))
                else:
                    price_for_stops = float(tick.get("bid"))
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
                price_for_stops = None

        stop_level = self._load_stop_level()
        if stop_level is not None and price_for_stops is not None and side.upper() == "BUY":
            old_sl, old_tp = sl, tp
            new_sl, new_tp = sl, tp
            if sl is not None and price_for_stops - sl < stop_level:
                new_sl = price_for_stops - stop_level
            if tp is not None and tp - price_for_stops < stop_level:
                new_tp = price_for_stops + stop_level
            if new_sl != old_sl or new_tp != old_tp:
                log.info(
                    E_BROKER_ADJUSTED_STOPS,
                    old_sl=old_sl,
                    old_tp=old_tp,
                    new_sl=new_sl,
                    new_tp=new_tp,
                )
                sl, tp = new_sl, new_tp

        cmd = {
            "id": uid,
            "action": side.upper(),
            "symbol": self.symbol,
            "volume": qty,
            "price": price,
            "sl": sl,
            "tp": tp,
        }

        cmd_path = self._command_path(uid)
        tmp_path = self.commands_dir / f"cmd_{uid}.json.tmp"
        try:
            tmp_path.write_text(json.dumps(cmd), encoding="utf-8")
            os.replace(tmp_path, cmd_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        send_latency_ms = (time.time() - start_ts) * 1000.0
        log.info(
            "order_sent",
            timestamp=start_ts,
            symbol=self.symbol,
            action="market_order",
            side=side.upper(),
            qty=qty,
            price=price,
            latency_ms=send_latency_ms,
            error=None,
            context={"id": uid},
        )

        res = self._wait_for_result(
            uid,
            qty,
            ctx=ctx,
            client_order_id=client_order_id,
        )
        total_latency_ms = (time.time() - start_ts) * 1000.0
        log.info(
            "order_result",
            timestamp=time.time(),
            symbol=self.symbol,
            action="market_order",
            side=side.upper(),
            qty=res.filled_qty,
            price=res.avg_price,
            latency_ms=total_latency_ms,
            error=res.error,
            context={"status": res.status, "id": res.id},
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
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):  # pragma: no cover - defensive
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
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):  # pragma: no cover - defensive
            log.exception("account_file_error")
            return 0.0

    # ------------------------------------------------------------------
    def set_price(self, price: float) -> None:  # pragma: no cover - unused
        # interfejs wymagany przez OrderRouter, ale nieużywany w MT4 bridge
        pass
