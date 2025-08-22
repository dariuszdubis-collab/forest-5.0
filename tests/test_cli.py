import pandas as pd
import pytest

from forest5.cli import build_parser, main
from forest5.config_live import LiveSettings


def _write_csv(path):
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_cli_backtest(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--fast",
            "2",
            "--slow",
            "3",
            "--atr-period",
            "1",
        ]
    )
    assert rc == 0


def test_cli_backtest_macd(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)

    captured = {}

    def fake_run_backtest(df, settings, **kwargs):
        captured["settings"] = settings
        class R:
            equity_curve = pd.Series([100.0])
            max_dd = 0.0
            trades = type("T", (), {"trades": []})()
        return R()

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = main(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--fast",
            "2",
            "--slow",
            "3",
            "--strategy",
            "macd_cross",
            "--signal",
            "5",
            "--atr-period",
            "1",
        ]
    )
    assert rc == 0
    s = captured["settings"].strategy
    assert s.name == "macd_cross"
    assert s.signal == 5


def test_cli_grid(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--fast-values",
            "2",
            "--slow-values",
            "3",
            "--jobs",
            "1",
            "--top",
            "1",
        ]
    )
    assert rc == 0


def test_cli_grid_macd(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)

    captured = {}

    def fake_run_grid(df, symbol, fast_values, slow_values, **kwargs):
        captured["kwargs"] = kwargs
        return pd.DataFrame(
            [{"fast": 2, "slow": 3, "equity_end": 1.0, "max_dd": 0.0, "cagr": 0.0, "rar": 0.0}]
        )

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    rc = main(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--fast-values",
            "2",
            "--slow-values",
            "3",
            "--jobs",
            "1",
            "--top",
            "1",
            "--strategy",
            "macd_cross",
            "--signal",
            "5",
        ]
    )
    assert rc == 0
    kw = captured["kwargs"]
    assert kw["strategy_name"] == "macd_cross"
    assert kw["signal"] == 5


def test_cli_missing_file():
    with pytest.raises(FileNotFoundError):
        main(["backtest", "--csv", "no_such_file.csv"])


def test_cli_missing_time_column(tmp_path):
    bad_csv = tmp_path / "bad_time.csv"
    bad_csv.write_text(
        "time,open,high,low,close\n" "bad,1,1,1,1\n" "also_bad,1,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Failed to parse"):
        main(["backtest", "--csv", str(bad_csv), "--time-col", "time"])


def test_cli_missing_ohlc_columns(tmp_path):
    bad_csv = tmp_path / "bad_ohlc.csv"
    bad_csv.write_text("time,open,high,low\n2020-01-01,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV missing required columns"):
        main(["backtest", "--csv", str(bad_csv)])


def test_percentage_out_of_range_error(capfd):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backtest", "--csv", "dummy.csv", "--risk", "2"])
    err = capfd.readouterr().err
    assert "between 0.0 and 1.0" in err


def test_percentage_with_percent_sign():
    parser = build_parser()
    args = parser.parse_args(["backtest", "--csv", "dummy.csv", "--risk", "1%"])
    assert args.risk == pytest.approx(0.01)


def test_percentage_with_comma_and_percent():
    parser = build_parser()
    args = parser.parse_args(["backtest", "--csv", "dummy.csv", "--risk", "0,5%"])
    assert args.risk == pytest.approx(0.005)


def test_cli_live(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("broker:\n  type: mt4\n  symbol: EURUSD\n", encoding="utf-8")

    captured = {}

    def fake_run_live(settings):
        captured["settings"] = settings

    monkeypatch.setattr("forest5.cli.run_live", fake_run_live)

    rc = main(["live", "--config", str(cfg), "--paper"])
    assert rc == 0
    assert isinstance(captured["settings"], LiveSettings)
    assert captured["settings"].broker.type == "paper"
    assert captured["settings"].broker.symbol == "EURUSD"
