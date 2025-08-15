from forest5.live.router import PaperBroker


def test_paper_broker_basic_flow():
    b = PaperBroker(fee_perc=0.0)
    b.connect()
    b.set_price(100.0)
    r = b.market_order("BUY", 10)
    assert r.status == "filled"
    assert b.position_qty() == 10
    b.set_price(110.0)
    eq_before = b.equity()
    r2 = b.market_order("SELL", 10)
    assert r2.status == "filled"
    assert b.position_qty() == 0
    assert b.equity() > eq_before


def test_rejects_without_price_or_connection():
    b = PaperBroker()
    assert b.market_order("BUY", 1).status == "rejected"  # not connected
    b.connect()
    assert b.market_order("BUY", 1).status == "rejected"  # no price


