"""Unit tests for the ORB strategy and metrics."""
import pandas as pd
import pytest
from datetime import timezone

from metrics import build_trade_df, build_equity_curve, compute_metrics, _max_consecutive
from strategies.orb import _check_exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(low, high, close, time_str="2024-01-02 13:15:00+00:00"):
    idx = pd.Timestamp(time_str)
    return pd.Series({"open": low, "high": high, "low": low, "close": close}, name=idx)


def _dummy_config():
    return {
        "initial_equity": 10000.0,
        "risk_per_trade": 100.0,
        "volatility_thresholds": {"low": 33, "high": 67},
    }


def _make_trade(direction="long", entry=2070.0, sl=2060.0, tp=2090.0):
    risk = abs(entry - sl)
    return {
        "date": "2024-01-02",
        "direction": direction,
        "entry_time": pd.Timestamp("2024-01-02 13:15:00+00:00"),
        "entry_price": entry,
        "sl": sl,
        "tp": tp,
        "orb_high": 2068.0,
        "orb_low": 2060.0,
        "orb_range": 8.0,
        "regime": "normal",
    }


# ---------------------------------------------------------------------------
# _check_exit
# ---------------------------------------------------------------------------

class TestCheckExit:
    def test_long_tp_hit(self):
        trade = _make_trade("long", entry=2070.0, sl=2060.0, tp=2090.0)
        candle = _make_candle(low=2068.0, high=2092.0, close=2091.0)
        result = _check_exit(candle, trade, tp_r=2.0)
        assert result is not None
        assert result["exit_reason"] == "tp"
        assert result["r_multiple"] == 2.0

    def test_long_sl_hit(self):
        trade = _make_trade("long", entry=2070.0, sl=2060.0, tp=2090.0)
        candle = _make_candle(low=2058.0, high=2072.0, close=2059.0)
        result = _check_exit(candle, trade, tp_r=2.0)
        assert result is not None
        assert result["exit_reason"] == "sl"
        assert result["r_multiple"] == -1.0

    def test_short_tp_hit(self):
        trade = _make_trade("short", entry=2060.0, sl=2070.0, tp=2040.0)
        candle = _make_candle(low=2038.0, high=2062.0, close=2039.0)
        result = _check_exit(candle, trade, tp_r=2.0)
        assert result is not None
        assert result["exit_reason"] == "tp"
        assert result["r_multiple"] == 2.0

    def test_both_hit_assumes_sl(self):
        """Gap-through candle: SL and TP both within range — SL assumed first."""
        trade = _make_trade("long", entry=2070.0, sl=2060.0, tp=2090.0)
        candle = _make_candle(low=2055.0, high=2095.0, close=2080.0)
        result = _check_exit(candle, trade, tp_r=2.0)
        assert result["exit_reason"] == "sl"

    def test_no_exit(self):
        trade = _make_trade("long", entry=2070.0, sl=2060.0, tp=2090.0)
        candle = _make_candle(low=2068.0, high=2075.0, close=2073.0)
        result = _check_exit(candle, trade, tp_r=2.0)
        assert result is None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def _make_trades(self):
        return [
            {"r_multiple": 2.0, "pnl_points": 20.0, "regime": "normal",
             "exit_reason": "tp", "direction": "long", "date": "2024-01-02",
             "exit_time": pd.Timestamp("2024-01-02 20:00:00+00:00")},
            {"r_multiple": -1.0, "pnl_points": -10.0, "regime": "normal",
             "exit_reason": "sl", "direction": "long", "date": "2024-01-03",
             "exit_time": pd.Timestamp("2024-01-03 20:00:00+00:00")},
            {"r_multiple": 2.0, "pnl_points": 20.0, "regime": "high",
             "exit_reason": "tp", "direction": "short", "date": "2024-01-04",
             "exit_time": pd.Timestamp("2024-01-04 20:00:00+00:00")},
            {"r_multiple": -1.0, "pnl_points": -10.0, "regime": "low",
             "exit_reason": "sl", "direction": "long", "date": "2024-01-05",
             "exit_time": pd.Timestamp("2024-01-05 20:00:00+00:00")},
        ]

    def test_win_rate(self):
        trades = self._make_trades()
        df = build_trade_df(trades, risk_per_trade=100.0)
        m = compute_metrics(df, _dummy_config())
        assert m["win_rate_pct"] == 50.0

    def test_profit_factor(self):
        trades = self._make_trades()
        df = build_trade_df(trades, risk_per_trade=100.0)
        m = compute_metrics(df, _dummy_config())
        assert m["profit_factor"] == pytest.approx(2.0, rel=1e-3)

    def test_max_consecutive_losses(self):
        mask = pd.Series([False, True, True, False, True])
        assert _max_consecutive(mask) == 2

    def test_equity_curve_starts_at_initial(self):
        trades = self._make_trades()
        df = build_trade_df(trades, risk_per_trade=100.0)
        curve = build_equity_curve(df, initial_equity=10000.0)
        assert curve["equity"].iloc[0] == pytest.approx(10200.0)

    def test_empty_trades(self):
        df = build_trade_df([], risk_per_trade=100.0)
        m = compute_metrics(df, _dummy_config())
        assert "error" in m
