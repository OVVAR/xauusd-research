"""Metrics computation and output writers for the ORB backtest."""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADE_LOG_COLS = [
    "date", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "sl", "tp",
    "orb_high", "orb_low", "orb_range",
    "exit_reason", "r_multiple", "pnl_points", "pnl_dollars", "regime",
]


def build_trade_df(trades: list, risk_per_trade: float) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["pnl_dollars"] = df["r_multiple"] * risk_per_trade
    return df


def build_equity_curve(trade_df: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["exit_time", "equity", "drawdown", "drawdown_pct"])
    curve = trade_df[["exit_time", "pnl_dollars"]].copy()
    curve["equity"] = initial_equity + curve["pnl_dollars"].cumsum()
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] - curve["peak"]
    curve["drawdown_pct"] = curve["drawdown"] / curve["peak"] * 100
    return curve[["exit_time", "equity", "drawdown", "drawdown_pct"]]


def _max_consecutive(mask: pd.Series) -> int:
    count, best = 0, 0
    for v in mask:
        count = count + 1 if v else 0
        best = max(best, count)
    return best


def _group_stats(group_df: pd.DataFrame) -> dict:
    wins = (group_df["r_multiple"] > 0).sum()
    total = len(group_df)
    return {
        "trades": total,
        "wins": int(wins),
        "win_rate_pct": round(wins / total * 100, 1) if total else 0,
        "avg_r": round(group_df["r_multiple"].mean(), 3),
        "total_r": round(group_df["r_multiple"].sum(), 3),
        "total_pnl": round(group_df["pnl_dollars"].sum(), 2),
    }


def compute_metrics(trade_df: pd.DataFrame, config: dict) -> dict:
    if trade_df.empty:
        return {"error": "no trades generated"}

    wins = trade_df[trade_df["r_multiple"] > 0]
    losses = trade_df[trade_df["r_multiple"] <= 0]

    win_rate = len(wins) / len(trade_df)
    avg_win_r = wins["r_multiple"].mean() if len(wins) else 0.0
    avg_loss_r = losses["r_multiple"].mean() if len(losses) else 0.0
    expectancy_r = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r

    gross_profit = wins["pnl_dollars"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["pnl_dollars"].sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    curve = build_equity_curve(trade_df, config["initial_equity"])
    max_dd = float(curve["drawdown"].min()) if not curve.empty else 0.0
    max_dd_pct = float(curve["drawdown_pct"].min()) if not curve.empty else 0.0

    metrics: dict[str, Any] = {
        "total_trades": len(trade_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_r_multiple": round(float(trade_df["r_multiple"].mean()), 3),
        "expectancy_r": round(expectancy_r, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_consecutive_losses": _max_consecutive(trade_df["r_multiple"] <= 0),
        "max_consecutive_wins": _max_consecutive(trade_df["r_multiple"] > 0),
        "total_pnl_dollars": round(float(trade_df["pnl_dollars"].sum()), 2),
    }

    # Breakdowns
    metrics["regime_breakdown"] = {
        regime: _group_stats(gdf)
        for regime, gdf in trade_df.groupby("regime")
    }
    metrics["exit_reason_breakdown"] = {
        reason: _group_stats(gdf)
        for reason, gdf in trade_df.groupby("exit_reason")
    }
    metrics["direction_breakdown"] = {
        direction: _group_stats(gdf)
        for direction, gdf in trade_df.groupby("direction")
    }

    return metrics


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def save_outputs(trades: list, config: dict, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not trades:
        logger.warning("No trades to save.")
        return

    trade_df = build_trade_df(trades, config["risk_per_trade"])

    # trade_log.csv
    cols = [c for c in TRADE_LOG_COLS if c in trade_df.columns]
    trade_df[cols].to_csv(f"{output_dir}/trade_log.csv", index=False)
    logger.info("Saved trade_log.csv (%d rows)", len(trade_df))

    # equity_curve.csv
    curve = build_equity_curve(trade_df, config["initial_equity"])
    curve.to_csv(f"{output_dir}/equity_curve.csv", index=False)
    logger.info("Saved equity_curve.csv")

    # summary_metrics.csv
    metrics = compute_metrics(trade_df, config)
    flat = _flatten(metrics)
    pd.DataFrame(flat.items(), columns=["metric", "value"]).to_csv(
        f"{output_dir}/summary_metrics.csv", index=False
    )
    logger.info("Saved summary_metrics.csv")

    # Print summary to console
    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    for key in [
        "total_trades", "win_rate_pct", "expectancy_r",
        "profit_factor", "max_drawdown_pct", "total_pnl_dollars",
        "max_consecutive_losses",
    ]:
        print(f"  {key:<30} {metrics.get(key)}")
    print("=" * 50 + "\n")
