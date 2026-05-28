"""Opening Range Breakout (ORB) strategy for XAUUSD."""
import logging
from datetime import time as dtime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_time(t: str) -> dtime:
    h, m = map(int, t.split(":"))
    return dtime(h, m)


def _check_exit(candle: pd.Series, trade: dict, tp_r: float) -> Optional[dict]:
    """
    Evaluate whether TP or SL is hit on this candle.

    Anti-lookahead: decision is based solely on the candle's OHLC that has
    already closed. No intrabar data or future prices are referenced.

    When both TP and SL fall within the candle's range (gap-through), SL is
    assumed to have been hit first — the conservative assumption.
    """
    direction = trade["direction"]
    sl, tp = trade["sl"], trade["tp"]

    if direction == "long":
        sl_hit = candle["low"] <= sl
        tp_hit = candle["high"] >= tp
    else:
        sl_hit = candle["high"] >= sl
        tp_hit = candle["low"] <= tp

    if not sl_hit and not tp_hit:
        return None

    if sl_hit and tp_hit:
        exit_price, exit_reason, r_multiple = sl, "sl", -1.0
    elif sl_hit:
        exit_price, exit_reason, r_multiple = sl, "sl", -1.0
    else:
        exit_price, exit_reason, r_multiple = tp, "tp", tp_r

    pnl_points = (
        exit_price - trade["entry_price"]
        if direction == "long"
        else trade["entry_price"] - exit_price
    )

    return {
        **trade,
        "exit_time": candle.name,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "r_multiple": r_multiple,
        "pnl_points": pnl_points,
    }


def run_orb(df: pd.DataFrame, config: dict) -> list:
    """
    Run the ORB strategy and return a list of completed trade dicts.

    Rules:
    - Opening range: orb_start to orb_end (exclusive)
    - One trade maximum per session
    - Long on first candle close > ORB high
    - Short on first candle close < ORB low
    - Stop loss: opposite ORB boundary
    - Take profit: 2R
    - Open trades closed at session_end at market (EOD exit)
    """
    orb_start = _parse_time(config["orb_start"])
    orb_end = _parse_time(config["orb_end"])
    session_end = _parse_time(config["session_end"])
    tp_r = config["take_profit_r"]
    min_orb_range = config.get("min_orb_range", 0.0)
    allowed_directions = set(config.get("allowed_directions", ["long", "short"]))

    trades = []
    # groupby is O(n) total vs O(n*sessions) for repeated index.date comparisons
    grouped = df.groupby(df.index.date)
    dates = sorted(grouped.groups.keys())

    for date, session_df in grouped:

        # ORB window: [orb_start, orb_end) — candles that have fully closed
        # before the ORB end time. Anti-lookahead: we never peek at the first
        # post-ORB candle to construct the range.
        orb_mask = (session_df.index.time >= orb_start) & (
            session_df.index.time < orb_end
        )
        orb_candles = session_df[orb_mask]

        if orb_candles.empty:
            logger.debug("%s: no ORB candles — skipping", date)
            continue

        orb_high = orb_candles["high"].max()
        orb_low = orb_candles["low"].min()
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            logger.debug("%s: zero ORB range — skipping", date)
            continue

        if orb_range < min_orb_range:
            logger.debug("%s: ORB range %.2f below minimum %.2f — skipping", date, orb_range, min_orb_range)
            continue

        trade_mask = (session_df.index.time >= orb_end) & (
            session_df.index.time <= session_end
        )
        trade_candles = session_df[trade_mask]

        if trade_candles.empty:
            continue

        trade_taken = False
        open_trade: Optional[dict] = None

        for idx, candle in trade_candles.iterrows():
            # Step 1: manage any open trade
            if open_trade is not None:
                result = _check_exit(candle, open_trade, tp_r)
                if result is not None:
                    trades.append(result)
                    open_trade = None

            # Step 2: look for entry signal (one per session)
            if not trade_taken and open_trade is None:
                close = candle["close"]
                regime = candle.get("regime", "normal")

                if close > orb_high and "long" in allowed_directions:
                    entry = close
                    sl = orb_low
                    risk = entry - sl
                    open_trade = {
                        "date": str(date),
                        "direction": "long",
                        "entry_time": idx,
                        "entry_price": entry,
                        "sl": sl,
                        "tp": entry + tp_r * risk,
                        "orb_high": orb_high,
                        "orb_low": orb_low,
                        "orb_range": orb_range,
                        "regime": regime,
                    }
                    trade_taken = True
                    logger.debug(
                        "%s: LONG %.2f  SL %.2f  TP %.2f",
                        date, entry, sl, entry + tp_r * risk,
                    )

                elif close < orb_low and "short" in allowed_directions:
                    entry = close
                    sl = orb_high
                    risk = sl - entry
                    open_trade = {
                        "date": str(date),
                        "direction": "short",
                        "entry_time": idx,
                        "entry_price": entry,
                        "sl": sl,
                        "tp": entry - tp_r * risk,
                        "orb_high": orb_high,
                        "orb_low": orb_low,
                        "orb_range": orb_range,
                        "regime": regime,
                    }
                    trade_taken = True
                    logger.debug(
                        "%s: SHORT %.2f  SL %.2f  TP %.2f",
                        date, entry, sl, entry - tp_r * risk,
                    )

        # EOD: close any trade still open at session end
        if open_trade is not None:
            last = trade_candles.iloc[-1]
            direction = open_trade["direction"]
            exit_price = last["close"]
            pnl_points = (
                exit_price - open_trade["entry_price"]
                if direction == "long"
                else open_trade["entry_price"] - exit_price
            )
            risk = abs(open_trade["entry_price"] - open_trade["sl"])
            r_multiple = pnl_points / risk if risk > 0 else 0.0

            trades.append(
                {
                    **open_trade,
                    "exit_time": last.name,
                    "exit_price": exit_price,
                    "exit_reason": "eod",
                    "r_multiple": r_multiple,
                    "pnl_points": pnl_points,
                }
            )

    logger.info("ORB strategy: %d trades across %d sessions", len(trades), len(dates))
    return trades
