"""Load, clean, and enrich TradingView OHLCV data."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_OHLCV_COLS = {"open", "high", "low", "close", "volume"}
_OHLCV_DTYPE = {"open": "float32", "high": "float32", "low": "float32",
                "close": "float32", "volume": "float32"}


def load_raw(path: str) -> pd.DataFrame:
    # Peek at headers to select only OHLCV columns — skips TradingView indicator
    # columns (Volume MA, OR High, VWAP, etc.) that waste memory and parse time.
    header = pd.read_csv(path, nrows=0)
    orig_cols = list(header.columns)
    norm_cols = [c.lower().strip() for c in orig_cols]
    if "time" not in norm_cols:
        raise ValueError("CSV must contain a 'time' column")

    # Map normalised → original name for usecols (read_csv needs original names)
    norm_to_orig = dict(zip(norm_cols, orig_cols))
    keep_norm = ["time"] + [c for c in norm_cols if c in _OHLCV_COLS]
    keep_orig = [norm_to_orig[c] for c in keep_norm]
    dtype_map = {norm_to_orig[k]: v for k, v in _OHLCV_DTYPE.items() if k in keep_norm}

    df = pd.read_csv(path, usecols=keep_orig, dtype=dtype_map)
    df.columns = [c.lower().strip() for c in df.columns]

    sample = str(df["time"].iloc[0]).strip()
    if sample.isdigit():
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    df = df.set_index("time").sort_index()
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info("Loaded %.1f MB from %s (%d cols)", mb, path, len(df.columns))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["high"] >= df["low"]) & (df["open"] > 0) & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="first")]
    logger.info(f"Cleaned data: {before} -> {len(df)} rows")
    return df


def convert_timezone(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df.index = df.index.tz_convert(tz)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_volatility_regime(
    atr: pd.Series,
    window: int = 252,
    low_pct: int = 33,
    high_pct: int = 67,
) -> pd.Series:
    """
    Classify each bar into low/normal/high volatility.

    Anti-lookahead: the percentile rank of the current ATR value is computed
    only against the preceding `window` bars — the current bar is excluded
    from its own reference distribution.
    """

    def pct_rank(x: np.ndarray) -> float:
        if len(x) < 2:
            return 50.0
        history, current = x[:-1], x[-1]
        return float((history < current).sum()) / len(history) * 100

    rolling_pct = atr.rolling(window, min_periods=2).apply(pct_rank, raw=True)

    regime = pd.Series("normal", index=atr.index, dtype=object)
    regime[rolling_pct < low_pct] = "low"
    regime[rolling_pct > high_pct] = "high"
    return regime


def load_and_prepare(path: str, config: dict) -> pd.DataFrame:
    df = load_raw(path)
    df = clean(df)
    df = convert_timezone(df, config["timezone"])

    atr = compute_atr(df, config["atr_period"])
    df = df.copy()
    df["atr"] = atr
    df["regime"] = compute_volatility_regime(
        atr,
        window=config["atr_percentile_window"],
        low_pct=config["volatility_thresholds"]["low"],
        high_pct=config["volatility_thresholds"]["high"],
    )
    return df
