# xauusd-research

XAU/USD trading research and utilities — ORB backtesting framework with volatility regime classification.

## Project structure

```
backtest.py          # CLI entry point
data_loader.py       # data loading, cleaning, ATR, regime classification
metrics.py           # metrics computation and output writers
strategies/
  orb.py             # Opening Range Breakout strategy
configs/
  orb_config.yaml    # strategy configuration
data/
  raw/               # TradingView CSV exports go here
  clean/             # processed data (auto-generated)
reports/             # trade_log.csv, equity_curve.csv, summary_metrics.csv
logs/                # backtest.log
tests/
  test_orb.py        # unit tests
notebooks/           # Jupyter notebooks for analysis
sample_data.csv      # example CSV schema
```

## Setup

```bash
pip install pandas numpy pyyaml
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Exporting data from TradingView

1. Open TradingView and load the **XAUUSD** chart
2. Set the timeframe to **5 minutes** (recommended) or 1 minute
3. Go to **Export chart data**: right-click the chart → *Download chart data…*
   - Alternatively: *Chart properties* → *Export* → *Export chart data*
4. Save the file to `data/raw/xauusd_5m.csv`
5. TradingView exports UTC timestamps — the backtester converts them automatically

**Expected CSV schema:**

| Column | Type | Example |
|--------|------|---------|
| `time` | UTC datetime or Unix timestamp | `2024-01-02 13:00:00+00:00` |
| `open` | float | `2063.40` |
| `high` | float | `2064.10` |
| `low` | float | `2062.80` |
| `close` | float | `2063.75` |
| `volume` | int | `312` |

See `sample_data.csv` for a minimal working example.

## Running the backtest

```bash
python backtest.py data/raw/xauusd_5m.csv
```

With custom config or output directory:

```bash
python backtest.py data/raw/xauusd_5m.csv \
  --config configs/orb_config.yaml \
  --output-dir reports \
  --log-dir logs
```

## Strategy: Opening Range Breakout (ORB)

| Parameter | Value |
|-----------|-------|
| Instrument | XAUUSD |
| Timezone | America/Los_Angeles |
| Opening range | 5:00 AM – 5:15 AM PT |
| Entry | Candle close above ORB high (long) / below ORB low (short) |
| Stop loss | Opposite ORB boundary |
| Take profit | 2R (fixed) |
| Max trades | 1 per session |
| Session end | 3:00 PM PT (open trades closed at market) |

Anti-lookahead protections:
- ORB high/low computed only from candles that closed before 5:15 AM
- Volatility regime uses a rolling percentile over past bars only — current bar excluded from its own reference distribution
- Entry signals trigger on candle *close*, not intrabar

## Interpreting the outputs

### `reports/trade_log.csv`
One row per trade. Key columns:

| Column | Meaning |
|--------|---------|
| `r_multiple` | Outcome in R. `+2.0` = TP hit, `-1.0` = SL hit |
| `exit_reason` | `tp` / `sl` / `eod` (closed at session end) |
| `regime` | Volatility regime at entry: `low` / `normal` / `high` |
| `pnl_dollars` | P&L in dollars based on `risk_per_trade` in config |

### `reports/equity_curve.csv`
Equity after each trade. `drawdown_pct` is the percentage drop from the most recent peak — use this to assess risk of ruin.

### `reports/summary_metrics.csv`
Flat key/value file. Key metrics explained:

| Metric | What it means |
|--------|--------------|
| `win_rate_pct` | % of trades that were profitable |
| `expectancy_r` | Average R earned per trade. Positive = edge exists |
| `profit_factor` | Gross profit / gross loss. >1.5 is decent, >2.0 is strong |
| `max_drawdown_pct` | Worst peak-to-trough equity drop |
| `max_consecutive_losses` | Longest losing streak — size positions accordingly |
| `regime_breakdown.*` | Performance split by volatility regime |
| `exit_reason_breakdown.*` | How many trades hit TP vs SL vs closed EOD |

**Rule of thumb:** if `expectancy_r` is negative or `profit_factor` < 1.0, the strategy has no edge on this data. Check regime breakdown — the strategy may work in specific volatility conditions only.

## Running tests

```bash
pip install pytest
pytest tests/
```

## Pre-commit review

Every commit is automatically reviewed by GPT-4o. Reviews are saved to `reviews/`.

## Files

| File | Description |
|------|-------------|
| `trade_bot.py` | Price fetcher and trade execution loop |
| `utils.py` | Password hashing, config loading, webhook delivery |
| `backtest.py` | ORB backtester CLI |
| `data_loader.py` | Data pipeline |
| `strategies/orb.py` | ORB strategy logic |
| `metrics.py` | Metrics and output writers |
| `configs/orb_config.yaml` | Strategy configuration |
| `sample_data.csv` | Example CSV schema |
