#!/usr/bin/env python3
"""
Robustness study for the XAUUSD ORB strategy.

Runs the strategy EXACTLY as configured — no parameter changes.
Performs statistical analysis, Monte Carlo simulation, and chart generation.
"""
import argparse
import logging
import math
import sys
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

from data_loader import load_and_prepare
from metrics import build_equity_curve, build_trade_df
from strategies.orb import run_orb

logger = logging.getLogger(__name__)

BREAKEVEN_WIN_RATE = 1 / 3  # 33.3% at 2R


# ─── logging ─────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/robustness.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ─── risk metrics ─────────────────────────────────────────────────────────────

def sharpe_ratio(r: pd.Series, trades_per_year: float) -> float:
    return float(r.mean() / r.std() * math.sqrt(trades_per_year)) if r.std() > 0 else 0.0


def sortino_ratio(r: pd.Series, trades_per_year: float) -> float:
    neg = r[r < 0]
    if neg.empty or neg.std() == 0:
        return float("inf") if r.mean() > 0 else 0.0
    return float(r.mean() / neg.std() * math.sqrt(trades_per_year))


def ulcer_index(equity: pd.Series) -> float:
    dd_pct = (equity - equity.cummax()) / equity.cummax() * 100
    return float(np.sqrt((dd_pct ** 2).mean()))


def recovery_factor(equity: pd.Series, initial: float) -> float:
    net = float(equity.iloc[-1] - initial)
    max_dd = float((equity - equity.cummax()).min())
    return round(net / abs(max_dd), 3) if max_dd != 0 else float("inf")


def stability_r2(equity: pd.Series) -> float:
    """R² of linear regression through equity curve. 1.0 = perfectly smooth."""
    if len(equity) < 2:
        return 0.0
    x = np.arange(len(equity), dtype=float)
    y = equity.values.astype(float)
    y_hat = np.polyval(np.polyfit(x, y, 1), x)
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return round(float(1 - ss_res / ss_tot), 4) if ss_tot > 0 else 0.0


# ─── time breakdowns ──────────────────────────────────────────────────────────

def monthly_performance(trade_df: pd.DataFrame) -> pd.DataFrame:
    df = trade_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["month"] = df["exit_time"].dt.to_period("M")
    rows = []
    for period, g in df.groupby("month"):
        wins = int((g["r_multiple"] > 0).sum())
        n = len(g)
        rows.append({
            "month": str(period),
            "trades": n,
            "wins": wins,
            "win_rate_pct": round(wins / n * 100, 1),
            "total_r": round(g["r_multiple"].sum(), 3),
            "total_pnl": round(g["pnl_dollars"].sum(), 2),
            "avg_r": round(g["r_multiple"].mean(), 3),
        })
    return pd.DataFrame(rows)


def yearly_performance(trade_df: pd.DataFrame) -> pd.DataFrame:
    df = trade_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    rows = []
    for year, g in df.groupby("year"):
        wins = int((g["r_multiple"] > 0).sum())
        n = len(g)
        rows.append({
            "year": year,
            "trades": n,
            "wins": wins,
            "win_rate_pct": round(wins / n * 100, 1),
            "total_r": round(g["r_multiple"].sum(), 3),
            "total_pnl": round(g["pnl_dollars"].sum(), 2),
            "avg_r": round(g["r_multiple"].mean(), 3),
        })
    return pd.DataFrame(rows)


# ─── segmentation ─────────────────────────────────────────────────────────────

def _group_stats(g: pd.DataFrame) -> dict:
    wins = int((g["r_multiple"] > 0).sum())
    n = len(g)
    return {
        "trades": n,
        "wins": wins,
        "win_rate_pct": round(wins / n * 100, 1) if n else 0,
        "avg_r": round(float(g["r_multiple"].mean()), 3),
        "total_r": round(float(g["r_multiple"].sum()), 3),
        "total_pnl": round(float(g["pnl_dollars"].sum()), 2),
        "expectancy_r": round(float(g["r_multiple"].mean()), 3),
    }


def segment(trade_df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = [{"group": str(k), **_group_stats(g)} for k, g in trade_df.groupby(col)]
    return pd.DataFrame(rows)


def orb_range_buckets(trade_df: pd.DataFrame) -> pd.DataFrame:
    df = trade_df.copy()
    df["range_bucket"] = pd.cut(
        df["orb_range"],
        bins=[0, 8, 12, 18, float("inf")],
        labels=["<8", "8-12", "12-18", "18+"],
    )
    return segment(df, "range_bucket")


def rolling_expectancy(trade_df: pd.DataFrame, window: int = 10) -> pd.Series:
    return trade_df["r_multiple"].rolling(window, min_periods=window).mean()


# ─── distribution & tail risk ─────────────────────────────────────────────────

def distribution_stats(r: pd.Series) -> dict:
    q05 = float(r.quantile(0.05))
    cvar = float(r[r <= q05].mean()) if (r <= q05).any() else q05
    return {
        "mean_r": round(float(r.mean()), 4),
        "median_r": round(float(r.median()), 4),
        "std_r": round(float(r.std()), 4),
        "skewness": round(float(r.skew()), 4),
        "excess_kurtosis": round(float(r.kurt()), 4),
        "var_95": round(q05, 4),
        "cvar_95": round(cvar, 4),
        "worst_trade_r": round(float(r.min()), 4),
        "best_trade_r": round(float(r.max()), 4),
    }


def consecutive_loss_stats(r: pd.Series) -> dict:
    streaks, cur = [], 0
    for v in r:
        if v <= 0:
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    return {
        "max_consec_losses": int(max(streaks)) if streaks else 0,
        "avg_consec_losses": round(float(np.mean(streaks)), 2) if streaks else 0,
        "total_loss_streaks": len(streaks),
    }


# ─── statistical tests ────────────────────────────────────────────────────────

def statistical_tests(r: pd.Series, tp_r: float = 2.0) -> dict:
    n = len(r)
    wins = int((r > 0).sum())
    be_wr = 1 / (1 + tp_r)

    # t-test: H0: mean R = 0
    t_stat, p_ttest = sp_stats.ttest_1samp(r.values, popmean=0)

    # Binomial test: H0: win_rate = breakeven
    binom_result = sp_stats.binomtest(wins, n, p=be_wr, alternative="greater")
    p_binom = float(binom_result.pvalue)

    # Bootstrap 95% CI on expectancy
    np.random.seed(42)
    boot = [np.random.choice(r.values, size=n, replace=True).mean() for _ in range(2000)]
    ci_low = round(float(np.percentile(boot, 2.5)), 4)
    ci_high = round(float(np.percentile(boot, 97.5)), 4)

    return {
        "n_trades": n,
        "mean_r": round(float(r.mean()), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value_ttest": round(float(p_ttest), 4),
        "edge_significant_p05": bool(p_ttest < 0.05),
        "observed_win_rate": round(float(wins / n), 4),
        "breakeven_win_rate": round(be_wr, 4),
        "p_value_binomial_gt_breakeven": round(p_binom, 4),
        "win_rate_significant_p05": bool(p_binom < 0.05),
        "expectancy_ci95_low": ci_low,
        "expectancy_ci95_high": ci_high,
        "ci_contains_zero": bool(ci_low <= 0 <= ci_high),
    }


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def monte_carlo(
    trade_df: pd.DataFrame,
    n_sims: int = 1000,
    initial_equity: float = 10000.0,
    risk_per_trade: float = 100.0,
) -> pd.DataFrame:
    r_vals = trade_df["r_multiple"].values
    n = len(r_vals)
    np.random.seed(42)

    rows = []
    for i in range(n_sims):
        sampled = np.random.choice(r_vals, size=n, replace=True)
        pnl = sampled * risk_per_trade
        eq = np.concatenate([[initial_equity], initial_equity + np.cumsum(pnl)])
        eq_s = pd.Series(eq)
        peak = eq_s.cummax()
        dd = (eq_s - peak).min()
        rows.append({
            "sim": i,
            "final_equity": round(float(eq_s.iloc[-1]), 2),
            "total_return_pct": round((float(eq_s.iloc[-1]) - initial_equity) / initial_equity * 100, 2),
            "max_drawdown": round(float(dd), 2),
            "max_drawdown_pct": round(float(dd) / initial_equity * 100, 2),
            "expectancy_r": round(float(sampled.mean()), 4),
            "win_rate_pct": round(float((sampled > 0).mean() * 100), 2),
            "sharpe": round(float(sampled.mean() / sampled.std()) if sampled.std() > 0 else 0, 4),
        })

    return pd.DataFrame(rows)


# ─── charts ───────────────────────────────────────────────────────────────────

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved chart: %s", path)


def plot_equity_drawdown(curve_df: pd.DataFrame, out_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    trades = range(len(curve_df))

    ax1.plot(trades, curve_df["equity"], color="#2ecc71", linewidth=2)
    ax1.axhline(curve_df["equity"].iloc[0], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_title("Equity Curve (by trade)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.25)

    ax2.fill_between(trades, curve_df["drawdown_pct"], 0, color="#e74c3c", alpha=0.65)
    ax2.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Trade #")
    ax2.grid(alpha=0.25)

    _savefig(f"{out_dir}/equity_drawdown.png")


def plot_monthly_heatmap(trade_df: pd.DataFrame, out_dir: str) -> None:
    df = trade_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    df["month"] = df["exit_time"].dt.month
    pivot = df.groupby(["year", "month"])["r_multiple"].sum().unstack()
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    nrows = len(pivot)
    fig, ax = plt.subplots(figsize=(14, max(2.5, nrows * 1.4)))

    data = pivot.values.astype(float)
    finite = data[np.isfinite(data)]
    vmax = max(abs(finite).max(), 0.1) if len(finite) else 1.0
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Monthly P&L Heatmap (Total R per month)", fontsize=13, fontweight="bold")

    for i in range(nrows):
        for j in range(12):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}R", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > vmax * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Total R")
    _savefig(f"{out_dir}/monthly_heatmap.png")


def plot_regime_breakdown(trade_df: pd.DataFrame, out_dir: str) -> None:
    reg = trade_df.groupby("regime").agg(
        total_r=("r_multiple", "sum"),
        win_rate=("r_multiple", lambda x: (x > 0).mean() * 100),
        trades=("r_multiple", "count"),
    ).reset_index()

    palette = {"low": "#e74c3c", "normal": "#f39c12", "high": "#3498db"}
    colors = [palette.get(r, "#95a5a6") for r in reg["regime"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(reg["regime"], reg["total_r"], color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Total R by Volatility Regime", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total R")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(reg["regime"], reg["win_rate"], color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(BREAKEVEN_WIN_RATE * 100, color="red", linestyle="--", linewidth=1.5, label="Breakeven 33.3%")
    ax2.set_title("Win Rate % by Volatility Regime", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Win Rate %")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    _savefig(f"{out_dir}/regime_breakdown.png")


def plot_monte_carlo(mc_df: pd.DataFrame, actual_final: float, initial: float, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(mc_df["final_equity"], bins=60, color="#3498db", alpha=0.72, edgecolor="white")
    ax.axvline(actual_final, color="#e74c3c", linewidth=2.5, label=f"Actual ${actual_final:,.0f}")
    ax.axvline(initial, color="gray", linewidth=1.5, linestyle="--", label=f"Initial ${initial:,.0f}")
    ax.axvline(mc_df["final_equity"].quantile(0.05), color="#e67e22", linewidth=1.5, linestyle=":", label="5th pct")
    ax.axvline(mc_df["final_equity"].quantile(0.95), color="#2ecc71", linewidth=1.5, linestyle=":", label="95th pct")
    ax.set_title("Monte Carlo: Final Equity Distribution (1,000 bootstrap simulations)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Final Equity ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.25)
    _savefig(f"{out_dir}/monte_carlo.png")


# ─── report generation ────────────────────────────────────────────────────────

def generate_summary_md(
    tests: dict,
    mc_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    dist: dict,
    consec: dict,
    curve_metrics: dict,
    monthly_df: pd.DataFrame,
    range_df: pd.DataFrame,
    dir_df: pd.DataFrame,
    n_sessions: int,
) -> str:
    mc_pct_profitable = round((mc_df["final_equity"] > mc_df["final_equity"].iloc[0].item() if False
                               else (mc_df["final_equity"] > mc_df["final_equity"].quantile(0))).mean() * 100, 1)
    mc_positive = round((mc_df["final_equity"] > mc_df["final_equity"].median()).mean() * 100, 1)
    mc_profitable_pct = round((mc_df["total_return_pct"] > 0).mean() * 100, 1)
    mc_ruin_pct = round((mc_df["max_drawdown_pct"] < -20).mean() * 100, 1)

    # Q1: statistically robust?
    sig = tests["edge_significant_p05"]
    q1 = (
        f"**No** — with {tests['n_trades']} trades, the t-test returns p={tests['p_value_ttest']:.3f} "
        f"(threshold p<0.05). The edge is not yet statistically significant. "
        f"The 95% bootstrap CI on expectancy is [{tests['expectancy_ci95_low']}R, {tests['expectancy_ci95_high']}R], "
        f"which {'does NOT contain' if not tests['ci_contains_zero'] else 'contains'} zero. "
        f"Need ~{max(30, round(30 / max(abs(tests['mean_r']), 0.01)))} trades minimum for confidence at this expectancy level."
    ) if not sig else (
        f"**Yes** — t-test p={tests['p_value_ttest']:.3f} < 0.05. Edge is statistically significant at this sample size."
    )

    # Q2: overfit?
    q2 = (
        f"**Likely yes.** Two filters were applied post-hoc (min_orb_range, shorts-only) to a dataset of "
        f"{n_sessions} sessions yielding {tests['n_trades']} trades. "
        f"Rule of thumb: need 30x trades per free parameter. "
        f"With 2 parameters and {tests['n_trades']} trades, ratio = {tests['n_trades'] // 2}x "
        f"({'adequate' if tests['n_trades'] // 2 >= 30 else 'below threshold — overfitting risk is HIGH'})."
    )

    # Q3: worst conditions?
    worst_regime = reg_df.sort_values("avg_r").iloc[0]
    q3 = (
        f"**{worst_regime['group'].capitalize()} volatility** degrades expectancy most "
        f"(avg R = {worst_regime['avg_r']:.3f}, win rate = {worst_regime['win_rate_pct']}%). "
    )
    worst_range = range_df.sort_values("avg_r").iloc[0] if not range_df.empty else None
    if worst_range is not None:
        q3 += f"ORB range bucket **{worst_range['group']}** also shows the worst avg R ({worst_range['avg_r']:.3f})."

    # Q4: edge concentrated in shorts?
    short_row = dir_df[dir_df["group"] == "short"]
    long_row = dir_df[dir_df["group"] == "long"]
    if not short_row.empty and not long_row.empty:
        s_exp = float(short_row["avg_r"].iloc[0])
        l_exp = float(long_row["avg_r"].iloc[0])
        q4 = (
            f"**{'Yes' if s_exp > 0 and l_exp <= 0 else 'Partially'}** — "
            f"shorts avg R = {s_exp:.3f}, longs avg R = {l_exp:.3f}. "
            f"{'Longs have negative expectancy; shorts carry the edge.' if l_exp < 0 and s_exp > 0 else 'Both directions show positive expectancy.' if l_exp > 0 and s_exp > 0 else 'Edge is concentrated in shorts only.'}"
        )
    else:
        q4 = "Direction breakdown not available — strategy is shorts-only as configured."

    # Q5: regime clustering?
    best_regime = reg_df.sort_values("avg_r", ascending=False).iloc[0]
    q5 = (
        f"**{'Yes' if reg_df['avg_r'].std() > 0.1 else 'No clear clustering'}** — "
        f"best regime is **{best_regime['group']}** volatility "
        f"(avg R = {best_regime['avg_r']:.3f}, {best_regime['trades']} trades). "
        f"Standard deviation of avg R across regimes = {reg_df['avg_r'].std():.3f} "
        f"({'high' if reg_df['avg_r'].std() > 0.2 else 'low'} dispersion)."
    )

    return dedent(f"""
# XAUUSD ORB Robustness Report

**Dataset:** {tests['n_trades']} trades across {n_sessions} sessions
**Strategy:** Shorts-only, min ORB range 8.0, 2R TP, 5:00-5:15 AM PT
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## Core Metrics

| Metric | Value |
|--------|-------|
| Total trades | {tests['n_trades']} |
| Win rate | {tests['observed_win_rate']*100:.1f}% (breakeven: 33.3%) |
| Expectancy | {tests['mean_r']:.4f}R |
| 95% CI on expectancy | [{tests['expectancy_ci95_low']}R, {tests['expectancy_ci95_high']}R] |
| Sharpe (annualised) | {curve_metrics.get('sharpe', 'N/A')} |
| Sortino (annualised) | {curve_metrics.get('sortino', 'N/A')} |
| Ulcer Index | {curve_metrics.get('ulcer', 'N/A')} |
| Recovery Factor | {curve_metrics.get('recovery_factor', 'N/A')} |
| Equity stability R² | {curve_metrics.get('stability_r2', 'N/A')} |
| Max drawdown % | {curve_metrics.get('max_dd_pct', 'N/A')}% |
| Max consecutive losses | {consec['max_consec_losses']} |

---

## Monte Carlo Summary (1,000 simulations)

| Percentile | Final Equity | Return % | Max DD % |
|------------|-------------|----------|----------|
| 5th  | ${mc_df['final_equity'].quantile(0.05):,.0f} | {mc_df['total_return_pct'].quantile(0.05):.1f}% | {mc_df['max_drawdown_pct'].quantile(0.95):.1f}% |
| 25th | ${mc_df['final_equity'].quantile(0.25):,.0f} | {mc_df['total_return_pct'].quantile(0.25):.1f}% | {mc_df['max_drawdown_pct'].quantile(0.75):.1f}% |
| 50th | ${mc_df['final_equity'].quantile(0.50):,.0f} | {mc_df['total_return_pct'].quantile(0.50):.1f}% | {mc_df['max_drawdown_pct'].quantile(0.50):.1f}% |
| 75th | ${mc_df['final_equity'].quantile(0.75):,.0f} | {mc_df['total_return_pct'].quantile(0.75):.1f}% | {mc_df['max_drawdown_pct'].quantile(0.25):.1f}% |
| 95th | ${mc_df['final_equity'].quantile(0.95):,.0f} | {mc_df['total_return_pct'].quantile(0.95):.1f}% | {mc_df['max_drawdown_pct'].quantile(0.05):.1f}% |

Simulations ending profitably: **{mc_profitable_pct}%**
Simulations with drawdown > 20%: **{mc_ruin_pct}%**

---

## Distribution & Tail Risk

| Stat | Value |
|------|-------|
| Skewness | {dist['skewness']} |
| Excess kurtosis | {dist['excess_kurtosis']} |
| 95% VaR | {dist['var_95']}R |
| 95% CVaR | {dist['cvar_95']}R |
| Worst trade | {dist['worst_trade_r']}R |
| Best trade | {dist['best_trade_r']}R |

---

## Key Questions

### 1. Is the edge statistically robust?

{q1}

### 2. Is the strategy likely overfit?

{q2}

### 3. Which market conditions degrade expectancy most?

{q3}

### 4. Is the edge concentrated in shorts only?

{q4}

### 5. Does performance cluster around specific volatility environments?

{q5}

### 6. What does the Monte Carlo say?

{mc_profitable_pct}% of simulations ended profitably. The median outcome is
${mc_df['final_equity'].quantile(0.50):,.0f} (from $10,000 initial). The 5th percentile
outcome is ${mc_df['final_equity'].quantile(0.05):,.0f}, indicating meaningful downside
risk even if the edge is real. Ruin risk (drawdown > 20%) appears in {mc_ruin_pct}% of sims.

---

## Recommendation

{"Collect more data before drawing conclusions. The current sample is statistically insufficient to confirm or deny edge." if not tests["edge_significant_p05"] else "Edge appears real — focus on understanding regime conditions and position sizing."}
Apply minimum 6-month dataset before live deployment. Monitor shorts-only performance in isolation — if longs are re-enabled, re-validate separately.
""").strip()


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD ORB Robustness Study")
    parser.add_argument("data", help="Path to TradingView CSV export")
    parser.add_argument("--config", default="configs/orb_config.yaml")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args()

    setup_logging(args.log_dir)
    chart_dir = f"{args.output_dir}/charts"
    Path(chart_dir).mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log = logging.getLogger(__name__)
    log.info("Loading data: %s", args.data)
    df = load_and_prepare(args.data, config)
    log.info("Loaded %d candles", len(df))

    log.info("Running ORB strategy (exact config — no changes)")
    trades = run_orb(df, config)
    n_sessions = len(set(df.index.date))
    log.info("%d trades across %d sessions", len(trades), n_sessions)

    if not trades:
        log.error("No trades generated — cannot produce robustness report.")
        sys.exit(1)

    risk = config["risk_per_trade"]
    initial = config["initial_equity"]
    trade_df = build_trade_df(trades, risk)
    curve_df = build_equity_curve(trade_df, initial)

    # Annualised trade frequency
    days = (df.index[-1] - df.index[0]).days
    trades_per_year = len(trades) / max(days, 1) * 252

    r = trade_df["r_multiple"]
    equity = curve_df["equity"]

    # ── risk metrics ──
    curve_metrics = {
        "sharpe": round(sharpe_ratio(r, trades_per_year), 3),
        "sortino": round(sortino_ratio(r, trades_per_year), 3),
        "ulcer": round(ulcer_index(equity), 3),
        "recovery_factor": recovery_factor(equity, initial),
        "stability_r2": stability_r2(equity),
        "max_dd_pct": round(float(curve_df["drawdown_pct"].min()), 2),
    }
    log.info("Risk metrics: %s", curve_metrics)

    # ── breakdowns ──
    monthly_df = monthly_performance(trade_df)
    yearly_df = yearly_performance(trade_df)
    regime_df = segment(trade_df, "regime")
    dir_df = segment(trade_df, "direction")
    range_df = orb_range_buckets(trade_df)
    exit_df = segment(trade_df, "exit_reason")
    roll_exp = rolling_expectancy(trade_df, window=min(10, len(trade_df) // 2))

    dist = distribution_stats(r)
    consec = consecutive_loss_stats(r)
    tests = statistical_tests(r, tp_r=config["take_profit_r"])
    log.info("Statistical tests: %s", tests)

    # ── Monte Carlo ──
    log.info("Running Monte Carlo (%d simulations)...", args.n_sims)
    mc_df = monte_carlo(trade_df, n_sims=args.n_sims, initial_equity=initial, risk_per_trade=risk)

    # ── charts ──
    log.info("Generating charts...")
    plot_equity_drawdown(curve_df, chart_dir)
    plot_monthly_heatmap(trade_df, chart_dir)
    plot_regime_breakdown(trade_df, chart_dir)
    plot_monte_carlo(mc_df, float(equity.iloc[-1]), initial, chart_dir)

    # ── save outputs ──
    mc_df.to_csv(f"{args.output_dir}/monte_carlo_results.csv", index=False)
    log.info("Saved monte_carlo_results.csv")

    # Flat robustness report
    rows = []
    for k, v in {**curve_metrics, **dist, **consec, **tests}.items():
        rows.append({"metric": k, "value": v})
    for _, row in monthly_df.iterrows():
        for col in monthly_df.columns:
            if col != "month":
                rows.append({"metric": f"monthly.{row['month']}.{col}", "value": row[col]})
    for _, row in regime_df.iterrows():
        for col in regime_df.columns:
            if col != "group":
                rows.append({"metric": f"regime.{row['group']}.{col}", "value": row[col]})
    for _, row in range_df.iterrows():
        for col in range_df.columns:
            if col != "group":
                rows.append({"metric": f"orb_range.{row['group']}.{col}", "value": row[col]})
    for _, row in dir_df.iterrows():
        for col in dir_df.columns:
            if col != "group":
                rows.append({"metric": f"direction.{row['group']}.{col}", "value": row[col]})

    pd.DataFrame(rows).to_csv(f"{args.output_dir}/robustness_report.csv", index=False)
    log.info("Saved robustness_report.csv")

    summary_md = generate_summary_md(
        tests=tests,
        mc_df=mc_df,
        reg_df=regime_df,
        dist=dist,
        consec=consec,
        curve_metrics=curve_metrics,
        monthly_df=monthly_df,
        range_df=range_df,
        dir_df=dir_df,
        n_sessions=n_sessions,
    )
    Path(f"{args.output_dir}/robustness_summary.md").write_text(summary_md, encoding="utf-8")
    log.info("Saved robustness_summary.md")

    # Console summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS STUDY COMPLETE")
    print("=" * 60)
    for k, v in curve_metrics.items():
        print(f"  {k:<30} {v}")
    print(f"  {'MC profitable sims':<30} {round((mc_df['total_return_pct'] > 0).mean() * 100, 1)}%")
    print(f"  {'Edge significant p<0.05':<30} {tests['edge_significant_p05']}")
    print(f"  {'95% CI contains zero':<30} {tests['ci_contains_zero']}")
    print("=" * 60)
    print(f"\nCharts saved to:  {chart_dir}/")
    print(f"Reports saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
