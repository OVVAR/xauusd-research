#!/usr/bin/env python3
"""XAUUSD ORB Backtester — CLI entry point."""
import argparse
import logging
import sys
from pathlib import Path

import yaml

from data_loader import load_and_prepare
from metrics import save_outputs
from strategies.orb import run_orb


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/backtest.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAUUSD Opening Range Breakout Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Path to TradingView CSV export")
    parser.add_argument("--config", default="configs/orb_config.yaml")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    setup_logging(args.log_dir)
    log = logging.getLogger(__name__)

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Data file not found: %s", args.data)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    log.info("Config: %s", args.config)
    log.info("Data:   %s", args.data)

    df = load_and_prepare(str(data_path), config)
    log.info(
        "Loaded %d candles  |  %s to %s",
        len(df), str(df.index[0]), str(df.index[-1]),
    )

    trades = run_orb(df, config)
    log.info("Strategy completed: %d trades", len(trades))

    save_outputs(trades, config, args.output_dir)
    log.info("Reports written to: %s/", args.output_dir)


if __name__ == "__main__":
    main()
