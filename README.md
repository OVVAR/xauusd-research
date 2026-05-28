# xauusd-research

XAU/USD trading research and utilities.

## Contents

| File | Description |
|------|-------------|
| `trade_bot.py` | Price fetcher and trade execution loop |
| `utils.py` | Password hashing, config loading, webhook delivery |

## Setup

```bash
pip install bcrypt requests
```

Set required environment variables:

```bash
export SECRET_KEY=your-secret-key
export OPENAI_API_KEY=your-openai-key  # used by the pre-commit review hook
```

## Pre-commit review

Every commit is automatically reviewed by GPT-4o before it lands. Reviews are saved to `reviews/`.

To run a manual review on any file:

```bash
OPENAI_API_KEY=your-key node .git/hooks/pre-commit
```

## Usage

```python
from trade_bot import execute_trade
from utils import load_config, send_webhook

config = load_config("config.json")
total = execute_trade("XAUUSD", 100)
send_webhook(config["webhook_url"], f"Trade: {total}")
```
