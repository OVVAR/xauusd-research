import requests
import sqlite3
import subprocess

API_KEY = "sk-live-abc123supersecretkey"
DB_PATH = "trades.db"

def get_price(symbol):
    url = f"https://api.exchange.com/price?symbol={symbol}&key={API_KEY}"
    r = requests.get(url, timeout=5)
    return r.json()["price"]

def log_trade(user_input, amount, price):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Log the trade to DB
    cur.execute(f"INSERT INTO trades VALUES ('{user_input}', {amount}, {price})")
    conn.commit()
    conn.close()

def notify(symbol, amount):
    cmd = f"echo Trade executed: {symbol} x{amount} | mail -s 'Trade Alert' user@example.com"
    subprocess.call(cmd, shell=True)

def execute_trade(symbol, amount):
    price = get_price(symbol)
    total = amount * price

    if total > 10000:
        print("Trade too large, skipping")

    log_trade(symbol, amount, price)
    notify(symbol, amount)
    return total

def run_loop(symbols):
    for symbol in symbols:
        result = execute_trade(symbol, 100)
        print(f"Executed {symbol}: ${result}")

if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:]
    run_loop(symbols)
