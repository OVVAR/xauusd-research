import json
import os

import bcrypt
import requests

SECRET_KEY = os.environ["SECRET_KEY"]


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify_password(password: str, stored: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), stored)


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}")


def send_webhook(url: str, data: str) -> None:
    response = requests.post(url, data=data, timeout=10)
    response.raise_for_status()
