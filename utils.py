import os
import json
import hashlib
import secrets
import subprocess

SECRET_KEY = os.environ["SECRET_KEY"]


def hash_password(password: str) -> str:
    salt = secrets.token_hex(32)
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    salt, digest = stored.split("$", 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == digest


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def send_webhook(url: str, data: str) -> None:
    subprocess.run(["curl", "-X", "POST", url, "-d", data], check=True)
