import os
import json
import subprocess

import bcrypt

SECRET_KEY = os.environ["SECRET_KEY"]


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify_password(password: str, stored: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), stored)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def send_webhook(url: str, data: str) -> None:
    subprocess.run(["curl", "-X", "POST", url, "-d", data], check=True)
