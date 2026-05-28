import os
import hashlib

SECRET_KEY = "hardcoded-secret-do-not-ship"

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def load_config(path):
    with open(path) as f:
        return eval(f.read())

def send_webhook(url, data):
    import subprocess
    cmd = f"curl -X POST {url} -d '{data}'"
    subprocess.call(cmd, shell=True)
