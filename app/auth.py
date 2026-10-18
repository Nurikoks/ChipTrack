import hashlib
import hmac
import re
import secrets
import time
import uuid
from fastapi import HTTPException, Request
from . import store

COOKIE = "chiptrick_session"


def hash_password(password):
    salt = secrets.token_hex(16)
    value = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()
    return salt + ":" + value


def verify_password(password, encoded):
    salt, expected = encoded.split(":")
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()
    return hmac.compare_digest(actual, expected)


def create_user(username, password, role="user"):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,40}", username):
        raise ValueError("Логин: 3–40 латинских букв, цифр, точек, дефисов или подчёркиваний")
    if len(password) < 10 or len(password) > 200:
        raise ValueError("Пароль должен содержать от 10 до 200 символов")
    uid = uuid.uuid4().hex
    store.execute("INSERT INTO users VALUES(?,?,?,?,?)", (uid, username, hash_password(password), role, time.time()))
    return {"id": uid, "username": username, "role": role}


def login(username, password):
    u = store.one("SELECT * FROM users WHERE username=?", (username.strip().lower(),))
    # Use the same KDF cost for unknown names.
    valid = verify_password(password, u["password"] if u else "00" * 16 + ":" + "00" * 32)
    if not u or not valid or u['username']=='jury-api':
        raise HTTPException(401, "Неверный логин или пароль")
    token = secrets.token_urlsafe(32)
    store.execute("INSERT INTO sessions VALUES(?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), u["id"], time.time() + 86400))
    return token


def current_user(request: Request):
    token = request.cookies.get(COOKIE, "")
    u = store.one("SELECT u.id,u.username,u.role FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=? AND s.expires>?",
                  (hashlib.sha256(token.encode()).hexdigest(), time.time()))
    if not u:
        raise HTTPException(401, "Войдите в аккаунт")
    return u


def admin(request: Request):
    u = current_user(request)
    if u["role"] != "admin":
        raise HTTPException(403, "Требуются права администратора")
    return u
