import os, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FIREBASE_WEB_API_KEY")
BASE = "https://identitytoolkit.googleapis.com/v1"
SECURE = "https://securetoken.googleapis.com/v1"

def signup(email: str, password: str):
    url = f"{BASE}/accounts:signUp?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload); r.raise_for_status()
    return r.json()  # idToken, refreshToken, localId

def signin(email: str, password: str):
    url = f"{BASE}/accounts:signInWithPassword?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload); r.raise_for_status()
    return r.json()

def send_password_reset(email: str):
    url = f"{BASE}/accounts:sendOobCode?key={API_KEY}"
    payload = {"requestType": "PASSWORD_RESET", "email": email}
    r = requests.post(url, json=payload); r.raise_for_status()
    return True


def update_password(id_token: str, new_password: str):
    url = f"{BASE}/accounts:update?key={API_KEY}"
    payload = {"idToken": id_token, "password": new_password, "returnSecureToken": True}
    r = requests.post(url, json=payload); r.raise_for_status()
    return r.json()

def refresh_id_token(refresh_token: str):
    url = f"{SECURE}/token?key={API_KEY}"
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    r = requests.post(url, data=data); r.raise_for_status()
    return r.json()
