"""
firebase_config.py — Authentication & Database Layer for NewsLens

Dual-mode system:
  - Firebase mode: Google Sign-In + Firestore (production)
  - Local mode: Simple auth + SQLite (development/fallback)

The app automatically detects which mode to use based on whether
Firebase credentials are present in .streamlit/secrets.toml
"""

import os
import json
import sqlite3
import hashlib
import secrets as py_secrets
from datetime import datetime
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "newslens_users.db")

# ═══════════════════════════════════════════════════════════════════════════════
# Firebase Detection
# ═══════════════════════════════════════════════════════════════════════════════

_firebase_available = False
_firestore_client = None

def _try_init_firebase():
    """Attempt to initialize Firebase. Returns True if successful."""
    global _firebase_available, _firestore_client
    try:
        import streamlit as st
        if "firebase" not in st.secrets:
            return False

        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_dict = dict(st.secrets["firebase_service_account"])
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)

        _firestore_client = firestore.client()
        _firebase_available = True
        return True
    except Exception:
        return False


def is_firebase_mode() -> bool:
    """Check if Firebase is configured and available."""
    global _firebase_available
    if not _firebase_available:
        _try_init_firebase()
    return _firebase_available


# ═══════════════════════════════════════════════════════════════════════════════
# Password Hashing
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with SHA-256 + salt. Returns (hash, salt)."""
    if salt is None:
        salt = py_secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return pw_hash, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored hash."""
    pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return pw_hash == stored_hash


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Local Database (Fallback Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_local_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_local_db():
    conn = _get_local_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS nl_users (
            uid TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            display_name TEXT,
            photo_url TEXT DEFAULT '',
            interests TEXT DEFAULT '[]',
            onboarded INTEGER DEFAULT 0,
            auth_provider TEXT DEFAULT 'local',
            password_hash TEXT DEFAULT '',
            password_salt TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS nl_clicks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL,
            news_id TEXT NOT NULL,
            news_title TEXT,
            news_category TEXT,
            action TEXT DEFAULT 'like',
            dwell_time REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (uid) REFERENCES nl_users(uid)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS nl_preferences (
            uid TEXT PRIMARY KEY,
            category_weights TEXT DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (uid) REFERENCES nl_users(uid)
        )
    """)

    # Migrate: add new columns if they don't exist (for existing DBs)
    try:
        c.execute("ALTER TABLE nl_users ADD COLUMN auth_provider TEXT DEFAULT 'local'")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE nl_users ADD COLUMN password_hash TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE nl_users ADD COLUMN password_salt TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


# Initialize on import
_init_local_db()


# ═══════════════════════════════════════════════════════════════════════════════
# Unified API — Works with Firebase or SQLite
# ═══════════════════════════════════════════════════════════════════════════════

def create_or_update_user(uid: str, email: str, display_name: str, photo_url: str = "",
                          auth_provider: str = "local") -> Dict:
    """Create or update a user after authentication."""
    user_data = {
        "uid": uid,
        "email": email,
        "display_name": display_name,
        "photo_url": photo_url,
        "interests": [],
        "onboarded": False,
        "auth_provider": auth_provider,
        "last_login": datetime.now().isoformat(),
    }

    if is_firebase_mode():
        doc_ref = _firestore_client.collection("users").document(uid)
        doc = doc_ref.get()
        if doc.exists:
            # Preserve existing interests and onboarded status
            existing = doc.to_dict()
            user_data["interests"] = existing.get("interests", [])
            user_data["onboarded"] = existing.get("onboarded", False)
            doc_ref.update({"last_login": user_data["last_login"]})
        else:
            user_data["created_at"] = datetime.now().isoformat()
            doc_ref.set(user_data)
    else:
        conn = _get_local_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM nl_users WHERE uid = ?", (uid,))
        existing = c.fetchone()

        if existing:
            user_data["interests"] = json.loads(existing["interests"])
            user_data["onboarded"] = bool(existing["onboarded"])
            c.execute("UPDATE nl_users SET last_login = ? WHERE uid = ?",
                      (datetime.now(), uid))
        else:
            c.execute("""INSERT INTO nl_users (uid, email, display_name, photo_url, interests, onboarded, auth_provider, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (uid, email, display_name, photo_url, "[]", 0, auth_provider, datetime.now()))
        conn.commit()
        conn.close()

    return user_data


def get_user(uid: str) -> Optional[Dict]:
    """Get user data by UID."""
    if is_firebase_mode():
        doc = _firestore_client.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict()
        return None
    else:
        conn = _get_local_conn()
        c = conn.cursor()
        c.execute("SELECT * FROM nl_users WHERE uid = ?", (uid,))
        row = c.fetchone()
        conn.close()
        if row:
            data = dict(row)
            data["interests"] = json.loads(data["interests"])
            data["onboarded"] = bool(data["onboarded"])
            return data
        return None


def save_user_interests(uid: str, interests: List[str]):
    """Save user's selected interests (onboarding step)."""
    if is_firebase_mode():
        _firestore_client.collection("users").document(uid).update({
            "interests": interests,
            "onboarded": True,
        })
    else:
        conn = _get_local_conn()
        c = conn.cursor()
        c.execute("UPDATE nl_users SET interests = ?, onboarded = 1 WHERE uid = ?",
                  (json.dumps(interests), uid))
        conn.commit()
        conn.close()


def save_click_event(uid: str, news_id: str, title: str, category: str,
                     action: str = "like", dwell_time: float = 0):
    """Record a user interaction with an article."""
    event = {
        "uid": uid,
        "news_id": news_id,
        "news_title": title,
        "news_category": category,
        "action": action,
        "dwell_time": dwell_time,
        "created_at": datetime.now().isoformat(),
    }

    if is_firebase_mode():
        _firestore_client.collection("clicks").add(event)
    else:
        conn = _get_local_conn()
        c = conn.cursor()
        c.execute("""INSERT INTO nl_clicks (uid, news_id, news_title, news_category, action, dwell_time, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (uid, news_id, title, category, action, dwell_time, datetime.now()))
        conn.commit()
        conn.close()


def get_user_click_history(uid: str, limit: int = 200) -> List[Dict]:
    """Get all click history for a user (for RL reconstruction)."""
    if is_firebase_mode():
        docs = (_firestore_client.collection("clicks")
                .where("uid", "==", uid)
                .order_by("created_at", direction="DESCENDING")
                .limit(limit)
                .stream())
        return [doc.to_dict() for doc in docs]
    else:
        conn = _get_local_conn()
        c = conn.cursor()
        c.execute("""SELECT * FROM nl_clicks WHERE uid = ? ORDER BY created_at DESC LIMIT ?""",
                  (uid, limit))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows


def get_user_disliked(uid: str) -> List[str]:
    """Get news IDs the user has skipped/disliked."""
    if is_firebase_mode():
        docs = (_firestore_client.collection("clicks")
                .where("uid", "==", uid)
                .where("action", "in", ["skip", "not_interested"])
                .stream())
        return list(set(doc.to_dict()["news_id"] for doc in docs))
    else:
        conn = _get_local_conn()
        c = conn.cursor()
        c.execute("""SELECT DISTINCT news_id FROM nl_clicks
                     WHERE uid = ? AND action IN ('skip', 'not_interested')""",
                  (uid,))
        result = [row["news_id"] for row in c.fetchall()]
        conn.close()
        return result


def local_login(email: str, password: str) -> Optional[Dict]:
    """Local login with password verification."""
    conn = _get_local_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM nl_users WHERE email = ?", (email,))
    existing = c.fetchone()
    conn.close()

    if existing:
        stored_hash = existing["password_hash"] if existing["password_hash"] else ""
        stored_salt = existing["password_salt"] if existing["password_salt"] else ""

        # Legacy users without password hash — accept any password (migrate them)
        if not stored_hash:
            # Migrate: set password hash now
            pw_hash, salt = _hash_password(password)
            conn2 = _get_local_conn()
            c2 = conn2.cursor()
            c2.execute("UPDATE nl_users SET password_hash = ?, password_salt = ? WHERE uid = ?",
                       (pw_hash, salt, existing["uid"]))
            conn2.commit()
            conn2.close()
            return create_or_update_user(existing["uid"], email, existing["display_name"])

        # Verify password
        if _verify_password(password, stored_hash, stored_salt):
            return create_or_update_user(existing["uid"], email, existing["display_name"])
        else:
            return None  # Wrong password
    else:
        return None  # User not found


def local_signup(name: str, email: str, password: str) -> Optional[Dict]:
    """Local signup with password hashing."""
    uid = hashlib.sha256(email.encode()).hexdigest()[:28]

    conn = _get_local_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM nl_users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return None  # Already exists

    # Hash the password
    pw_hash, salt = _hash_password(password)

    c.execute("""INSERT INTO nl_users (uid, email, display_name, photo_url, interests, onboarded, auth_provider, password_hash, password_salt, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (uid, email, name, "", "[]", 0, "local", pw_hash, salt, datetime.now()))
    conn.commit()
    conn.close()

    return {
        "uid": uid,
        "email": email,
        "display_name": name,
        "photo_url": "",
        "interests": [],
        "onboarded": False,
        "auth_provider": "local",
    }


def google_login(email: str, name: str, photo_url: str = "") -> Dict:
    """Handle Google OAuth login — create or retrieve user."""
    uid = hashlib.sha256(email.encode()).hexdigest()[:28]
    return create_or_update_user(uid, email, name, photo_url, auth_provider="google")
