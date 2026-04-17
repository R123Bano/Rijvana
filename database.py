"""
database.py — SQLite Database Layer for News Recommender

Tables:
  - users: user profiles and preferences
  - click_history: article click tracking per user
  - sessions: session context (mood, time)
  - feedback: explicit user feedback on articles
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_recommender.db")


def get_connection():
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT,
            preferred_categories TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_mind_user INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS click_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            news_id TEXT NOT NULL,
            news_title TEXT,
            news_category TEXT,
            clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dwell_time_seconds REAL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            mood TEXT,
            mood_categories TEXT DEFAULT '[]',
            time_context TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            news_id TEXT NOT NULL,
            feedback_type TEXT CHECK(feedback_type IN ('like', 'dislike', 'not_interested', 'bookmark')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()


# ─── User Operations ──────────────────────────────────────────────────────────

def create_user(user_id: str, display_name: str = "", preferred_categories: list = None, is_mind_user: bool = False):
    """Create or update a user."""
    conn = get_connection()
    c = conn.cursor()
    cats = json.dumps(preferred_categories or [])
    c.execute(
        """INSERT OR REPLACE INTO users (user_id, display_name, preferred_categories, created_at, is_mind_user)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, display_name, cats, datetime.now(), int(is_mind_user))
    )
    conn.commit()
    conn.close()


def get_user(user_id: str) -> Optional[Dict]:
    """Get a user by ID."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_all_users() -> List[Dict]:
    """Get all users."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT user_id, display_name, is_mind_user FROM users ORDER BY user_id")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def update_user_preferences(user_id: str, preferred_categories: list):
    """Update user's preferred categories."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET preferred_categories = ? WHERE user_id = ?",
        (json.dumps(preferred_categories), user_id)
    )
    conn.commit()
    conn.close()


# ─── Click History Operations ──────────────────────────────────────────────────

def add_click(user_id: str, news_id: str, news_title: str = "", news_category: str = "", dwell_time: float = 0):
    """Record a user click on a news article."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """INSERT INTO click_history (user_id, news_id, news_title, news_category, clicked_at, dwell_time_seconds)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, news_id, news_title, news_category, datetime.now(), dwell_time)
    )
    conn.commit()
    conn.close()


def get_click_history(user_id: str, limit: int = 100) -> List[Dict]:
    """Get a user's recent click history."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """SELECT news_id, news_title, news_category, clicked_at, dwell_time_seconds
           FROM click_history WHERE user_id = ? ORDER BY clicked_at DESC LIMIT ?""",
        (user_id, limit)
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_click_category_counts(user_id: str) -> Dict[str, int]:
    """Get category distribution for a user's clicks."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """SELECT news_category, COUNT(*) as cnt
           FROM click_history WHERE user_id = ? AND news_category != ''
           GROUP BY news_category ORDER BY cnt DESC""",
        (user_id,)
    )
    result = {row["news_category"]: row["cnt"] for row in c.fetchall()}
    conn.close()
    return result


# ─── Session Operations ────────────────────────────────────────────────────────

def create_session(user_id: str, mood: str = "", mood_categories: list = None, time_context: str = ""):
    """Create a new session."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """INSERT INTO sessions (user_id, mood, mood_categories, time_context, started_at)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, mood, json.dumps(mood_categories or []), time_context, datetime.now())
    )
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_recent_sessions(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent sessions for a user."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ?""",
        (user_id, limit)
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ─── Feedback Operations ──────────────────────────────────────────────────────

def add_feedback(user_id: str, news_id: str, feedback_type: str):
    """Add user feedback on an article."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """INSERT INTO feedback (user_id, news_id, feedback_type, created_at)
           VALUES (?, ?, ?, ?)""",
        (user_id, news_id, feedback_type, datetime.now())
    )
    conn.commit()
    conn.close()


def get_disliked_news(user_id: str) -> List[str]:
    """Get news IDs the user has disliked or marked not interested."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """SELECT DISTINCT news_id FROM feedback
           WHERE user_id = ? AND feedback_type IN ('dislike', 'not_interested')""",
        (user_id,)
    )
    result = [row["news_id"] for row in c.fetchall()]
    conn.close()
    return result


def get_user_stats(user_id: str) -> Dict:
    """Get aggregate stats for a user."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as cnt FROM click_history WHERE user_id = ?", (user_id,))
    total_clicks = c.fetchone()["cnt"]

    c.execute("SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ?", (user_id,))
    total_sessions = c.fetchone()["cnt"]

    c.execute(
        """SELECT COUNT(*) as cnt FROM feedback
           WHERE user_id = ? AND feedback_type = 'like'""",
        (user_id,)
    )
    total_likes = c.fetchone()["cnt"]

    conn.close()
    return {
        "total_clicks": total_clicks,
        "total_sessions": total_sessions,
        "total_likes": total_likes,
    }


# Initialize on import
init_db()
