"""
app.py — NewsLens: Hyper-Personalized News Platform

4-Stage Flow:
  1. Login     — authenticate via Google or local credentials
  2. Welcome   — greet the user with cinematic animation
  3. Onboarding — select 3 interests from MIND dataset labels
  4. Dashboard  — personalized news feed + analytics

Apple-inspired design. RL-based personalization.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
import streamlit.components.v1 as components

# ─── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from recommendation_engine import get_recommender, CATEGORIES
from mood_handler import (
    infer_mood_and_categories, get_full_mood_analysis,
    EMOJI_MOODS, get_emoji_mood, get_time_period
)
from database import (
    init_db, create_user, get_user, get_all_users,
    update_user_preferences, add_click, get_click_history,
    get_click_category_counts, create_session, get_recent_sessions,
    add_feedback, get_disliked_news, get_user_stats
)
from firebase_config import (
    create_or_update_user as fb_create_user,
    get_user as fb_get_user,
    save_user_interests,
    save_click_event,
    get_user_click_history,
    get_user_disliked,
    local_login,
    local_signup,
    google_login,
    is_firebase_mode,
)
from news_api import (
    get_cached_live_news,
    is_news_api_configured,
    fetch_top_headlines,
    NEWSAPI_TO_MIND,
)

# ─── Category Metadata ───────────────────────────────────────────────────────
CATEGORY_INFO = {
    "news":          {"emoji": "📰", "label": "News",           "color": "#5ac8fa"},
    "sports":        {"emoji": "⚽", "label": "Sports",         "color": "#30d158"},
    "finance":       {"emoji": "💰", "label": "Finance",        "color": "#ff9f0a"},
    "foodanddrink":  {"emoji": "🍕", "label": "Food & Drink",   "color": "#ff9f0a"},
    "lifestyle":     {"emoji": "✨", "label": "Lifestyle",       "color": "#bf5af2"},
    "travel":        {"emoji": "✈️", "label": "Travel",          "color": "#64d2ff"},
    "video":         {"emoji": "🎬", "label": "Video",           "color": "#ff375f"},
    "weather":       {"emoji": "🌤️", "label": "Weather",        "color": "#64d2ff"},
    "health":        {"emoji": "❤️", "label": "Health",          "color": "#ff453a"},
    "autos":         {"emoji": "🚗", "label": "Autos",           "color": "#8e8e93"},
    "tv":            {"emoji": "📺", "label": "TV",              "color": "#5e5ce6"},
    "music":         {"emoji": "🎵", "label": "Music",           "color": "#ff375f"},
    "movies":        {"emoji": "🎥", "label": "Movies",          "color": "#ff375f"},
    "entertainment": {"emoji": "🎭", "label": "Entertainment",   "color": "#ff375f"},
    "kids":          {"emoji": "👶", "label": "Kids",             "color": "#ff9f0a"},
    "middleeast":    {"emoji": "🌍", "label": "Middle East",     "color": "#64d2ff"},
    "northamerica":  {"emoji": "🌎", "label": "North America",   "color": "#0a84ff"},
}

def get_category_emoji(cat: str) -> str:
    return CATEGORY_INFO.get(cat, {}).get("emoji", "📄")

def get_category_label(cat: str) -> str:
    return CATEGORY_INFO.get(cat, {}).get("label", cat.title())

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NewsLens",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load CSS ─────────────────────────────────────────────────────────────────
css_path = os.path.join(BASE_DIR, "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Session State Defaults ──────────────────────────────────────────────────
defaults = {
    "stage": "login",
    "auth_user": None,
    "current_user_id": None,
    "recommendations": [],
    "current_mood": "neutral",
    "mood_categories": [],
    "session_clicks": [],
    "rec_latency": 0,
    "recommendation_history": [],
    "page": "Dashboard",
    "selected_interests": [],
    "recs_loaded": False,
    "past_history_loaded": False,
    "all_history": [],
    "live_news": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Load Recommender ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_engine():
    return get_recommender()

models_dir = os.path.join(BASE_DIR, "models")
if not os.path.exists(os.path.join(models_dir, "news_dict.pkl")):
    st.error("⚠️ Models not found! Run `python3 train_model.py` first.")
    st.stop()

engine = load_engine()


def load_past_history():
    """Load user's past click history from database on login/reopen."""
    uid = st.session_state.current_user_id
    if not uid or st.session_state.past_history_loaded:
        return

    # Load from nl_clicks (firebase_config database)
    past_clicks = get_user_click_history(uid, limit=100)

    # Also load from legacy click_history table
    legacy_clicks = get_click_history(uid, limit=100)

    # Merge and deduplicate
    all_history = []
    seen = set()
    for click in past_clicks:
        nid = click.get("news_id", "")
        if nid and nid not in seen:
            seen.add(nid)
            all_history.append({
                "news_id": nid,
                "title": click.get("news_title", click.get("title", "")),
                "category": click.get("news_category", click.get("category", "")),
                "action": click.get("action", "like"),
                "dwell_time": click.get("dwell_time", 0),
                "timestamp": click.get("created_at", ""),
            })
    for click in legacy_clicks:
        nid = click.get("news_id", "")
        if nid and nid not in seen:
            seen.add(nid)
            all_history.append({
                "news_id": nid,
                "title": click.get("news_title", ""),
                "category": click.get("news_category", ""),
                "action": "like",
                "dwell_time": click.get("dwell_time_seconds", 0),
                "timestamp": click.get("clicked_at", ""),
            })

    st.session_state.all_history = all_history

    # Rebuild RL bandit from past interactions
    for click in all_history:
        cat = click.get("category", "")
        if cat:
            action = click.get("action", "like")
            dwell = click.get("dwell_time", 0) or 0
            if action in ("like", "glance"):
                reward = min(1.0, 0.3 + (float(dwell) / 60.0)) if dwell else 0.5
                engine.bandit.update(uid, cat, reward=reward)
            elif action in ("skip", "not_interested"):
                engine.bandit.update(uid, cat, reward=0.0)

    # Create session record
    create_session(uid, mood=st.session_state.current_mood,
                   mood_categories=st.session_state.mood_categories,
                   time_context=get_time_period(datetime.now().hour))

    st.session_state.past_history_loaded = True

# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_google_client_id() -> str:
    """Get Google OAuth Client ID from environment or secrets."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    if not client_id:
        try:
            client_id = st.secrets.get("google_oauth", {}).get("client_id", "")
        except Exception:
            pass
    return client_id


def render_google_signin_button(client_id: str) -> str:
    """Render a Google Sign-In button using Google Identity Services JS API.
    Returns the credential JWT if available."""

    # Check for token in query params (returned from popup)
    params = st.query_params
    credential = params.get("credential", "")
    if credential:
        # Clear the query param
        st.query_params.clear()
        return credential

    # Render the Google Sign-In button via HTML component
    google_btn_html = f"""
    <script src="https://accounts.google.com/gsi/client" async></script>
    <div id="g_id_onload"
         data-client_id="{client_id}"
         data-context="signin"
         data-ux_mode="popup"
         data-callback="onSignIn"
         data-auto_prompt="false">
    </div>
    <div class="g_id_signin"
         data-type="standard"
         data-shape="pill"
         data-theme="filled_black"
         data-text="signin_with"
         data-size="large"
         data-logo_alignment="left"
         data-width="320">
    </div>
    <script>
    function onSignIn(response) {{
        // Redirect with credential as query param
        const url = new URL(window.parent.location.href);
        url.searchParams.set('credential', response.credential);
        window.parent.location.href = url.toString();
    }}
    </script>
    """
    components.html(google_btn_html, height=50)
    return ""


def decode_google_jwt(credential: str) -> dict:
    """Decode Google JWT to extract user info (without verification for hackathon)."""
    import base64
    try:
        # JWT has 3 parts: header.payload.signature
        parts = credential.split(".")
        if len(parts) != 3:
            return {}
        # Decode payload (add padding)
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        return {
            "email": data.get("email", ""),
            "name": data.get("name", ""),
            "picture": data.get("picture", ""),
            "sub": data.get("sub", ""),
        }
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: LOGIN
# ═══════════════════════════════════════════════════════════════════════════════

def render_login():
    """Apple-style login page with Google OAuth + email/password."""
    # Hide sidebar on login
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # Hero section
    st.markdown("""
    <div class="login-hero">
        <div class="login-hero-glow"></div>
        <div class="login-logo-mark">
            <span class="login-logo-icon">📰</span>
        </div>
        <div class="login-brand">NewsLens</div>
        <div class="login-tagline-hero">Your news. Intelligently curated.</div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Google Sign-In ───────────────────────────────────────────
    google_client_id = _get_google_client_id()

    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        if google_client_id:
            credential = render_google_signin_button(google_client_id)
            if credential:
                user_info = decode_google_jwt(credential)
                if user_info and user_info.get("email"):
                    user = google_login(
                        email=user_info["email"],
                        name=user_info.get("name", user_info["email"].split("@")[0]),
                        photo_url=user_info.get("picture", ""),
                    )
                    st.session_state.auth_user = user
                    st.session_state.current_user_id = user["uid"]
                    if user.get("onboarded"):
                        st.session_state.stage = "dashboard"
                        st.session_state.mood_categories = user.get("interests", [])
                    else:
                        st.session_state.stage = "welcome"
                    st.rerun()

            # Or divider
            st.markdown("""
            <div class="auth-divider">
                <div class="auth-divider-line"></div>
                <span class="auth-divider-text">or sign in with email</span>
                <div class="auth-divider-line"></div>
            </div>
            """, unsafe_allow_html=True)

        # ─── Email / Password ─────────────────────────────────────
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="you@example.com", key="login_email")
                password = st.text_input("Password", type="password", placeholder="Password", key="login_pass")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if submitted and email and password:
                    user = local_login(email, password)
                    if user:
                        st.session_state.auth_user = user
                        st.session_state.current_user_id = user["uid"]
                        if user.get("onboarded"):
                            st.session_state.stage = "dashboard"
                            st.session_state.mood_categories = user.get("interests", [])
                        else:
                            st.session_state.stage = "welcome"
                        st.rerun()
                    else:
                        st.error("Invalid email or password. Please try again or create an account.")

        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                name = st.text_input("Full Name", placeholder="John Doe", key="signup_name")
                email_s = st.text_input("Email", placeholder="you@example.com", key="signup_email")
                password_s = st.text_input("Password", type="password", placeholder="Create a password (min 6 chars)", key="signup_pass")
                submitted_s = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submitted_s and name and email_s and password_s:
                    if len(password_s) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        user = local_signup(name, email_s, password_s)
                        if user:
                            st.session_state.auth_user = user
                            st.session_state.current_user_id = user["uid"]
                            st.session_state.stage = "welcome"
                            st.rerun()
                        else:
                            st.error("An account with this email already exists. Try signing in.")

    # Footer
    st.markdown("""
    <div class="login-footer">
        <div class="login-footer-text">
            Powered by Reinforcement Learning · MIND Dataset · 51,282 Articles
        </div>
        <div class="login-footer-badges">
            <span class="footer-badge">🧠 RL Engine</span>
            <span class="footer-badge">🔒 Secure</span>
            <span class="footer-badge">⚡ Real-time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: WELCOME
# ═══════════════════════════════════════════════════════════════════════════════

def render_welcome():
    """Cinematic welcome screen after login."""
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    user = st.session_state.auth_user
    display_name = user.get("display_name", "Reader") if user else "Reader"
    first_name = display_name.split()[0] if display_name else "Reader"

    st.markdown(f"""
    <div class="welcome-container">
        <div class="welcome-hero-glow"></div>
        <div class="welcome-badge">WELCOME TO NEWSLENS</div>
        <div class="welcome-title">
            Hello, <span class="welcome-name">{first_name}</span>.
        </div>
        <div class="welcome-desc">
            Your personal news experience starts here. We use advanced AI and
            reinforcement learning to curate articles that matter to you — and
            it gets smarter every time you read.
        </div>
        <div class="welcome-features">
            <div class="welcome-feature">
                <div class="feature-icon">🧠</div>
                <div class="feature-text">AI-Powered</div>
            </div>
            <div class="welcome-feature">
                <div class="feature-icon">📊</div>
                <div class="feature-text">Learns from You</div>
            </div>
            <div class="welcome-feature">
                <div class="feature-icon">⚡</div>
                <div class="feature-text">Real-time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.3, 1, 1.3])
    with col2:
        if st.button("Continue", type="primary", use_container_width=True, key="welcome_continue"):
            st.session_state.stage = "onboarding"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: ONBOARDING — INTEREST SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_onboarding():
    """Interest selection screen — pick 3 categories."""
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="onboard-container">
        <div class="onboard-eyebrow">PERSONALIZATION</div>
        <div class="onboard-title">Let's personalize your news.</div>
        <div class="onboard-subtitle">Choose 3 topics you're interested in. We'll use these to start curating your feed.</div>
    </div>
    """, unsafe_allow_html=True)

    # Interest selection grid
    selected = st.session_state.selected_interests

    # Display categories in columns
    display_cats = [c for c in CATEGORIES if c in CATEGORY_INFO]
    cols_per_row = 5
    rows = [display_cats[i:i + cols_per_row] for i in range(0, len(display_cats), cols_per_row)]

    for row_cats in rows:
        cols = st.columns(cols_per_row)
        for i, cat in enumerate(row_cats):
            with cols[i]:
                info = CATEGORY_INFO[cat]
                is_selected = cat in selected

                # Styled card
                if is_selected:
                    st.markdown(f"""
                    <div class="interest-chip selected">
                        <div class="interest-chip-emoji">{info['emoji']}</div>
                        <div class="interest-chip-label" style="color: #0a84ff;">{info['label']}</div>
                        <div class="interest-chip-check">✓</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="interest-chip">
                        <div class="interest-chip-emoji">{info['emoji']}</div>
                        <div class="interest-chip-label">{info['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Toggle button
                btn_text = "Remove" if is_selected else "Select"
                if st.button(btn_text, key=f"interest_{cat}", use_container_width=True):
                    if is_selected:
                        st.session_state.selected_interests.remove(cat)
                    elif len(selected) < 3:
                        st.session_state.selected_interests.append(cat)
                    else:
                        st.toast("You can select up to 3 interests. Remove one first.", icon="⚠️")
                    st.rerun()

    # Counter and submit
    num_selected = len(st.session_state.selected_interests)
    st.markdown(f"""
    <div class="selection-status">
        <div class="selection-dots">
            <span class="sel-dot {'active' if num_selected >= 1 else ''}"></span>
            <span class="sel-dot {'active' if num_selected >= 2 else ''}"></span>
            <span class="sel-dot {'active' if num_selected >= 3 else ''}"></span>
        </div>
        <div class="selection-text">
            <span style="color: {'#0a84ff' if num_selected > 0 else '#48484a'}; font-weight: 700;">{num_selected}</span>
            <span style="color: #86868b;"> of 3 selected</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.3, 1, 1.3])
    with col2:
        if num_selected == 3:
            if st.button("Start Reading →", type="primary", use_container_width=True, key="start_reading"):
                uid = st.session_state.current_user_id
                interests = st.session_state.selected_interests

                # Save to database
                save_user_interests(uid, interests)

                # Bootstrap RL bandit with selected interests
                for cat in interests:
                    cat_lower = cat.lower()
                    engine.bandit.update(uid, cat_lower, reward=0.5)

                # Also create user in the legacy database
                user = st.session_state.auth_user
                create_user(uid, user.get("display_name", ""), is_mind_user=False)
                update_user_preferences(uid, interests)

                # Set mood categories for initial recommendations
                st.session_state.mood_categories = interests
                st.session_state.stage = "dashboard"
                st.session_state.recs_loaded = False
                st.rerun()
        else:
            st.button("Select 3 topics to continue", disabled=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def render_news_card(article: dict, show_signals: bool = False, show_actions: bool = True, idx: int = 0):
    """Render an Apple-style news card."""
    cat = article.get("category", "news")
    cat_class = cat if cat in CATEGORY_INFO else "default"
    emoji = get_category_emoji(cat)
    score = article.get("score", 0)

    signal_html = ""
    if show_signals and "signals" in article:
        for sig_name, sig_val in article["signals"].items():
            if sig_name in ("cold_start",):
                continue
            width = min(float(sig_val) * 100, 100)
            sig_class = sig_name if sig_name in ("rl", "content", "collab", "mood") else "rl"
            signal_html += f"""
            <div class="signal-bar">
                <span class="signal-label">{sig_name}</span>
                <div class="signal-track">
                    <div class="signal-fill {sig_class}" style="width: {width}%"></div>
                </div>
            </div>"""

    card_html = f"""
    <div class="news-card" style="animation-delay: {idx * 0.06}s">
        <div class="news-title">{article.get('title', 'Untitled')}</div>
        <div class="news-abstract">{article.get('abstract', '')}</div>
        <div class="news-meta">
            <span class="category-badge {cat_class}">{emoji} {cat}</span>
            {f'<span class="category-badge default">{article.get("subcategory", "")}</span>' if article.get("subcategory") else ''}
            <span class="score-badge">⚡ {score:.3f}</span>
        </div>
        {f'<div style="margin-top: 12px">{signal_html}</div>' if signal_html else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    if show_actions:
        read_key = f"reading_{article['news_id']}_{idx}"
        timer_key = f"timer_{article['news_id']}_{idx}"

        if not st.session_state.get(read_key, False):
            c1, c2, c3 = st.columns([1, 1, 1.5])
            with c1:
                if st.button("👍 Like", key=f"like_{article['news_id']}_{idx}"):
                    handle_click(article, "like")
            with c2:
                if st.button("👎 Skip", key=f"skip_{article['news_id']}_{idx}"):
                    handle_skip(article)
            with c3:
                if st.button("📖 Read", key=f"read_{article['news_id']}_{idx}"):
                    st.session_state[read_key] = True
                    st.session_state[timer_key] = time.time()
                    st.rerun()
        else:
            st.markdown("---")
            st.info(f"**{article.get('title')}**\n\n{article.get('abstract', 'No content available.')}\n\n*(Full article text)*")
            if st.button("✅ Done Reading", key=f"done_{article['news_id']}_{idx}", type="primary"):
                start = st.session_state.get(timer_key, time.time())
                dwell = time.time() - start
                st.session_state[read_key] = False

                # Show dwell time stats
                mins = int(dwell // 60)
                secs = int(dwell % 60)
                time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                if dwell > 5.0:
                    st.toast(f"📈 Deep read ({time_str}) — AI boosting {cat}!", icon="🧠")
                    st.success(f"⏱️ You spent **{time_str}** reading this article. Your preferences have been updated!")
                    handle_click(article, "like", dwell_time=dwell)
                else:
                    st.toast(f"Quick glance ({time_str}) — noted.", icon="⏱️")
                    st.info(f"⏱️ Quick read: **{time_str}**. Spend more time for stronger personalization.")
                    handle_click(article, "glance", dwell_time=dwell)
                st.rerun()


def handle_click(article: dict, feedback_type: str = "like", dwell_time: float = 5.0):
    """Handle a positive interaction — update RL + database."""
    uid = st.session_state.current_user_id
    if not uid:
        return

    # Scale reward by dwell time for deeper personalization
    if feedback_type == "like":
        reward = min(1.5, 0.5 + (dwell_time / 30.0))  # More time = higher reward
    else:
        reward = 0.3

    # Update RL model — this is what makes recommendations improve
    engine.record_click(uid, article["news_id"], reward=reward)

    # Also directly boost the category in the bandit
    cat = article.get("category", "")
    if cat:
        engine.bandit.update(uid, cat, reward=reward * 0.5)

    # Save to both databases
    add_click(uid, article["news_id"], article.get("title", ""),
              cat, dwell_time=dwell_time)
    add_feedback(uid, article["news_id"], feedback_type)
    save_click_event(uid, article["news_id"], article.get("title", ""),
                     cat, feedback_type, dwell_time)

    click_record = {
        "news_id": article["news_id"],
        "category": cat,
        "title": article.get("title", ""),
        "action": feedback_type,
        "dwell_time": dwell_time,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.session_clicks.append(click_record)
    st.session_state.all_history.insert(0, click_record)  # Add to persistent history

    # Force refresh recommendations after interaction
    st.session_state.recommendations = []

    if feedback_type == "like":
        st.toast(f"Liked: {article.get('title', '')[:50]}...", icon="👍")


def handle_skip(article: dict):
    """Handle skip — negative RL signal."""
    uid = st.session_state.current_user_id
    if not uid:
        return

    engine.record_skip(uid, article["news_id"])
    add_feedback(uid, article["news_id"], "not_interested")
    save_click_event(uid, article["news_id"], article.get("title", ""),
                     article.get("category", ""), "skip", 0)

    if "recommendations" in st.session_state:
        st.session_state.recommendations = [
            r for r in st.session_state.recommendations
            if r["news_id"] != article["news_id"]
        ]

    st.toast("Removed — we'll adjust your feed.", icon="🗑️")
    st.rerun()


def generate_recs():
    """Generate personalized recommendations."""
    uid = st.session_state.current_user_id
    if not uid:
        return []

    history = engine.user_profiles.get(uid, {}).get("history_ids", [])
    for click in st.session_state.session_clicks:
        if click["news_id"] not in history:
            history.append(click["news_id"])

    disliked = get_disliked_news(uid)
    mood_cats = st.session_state.mood_categories

    start = time.time()

    if history:
        recs = engine.recommend(
            user_id=uid,
            history_ids=history,
            mood_categories=mood_cats,
            excluded_ids=disliked,
            num_recommendations=20,
        )
    else:
        recs = engine.get_cold_start_recommendations(
            preferred_categories=mood_cats if mood_cats else None,
            num_recommendations=20,
        )

    st.session_state.rec_latency = (time.time() - start) * 1000
    st.session_state.recommendation_history.append({
        "timestamp": datetime.now().isoformat(),
        "num_recs": len(recs),
        "latency_ms": st.session_state.rec_latency,
        "mood": st.session_state.current_mood,
    })

    return recs


def render_dashboard():
    """Main dashboard with sidebar navigation."""
    # ─── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        user = st.session_state.auth_user
        display_name = user.get("display_name", "User") if user else "User"
        email = user.get("email", "") if user else ""
        initial = display_name[0].upper() if display_name else "U"
        photo_url = user.get("photo_url", "") if user else ""
        auth_provider = user.get("auth_provider", "local") if user else "local"

        # Profile photo or initial
        if photo_url:
            avatar_html = f'<img src="{photo_url}" class="user-avatar-img" referrerpolicy="no-referrer" />'
        else:
            avatar_html = f'<div class="user-avatar">{initial}</div>'

        provider_badge = "🔐 Google" if auth_provider == "google" else "📧 Email"

        st.markdown(f"""
        <div class="user-badge">
            {avatar_html}
            <div class="user-info">
                <div class="user-name">{display_name}</div>
                <div class="user-email">{email}</div>
                <div class="user-provider">{provider_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; padding: 4px 0 16px;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #f5f5f7; letter-spacing: -0.03em;">📰 NewsLens</div>
            <div style="font-size: 0.7rem; color: #48484a; letter-spacing: 0.05em; text-transform: uppercase;">Personalized for you</div>
        </div>
        """, unsafe_allow_html=True)

        nav_options = ["🏠 Dashboard", "📰 Feed", "👤 Profile", "📊 Analytics"]
        if is_news_api_configured():
            nav_options.insert(2, "🔴 Live News")
        page = st.radio(
            "Navigate",
            nav_options,
            label_visibility="collapsed",
        )
        st.session_state.page = page

        st.markdown("---")

        # Quick stats
        if st.session_state.current_user_id:
            uid = st.session_state.current_user_id
            profile = engine.get_user_profile_summary(uid)
            clicks = len(st.session_state.session_clicks)

            st.markdown(f"""
            <div style="padding: 0 4px;">
                <div style="font-size: 0.7rem; color: #48484a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Session</div>
                <div style="font-size: 0.85rem; color: #86868b; margin-bottom: 4px;">📊 {clicks} interactions</div>
                <div style="font-size: 0.85rem; color: #86868b; margin-bottom: 4px;">⚡ {st.session_state.rec_latency:.0f}ms latency</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("""
        <div style="text-align: center; padding: 16px 0; color: #48484a; font-size: 0.65rem; letter-spacing: 0.02em;">
            <div>Powered by MIND Dataset</div>
            <div>51,282 Articles · RL Engine</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Load past history from database on login ─────────────────
    load_past_history()

    # ─── Auto-load recommendations on first visit ─────────────────
    if not st.session_state.recs_loaded and not st.session_state.recommendations:
        with st.spinner("Curating your personalized feed..."):
            recs = generate_recs()
            st.session_state.recommendations = recs
            st.session_state.recs_loaded = True

    # ─── ROUTE PAGES ──────────────────────────────────────────────
    if page == "🏠 Dashboard":
        render_page_dashboard()
    elif page == "📰 Feed":
        render_page_feed()
    elif page == "🔴 Live News":
        render_page_live_news()
    elif page == "👤 Profile":
        render_page_profile()
    elif page == "📊 Analytics":
        render_page_analytics()


# ─── PAGE: DASHBOARD ─────────────────────────────────────────────────────────

def render_page_dashboard():
    user = st.session_state.auth_user
    display_name = user.get("display_name", "Reader") if user else "Reader"
    first_name = display_name.split()[0] if display_name else "Reader"

    now = datetime.now()
    hour = now.hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    st.markdown(f"""
    <div style="margin-bottom: 8px;">
        <div style="font-size: 0.8rem; color: #48484a; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px;">
            {greeting}
        </div>
        <h1 style="margin: 0; font-size: 2.5rem !important;">{first_name}'s Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    uid = st.session_state.current_user_id
    profile = engine.get_user_profile_summary(uid)
    total_clicks = profile["total_clicks"] if profile else 0
    session_clicks = len(st.session_state.session_clicks)
    user_type = "Warm User" if total_clicks >= 5 else "Cold Start"
    time_period = get_time_period(hour)

    # Context bar
    current_mood = st.session_state.current_mood
    top_cat = "news"
    if profile and profile.get("top_categories"):
        top_cat = profile["top_categories"][0][0]

    st.markdown(f"""
    <div class="context-bar">
        <div class="context-card">
            <div class="context-value">{current_mood.title()}</div>
            <div class="context-label">Mood</div>
        </div>
        <div class="context-card">
            <div class="context-value">{time_period.title()}</div>
            <div class="context-label">Time · {now.strftime('%I:%M %p')}</div>
        </div>
        <div class="context-card">
            <div class="context-value">{get_category_emoji(top_cat)} {top_cat.title()}</div>
            <div class="context-label">Top Interest</div>
        </div>
        <div class="context-card">
            <div class="context-value">{total_clicks}</div>
            <div class="context-label">Articles Read</div>
        </div>
        <div class="context-card">
            <div class="context-value">{session_clicks}</div>
            <div class="context-label">This Session</div>
        </div>
        <div class="context-card">
            <div class="context-value">{"🟢" if total_clicks >= 5 else "🔵"} {user_type}</div>
            <div class="context-label">Status</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Your Interests")
        if profile and profile.get("category_dist"):
            cat_data = profile["category_dist"]
            sorted_cats = sorted(cat_data.items(), key=lambda x: x[1], reverse=True)

            pref_html = ""
            max_val = max(v for _, v in sorted_cats) if sorted_cats else 1
            for cat, val in sorted_cats[:8]:
                pct = (val / max_val) * 100 if max_val > 0 else 0
                emoji = get_category_emoji(cat)
                pref_html += f"""
                <div class="pref-bar">
                    <span class="pref-label">{emoji} {cat}</span>
                    <div class="pref-track">
                        <div class="pref-fill" style="width: {pct}%"></div>
                    </div>
                    <span class="pref-percent">{val:.1%}</span>
                </div>"""
            st.markdown(pref_html, unsafe_allow_html=True)
        else:
            interests = st.session_state.auth_user.get("interests", []) if st.session_state.auth_user else []
            if interests:
                for cat in interests:
                    emoji = get_category_emoji(cat)
                    st.markdown(f"- {emoji} **{get_category_label(cat)}**")
            else:
                st.info("Start reading articles to build your preference profile!")

    with col2:
        st.markdown("### AI Personalization")
        if profile:
            cat_data = profile.get("category_dist", {})
            if cat_data:
                cats = list(cat_data.keys())[:8]
                vals = [cat_data[c] for c in cats]

                fig = go.Figure(data=go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=[c.title() for c in cats] + [cats[0].title()],
                    fill='toself',
                    fillcolor='rgba(10, 132, 255, 0.12)',
                    line=dict(color='#0a84ff', width=2),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, max(vals) * 1.2],
                                       showticklabels=False, gridcolor='rgba(255,255,255,0.04)'),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.04)',
                                         linecolor='rgba(255,255,255,0.04)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#86868b', family='Inter'),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=40),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="stat-card" style="margin-top: 20px;">
                <div class="stat-number">🆕</div>
                <div class="stat-label">New User — Cold Start<br>
                <span style="font-size: 0.65rem; color: #48484a;">Head to the Feed to start reading!</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Top recommended articles preview — show 10
    if st.session_state.recommendations:
        st.markdown("---")
        st.markdown("### Recommended for You")
        top_recs = st.session_state.recommendations[:10]
        cols = st.columns(2)
        for idx, article in enumerate(top_recs):
            with cols[idx % 2]:
                render_news_card(article, show_signals=False, show_actions=True, idx=idx)

    # ─── All History (from database, persists across sessions) ────
    st.markdown("---")
    st.markdown("### Recent Activity")

    # Combine session + past history
    all_activity = st.session_state.all_history
    if all_activity:
        for click in all_activity[:15]:
            emoji = get_category_emoji(click.get("category", ""))
            title = click.get("title", "Unknown")[:60]
            cat = click.get("category", "unknown")
            action = click.get("action", "")
            dwell = click.get("dwell_time", 0)
            ts = click.get("timestamp", "")

            # Format action badge
            if action == "like":
                action_badge = "👍"
            elif action == "skip" or action == "not_interested":
                action_badge = "👎"
            elif action == "glance":
                action_badge = "👁️"
            else:
                action_badge = "📖"

            # Format dwell time
            dwell_str = ""
            if dwell and float(dwell) > 0:
                d = float(dwell)
                dwell_str = f" · ⏱️ {int(d//60)}m {int(d%60)}s" if d >= 60 else f" · ⏱️ {int(d)}s"

            # Format timestamp
            time_str = ""
            if ts:
                try:
                    dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
                    time_str = f" · {dt.strftime('%b %d, %I:%M %p')}"
                except Exception:
                    pass

            st.markdown(f"- {action_badge} {emoji} **{title}...** ({cat}){dwell_str}{time_str}")
    else:
        st.info("No activity yet. Start reading articles to build your history!")


# ─── PAGE: FEED ──────────────────────────────────────────────────────────────

def render_page_feed():
    st.markdown("# Your Feed")
    st.markdown('<p style="color: #86868b; margin-top: -8px;">Personalized news, curated by AI.</p>', unsafe_allow_html=True)

    uid = st.session_state.current_user_id

    # Mood input
    st.markdown("### How are you feeling?")
    col_mood, col_emoji = st.columns([2, 1])

    with col_mood:
        mood_text = st.text_input(
            "Describe your mood",
            placeholder="e.g., I need something uplifting after a long day...",
            key="mood_input",
            label_visibility="collapsed",
        )

    with col_emoji:
        emoji_list = list(EMOJI_MOODS.keys())
        emoji_cols = st.columns(min(5, len(emoji_list)))
        for i, emoji in enumerate(emoji_list[:5]):
            with emoji_cols[i]:
                if st.button(emoji, key=f"em_{i}"):
                    mood_label, cats = get_emoji_mood(emoji)
                    st.session_state.current_mood = mood_label
                    st.session_state.mood_categories = cats
                    st.session_state.recommendations = []

    if mood_text:
        mood_result = get_full_mood_analysis(mood_text, hour=datetime.now().hour)
        st.session_state.current_mood = mood_result["detected_mood"]
        st.session_state.mood_categories = mood_result["final_categories"]

        st.markdown(f"""
        <div class="context-bar">
            <div class="context-card">
                <div class="context-value">{mood_result['detected_mood'].title()}</div>
                <div class="context-label">Detected Mood</div>
            </div>
            <div class="context-card">
                <div class="context-value">{', '.join(mood_result['mood_categories'][:3])}</div>
                <div class="context-label">Boosted Categories</div>
            </div>
            <div class="context-card">
                <div class="context-value">{mood_result['confidence']:.0%}</div>
                <div class="context-label">Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Category filter
    with st.expander("🏷️ Filter by categories", expanded=False):
        selected_cats = st.multiselect(
            "Focus on:", options=CATEGORIES, default=[],
            format_func=lambda x: f"{get_category_emoji(x)} {get_category_label(x)}",
        )

    # Generate
    col_gen, col_sig = st.columns([3, 1])
    with col_gen:
        gen_btn = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)
    with col_sig:
        show_signals = st.checkbox("Show signals", value=False)

    if gen_btn or not st.session_state.recommendations:
        mood_cats = list(set(selected_cats + st.session_state.get("mood_categories", [])))

        with st.spinner("AI is curating your feed..."):
            history = engine.user_profiles.get(uid, {}).get("history_ids", [])
            for click in st.session_state.session_clicks:
                if click["news_id"] not in history:
                    history.append(click["news_id"])

            disliked = get_disliked_news(uid)
            start_t = time.time()

            if history:
                recs = engine.recommend(
                    user_id=uid, history_ids=history,
                    mood_categories=mood_cats if mood_cats else [],
                    excluded_ids=disliked, num_recommendations=20,
                )
            else:
                recs = engine.get_cold_start_recommendations(
                    preferred_categories=mood_cats if mood_cats else st.session_state.mood_categories,
                    num_recommendations=20,
                )

            st.session_state.recommendations = recs
            st.session_state.rec_latency = (time.time() - start_t) * 1000
            st.rerun()

    # Display recommendations
    recs = st.session_state.recommendations
    if recs:
        latency = st.session_state.rec_latency
        latency_class = "slow" if latency > 2000 else ""

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="color: #86868b; font-size: 0.85rem;">
                <span class="live-dot"></span>
                {len(recs)} articles personalized for you
            </span>
            <span class="latency-badge {latency_class}">⚡ {latency:.0f}ms</span>
        </div>
        """, unsafe_allow_html=True)

        filtered = recs
        if selected_cats:
            filtered = [r for r in recs if r.get("category") in selected_cats]
            if not filtered:
                st.warning("No articles match. Showing all.")
                filtered = recs

        col1, col2 = st.columns(2)
        for idx, article in enumerate(filtered):
            with col1 if idx % 2 == 0 else col2:
                render_news_card(article, show_signals=show_signals, show_actions=True, idx=idx)

    elif uid:
        st.info("Hit **Get Recommendations** to see your personalized feed!")


# ─── PAGE: LIVE NEWS ─────────────────────────────────────────────────────────

def render_page_live_news():
    """Live news from NewsAPI — real-time current headlines."""
    st.markdown("# 🔴 Live News")
    st.markdown('<p style="color: #86868b; margin-top: -8px;">Current headlines from around the world.</p>', unsafe_allow_html=True)

    if not is_news_api_configured():
        st.warning("⚠️ NewsAPI is not configured. Add your API key to `.env`:")
        st.code('NEWS_API_KEY="your_newsapi_org_key_here"', language="bash")
        st.markdown("[Get a free API key at newsapi.org →](https://newsapi.org/register)")
        return

    uid = st.session_state.current_user_id

    # Category selector
    live_categories = ["general", "business", "technology", "sports", "entertainment", "health", "science"]
    selected_cat = st.selectbox(
        "Category",
        live_categories,
        format_func=lambda x: f"{NEWSAPI_TO_MIND.get(x, x).title()} ({x})",
        label_visibility="collapsed",
    )

    col_fetch, col_count = st.columns([3, 1])
    with col_fetch:
        fetch_btn = st.button("🔄 Fetch Latest Headlines", type="primary", use_container_width=True)
    with col_count:
        count = st.selectbox("Articles", [10, 15, 20], index=0, label_visibility="collapsed")

    # Fetch
    if fetch_btn or not st.session_state.live_news:
        with st.spinner("Fetching live headlines..."):
            articles = fetch_top_headlines(category=selected_cat, page_size=count)
            if articles:
                st.session_state.live_news = articles
                st.toast(f"Fetched {len(articles)} live articles!", icon="📡")
            else:
                st.error("Could not fetch news. Check your API key and try again.")
                return

    articles = st.session_state.live_news
    if not articles:
        return

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <span style="color: #86868b; font-size: 0.85rem;">
            <span class="live-dot"></span>
            {len(articles)} live articles
        </span>
        <span class="latency-badge" style="background: rgba(255,55,95,0.08); border-color: rgba(255,55,95,0.2); color: #ff375f;">
            🔴 LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for idx, article in enumerate(articles):
        with col1 if idx % 2 == 0 else col2:
            cat = article.get("category", "news")
            emoji = get_category_emoji(cat)
            source = article.get("source", "")
            pub_time = ""
            if article.get("published_at"):
                try:
                    dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                    pub_time = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    pass

            # Live news card
            st.markdown(f"""
            <div class="news-card" style="animation-delay: {idx * 0.06}s">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span class="live-dot"></span>
                    <span style="font-size: 0.7rem; color: #ff375f; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">LIVE</span>
                    {f'<span style="font-size: 0.7rem; color: #48484a;">· {source}</span>' if source else ''}
                    {f'<span style="font-size: 0.7rem; color: #48484a;">· {pub_time}</span>' if pub_time else ''}
                </div>
                <div class="news-title">{article.get('title', 'Untitled')}</div>
                <div class="news-abstract">{article.get('abstract', '')}</div>
                <div class="news-meta">
                    <span class="category-badge {cat}">{emoji} {cat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons for live articles
            c1, c2, c3 = st.columns([1, 1, 1.5])
            with c1:
                if st.button("👍 Like", key=f"live_like_{idx}"):
                    handle_click(article, "like", dwell_time=3.0)
            with c2:
                if st.button("👎 Skip", key=f"live_skip_{idx}"):
                    handle_skip(article)
            with c3:
                url = article.get("url", "")
                if url:
                    st.markdown(f"[🔗 Read Full Article]({url})")


# ─── PAGE: PROFILE ───────────────────────────────────────────────────────────

def render_page_profile():
    st.markdown("# Profile")

    uid = st.session_state.current_user_id
    profile = engine.get_user_profile_summary(uid)

    if profile:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{profile['total_clicks']}</div>
                <div class="stat-label">Articles Read</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{profile['num_sessions']}</div>
                <div class="stat-label">Sessions</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            top_cat = profile['top_categories'][0][0] if profile['top_categories'] else "N/A"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{get_category_emoji(top_cat)}</div>
                <div class="stat-label">Top: {top_cat}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            hour_str = f"{int(profile['avg_active_hour'])}:00"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">🕐</div>
                <div class="stat-label">Active ~{hour_str}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Category Distribution")
            if profile.get("category_dist"):
                cat_data = profile["category_dist"]
                df_cats = pd.DataFrame([
                    {"Category": f"{get_category_emoji(k)} {k.title()}", "Proportion": v}
                    for k, v in sorted(cat_data.items(), key=lambda x: x[1], reverse=True) if v > 0
                ])
                fig = px.bar(
                    df_cats.head(10), x="Proportion", y="Category", orientation="h",
                    color="Proportion",
                    color_continuous_scale=["#1c1c1e", "#0a84ff", "#5e5ce6"],
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#86868b', family='Inter'),
                    showlegend=False, coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, title=""),
                    yaxis=dict(showgrid=False, title=""),
                    margin=dict(l=0, r=0, t=10, b=10), height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### RL Bandit State")
            bandit_scores = engine.bandit.get_category_scores(uid)
            sorted_scores = sorted(bandit_scores.items(), key=lambda x: x[1], reverse=True)
            for cat, score in sorted_scores[:6]:
                emoji = get_category_emoji(cat)
                bar_len = int(score * 30)
                bar = "█" * bar_len
                st.markdown(f"`{emoji} {cat:>15s}` {bar} **{score:.3f}**")

        st.markdown("---")

        st.markdown("### Update Preferences")
        pref_cats = st.multiselect(
            "Preferred Categories", options=CATEGORIES,
            default=[c for c, _ in profile.get("top_categories", [])[:3]],
            format_func=lambda x: f"{get_category_emoji(x)} {get_category_label(x)}",
            label_visibility="collapsed",
        )
        if st.button("💾 Save Preferences"):
            update_user_preferences(uid, pref_cats)
            save_user_interests(uid, pref_cats)
            st.session_state.mood_categories = pref_cats
            st.success("Preferences saved!")

    else:
        st.markdown(f"### New User: `{uid}`")
        interests = st.session_state.auth_user.get("interests", []) if st.session_state.auth_user else []
        if interests:
            st.markdown("**Your selected interests:**")
            for cat in interests:
                st.markdown(f"- {get_category_emoji(cat)} **{get_category_label(cat)}**")
        st.info("Start reading articles on the Feed page to build your profile!")


# ─── PAGE: ANALYTICS ─────────────────────────────────────────────────────────

def render_page_analytics():
    st.markdown("# Analytics")

    uid = st.session_state.current_user_id

    # System metrics
    st.markdown("### System Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Users", f"{len(engine.get_all_user_ids()):,}")
    with c2:
        st.metric("Articles", f"{len(engine.news_dict):,}" if engine.news_dict else "0")
    with c3:
        st.metric("Categories", f"{len(set(a['category'] for a in engine.news_dict.values()))}" if engine.news_dict else "0")
    with c4:
        st.metric("Latency", f"{st.session_state.rec_latency:.0f}ms")

    st.markdown("---")

    if uid:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Recommendation Diversity")
            if st.session_state.recommendations:
                rec_cats = Counter(r["category"] for r in st.session_state.recommendations)
                df_div = pd.DataFrame([
                    {"Category": f"{get_category_emoji(k)} {k}", "Count": v}
                    for k, v in rec_cats.most_common()
                ])
                fig = px.bar(
                    df_div, x="Category", y="Count", color="Count",
                    color_continuous_scale=["#1c1c1e", "#0a84ff", "#5e5ce6"],
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#86868b', family='Inter'),
                    showlegend=False, coloraxis_showscale=False,
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                    margin=dict(l=0, r=0, t=10, b=10), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Generate recommendations first.")

        with col2:
            st.markdown("### Session Clicks")
            if st.session_state.session_clicks:
                click_cats = Counter(c["category"] for c in st.session_state.session_clicks)
                df_clicks = pd.DataFrame([
                    {"Category": f"{get_category_emoji(k)} {k}", "Clicks": v}
                    for k, v in click_cats.most_common()
                ])
                fig = px.pie(
                    df_clicks, values="Clicks", names="Category",
                    color_discrete_sequence=["#0a84ff", "#5e5ce6", "#bf5af2", "#30d158", "#ff9f0a"],
                    hole=0.5,
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#86868b', family='Inter'),
                    margin=dict(l=0, r=0, t=10, b=10), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Interact with articles to see patterns.")

        # Latency tracking
        session_data = st.session_state.recommendation_history
        if session_data:
            st.markdown("---")
            st.markdown("### Recommendation Latency")
            df_lat = pd.DataFrame(session_data)
            df_lat["idx"] = range(1, len(df_lat) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_lat["idx"], y=df_lat["latency_ms"],
                mode="lines+markers",
                line=dict(color="#0a84ff", width=2),
                marker=dict(size=6, color="#5e5ce6"),
                fill="tozeroy",
                fillcolor="rgba(10, 132, 255, 0.06)",
            ))
            fig.add_hline(y=2000, line_dash="dash", line_color="#ff453a",
                         annotation_text="2s threshold")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#86868b', family='Inter'),
                xaxis=dict(title="Request #", showgrid=False),
                yaxis=dict(title="Latency (ms)", showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
                margin=dict(l=0, r=0, t=20, b=40), height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Before/After comparison
        st.markdown("### Personalization Impact")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ❄️ Cold Start")
            cold_recs = engine.get_cold_start_recommendations(num_recommendations=5)
            for i, a in enumerate(cold_recs[:5]):
                e = get_category_emoji(a.get("category", ""))
                st.markdown(f"{i+1}. {e} {a['title'][:55]}...")
        with col2:
            st.markdown("#### 🔥 Personalized")
            if st.session_state.recommendations:
                for i, a in enumerate(st.session_state.recommendations[:5]):
                    e = get_category_emoji(a.get("category", ""))
                    st.markdown(f"{i+1}. {e} {a['title'][:55]}... ({a.get('score',0):.3f})")
            else:
                st.info("Generate recommendations first.")

        # RL Agent State
        st.markdown("---")
        st.markdown("### RL Agent State")
        bandit_scores = engine.bandit.get_category_scores(uid)
        df_bandit = pd.DataFrame([
            {"Category": f"{get_category_emoji(k)} {k.title()}", "P(click)": v}
            for k, v in sorted(bandit_scores.items(), key=lambda x: x[1], reverse=True)
            if v > 0.01
        ])
        if not df_bandit.empty:
            fig = px.bar(
                df_bandit.head(12), x="P(click)", y="Category", orientation="h",
                color="P(click)",
                color_continuous_scale=["#1c1c1e", "#0a84ff", "#5e5ce6", "#bf5af2"],
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#86868b', family='Inter'),
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title="P(click)"),
                yaxis=dict(showgrid=False, title=""),
                margin=dict(l=0, r=0, t=10, b=10), height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Sign in to see personalized analytics.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

stage = st.session_state.stage

if stage == "login":
    render_login()
elif stage == "welcome":
    render_welcome()
elif stage == "onboarding":
    render_onboarding()
elif stage == "dashboard":
    render_dashboard()
else:
    st.session_state.stage = "login"
    st.rerun()
