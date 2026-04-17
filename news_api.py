"""
news_api.py — Live News Integration for NewsLens

Fetches current news from NewsAPI.org and formats it for the recommendation engine.
Falls back gracefully when API key is not configured.
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# ─── Category Mapping: NewsAPI categories → MIND categories ──────────────────
NEWSAPI_TO_MIND = {
    "general": "news",
    "business": "finance",
    "technology": "news",
    "science": "health",
    "health": "health",
    "sports": "sports",
    "entertainment": "entertainment",
}

MIND_TO_NEWSAPI = {
    "news": "general",
    "finance": "business",
    "sports": "sports",
    "entertainment": "entertainment",
    "health": "health",
}


def _get_api_key() -> str:
    """Get NewsAPI key from environment."""
    key = os.environ.get("NEWS_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("news_api", {}).get("api_key", "")
        except Exception:
            pass
    return key


def _make_news_id(url: str) -> str:
    """Generate a stable news_id from article URL."""
    return "live_" + hashlib.md5(url.encode()).hexdigest()[:12]


def fetch_top_headlines(
    category: str = "general",
    country: str = "us",
    page_size: int = 20,
) -> List[Dict]:
    """
    Fetch top headlines from NewsAPI.

    Returns list of articles in MIND-compatible format:
    {news_id, title, category, subcategory, abstract, url, source, published_at, is_live}
    """
    api_key = _get_api_key()
    if not api_key:
        return []

    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "country": country,
            "category": category,
            "pageSize": page_size,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return []

        articles = []
        mind_cat = NEWSAPI_TO_MIND.get(category, "news")

        for raw in data.get("articles", []):
            if not raw.get("title") or raw["title"] == "[Removed]":
                continue

            article = {
                "news_id": _make_news_id(raw.get("url", "")),
                "title": raw.get("title", ""),
                "category": mind_cat,
                "subcategory": category,
                "abstract": raw.get("description", "") or "",
                "url": raw.get("url", ""),
                "source": raw.get("source", {}).get("name", ""),
                "published_at": raw.get("publishedAt", ""),
                "image_url": raw.get("urlToImage", ""),
                "is_live": True,
            }
            articles.append(article)

        return articles

    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def fetch_live_news_multi(
    categories: List[str] = None,
    country: str = "us",
    per_category: int = 10,
) -> List[Dict]:
    """
    Fetch live news across multiple categories.

    Args:
        categories: List of MIND category names (e.g., ["news", "sports", "finance"])
        country: Country code
        per_category: Articles per category

    Returns:
        Combined list of articles from all categories
    """
    if categories is None:
        categories = ["news", "sports", "finance", "entertainment", "health"]

    all_articles = []
    seen_ids = set()

    for mind_cat in categories:
        newsapi_cat = MIND_TO_NEWSAPI.get(mind_cat, "general")
        articles = fetch_top_headlines(
            category=newsapi_cat,
            country=country,
            page_size=per_category,
        )
        for a in articles:
            if a["news_id"] not in seen_ids:
                seen_ids.add(a["news_id"])
                all_articles.append(a)

    return all_articles


def is_news_api_configured() -> bool:
    """Check if NewsAPI is configured."""
    return bool(_get_api_key())


# ─── Cached fetch to avoid hammering the API ─────────────────────────────────

_cache = {"articles": [], "timestamp": 0, "categories": []}
CACHE_TTL = 300  # 5 minutes


def get_cached_live_news(categories: List[str] = None) -> List[Dict]:
    """
    Get live news with 5-minute cache to avoid rate limiting.
    """
    now = time.time()
    cats = categories or ["news", "sports", "finance", "entertainment", "health"]
    cats_key = sorted(cats)

    if (now - _cache["timestamp"] < CACHE_TTL
            and _cache["articles"]
            and _cache["categories"] == cats_key):
        return _cache["articles"]

    articles = fetch_live_news_multi(categories=cats, per_category=8)

    if articles:
        _cache["articles"] = articles
        _cache["timestamp"] = now
        _cache["categories"] = cats_key

    return articles
