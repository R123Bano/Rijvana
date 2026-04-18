"""
mood_handler.py — Mood Inference & Category Mapping

Adapted from Group-2_Moodflixx/MoodHandling/mood_handling_text.py
Maps user mood input → preferred news categories for personalization.
Falls back to rule-based mapping if LLM is unavailable.

Alert policy: critical safety news (weather/health) is ALWAYS injected
regardless of mood so users never miss a flood, storm, or health alert.
"""

import os
import json
from typing import List, Dict, Tuple

# ─── Alert categories — ALWAYS injected regardless of mood ────────────────────
# These are non-negotiable: safety alerts should surface even for happy users.
ALERT_CATEGORIES = ["weather", "health"]

# ─── Mood → Category Mapping ──────────────────────────────────────────────────
# Primary categories listed first (strongest boost).
# ALERT_CATEGORIES are injected separately — not duplicated here.

MOOD_CATEGORY_MAP = {
    # ── Positive / High Energy ─────────────────────────────────────────────
    "happy": {
        "primary": ["entertainment", "sports", "lifestyle", "travel", "music"],
        "secondary": ["foodanddrink", "movies"],
        "suppress": [],  # nothing suppressed — happy users just see uplifting content first
        "label": "You're in a great mood 😊",
        "color": "#30d158",
    },
    "excited": {
        "primary": ["sports", "entertainment", "travel", "autos", "movies"],
        "secondary": ["lifestyle", "music"],
        "suppress": [],
        "label": "Feeling pumped! ⚡",
        "color": "#ff9f0a",
    },
    "energetic": {
        "primary": ["sports", "lifestyle", "autos", "travel"],
        "secondary": ["entertainment", "health"],
        "suppress": [],
        "label": "Full of energy 💪",
        "color": "#ff9f0a",
    },
    "motivated": {
        "primary": ["finance", "news", "sports", "health", "lifestyle"],
        "secondary": ["autos", "entertainment"],
        "suppress": [],
        "label": "In the zone 🎯",
        "color": "#0a84ff",
    },
    "optimistic": {
        "primary": ["lifestyle", "travel", "entertainment", "sports", "finance"],
        "secondary": ["music", "foodanddrink"],
        "suppress": [],
        "label": "Looking on the bright side ✨",
        "color": "#30d158",
    },

    # ── Calm / Neutral ─────────────────────────────────────────────────────
    "relaxed": {
        "primary": ["lifestyle", "travel", "foodanddrink", "music"],
        "secondary": ["health", "entertainment"],
        "suppress": ["finance"],
        "label": "Taking it easy 😌",
        "color": "#64d2ff",
    },
    "calm": {
        "primary": ["lifestyle", "health", "travel", "foodanddrink"],
        "secondary": ["music", "weather"],
        "suppress": ["finance", "sports"],
        "label": "Peaceful and calm 🍃",
        "color": "#64d2ff",
    },
    "content": {
        "primary": ["lifestyle", "foodanddrink", "travel", "entertainment", "music"],
        "secondary": ["health"],
        "suppress": [],
        "label": "Feeling content 😊",
        "color": "#30d158",
    },
    "curious": {
        "primary": ["news", "finance", "health", "autos", "entertainment"],
        "secondary": ["travel", "lifestyle", "middleeast", "northamerica"],
        "suppress": [],
        "label": "Curious and exploring 🔍",
        "color": "#bf5af2",
    },
    "focused": {
        "primary": ["news", "finance", "health", "autos"],
        "secondary": ["sports", "lifestyle"],
        "suppress": ["entertainment", "music", "movies", "tv"],
        "label": "In deep focus 🧠",
        "color": "#5e5ce6",
    },
    "neutral": {
        "primary": ["news", "entertainment", "sports", "lifestyle", "health"],
        "secondary": ["finance", "travel"],
        "suppress": [],
        "label": "Just browsing",
        "color": "#86868b",
    },

    # ── Negative / Low Energy ──────────────────────────────────────────────
    "stressed": {
        "primary": ["lifestyle", "foodanddrink", "entertainment", "music"],
        "secondary": ["health", "travel"],
        "suppress": ["finance", "news", "middleeast", "northamerica"],
        "label": "Need to de-stress 😤",
        "color": "#ff453a",
    },
    "anxious": {
        "primary": ["lifestyle", "entertainment", "music", "foodanddrink"],
        "secondary": ["health"],
        "suppress": ["finance", "news", "middleeast"],
        "label": "Feeling a bit anxious 😟",
        "color": "#ff453a",
    },
    "sad": {
        "primary": ["entertainment", "music", "lifestyle", "movies", "tv"],
        "secondary": ["foodanddrink"],
        "suppress": ["news", "middleeast", "northamerica", "finance"],
        "label": "Feeling down 💙",
        "color": "#5e5ce6",
    },
    "bored": {
        "primary": ["entertainment", "sports", "movies", "tv", "travel"],
        "secondary": ["music", "autos", "lifestyle"],
        "suppress": [],
        "label": "Looking for something interesting 🎲",
        "color": "#bf5af2",
    },
    "tired": {
        "primary": ["entertainment", "lifestyle", "foodanddrink", "music", "movies"],
        "secondary": ["tv"],
        "suppress": ["finance", "news"],
        "label": "Winding down 😴",
        "color": "#8e8e93",
    },

    # ── Intellectual ───────────────────────────────────────────────────────
    "intellectual": {
        "primary": ["news", "finance", "health", "middleeast", "northamerica"],
        "secondary": ["autos", "entertainment"],
        "suppress": ["music", "movies", "tv", "kids"],
        "label": "In thinking mode 🧠",
        "color": "#5e5ce6",
    },
    "analytical": {
        "primary": ["finance", "news", "autos", "health", "sports"],
        "secondary": ["middleeast", "northamerica"],
        "suppress": ["music", "movies", "tv"],
        "label": "Analysing everything 📊",
        "color": "#0a84ff",
    },
    "creative": {
        "primary": ["entertainment", "music", "movies", "lifestyle", "travel"],
        "secondary": ["foodanddrink", "arts"],
        "suppress": ["finance"],
        "label": "In creative mode 🎨",
        "color": "#bf5af2",
    },
}

# ─── Keywords → Mood labels ───────────────────────────────────────────────────
MOOD_KEYWORDS = {
    "happy":       ["happy", "great", "wonderful", "amazing", "fantastic", "good day", "awesome", "joyful", "cheerful", "delighted"],
    "excited":     ["excited", "pumped", "thrilled", "can't wait", "hyped", "looking forward", "stoked"],
    "energetic":   ["energetic", "active", "vibrant", "lively", "full of energy", "raring to go"],
    "motivated":   ["motivated", "inspired", "driven", "productive", "accomplished", "on a roll"],
    "optimistic":  ["optimistic", "hopeful", "positive", "bright side", "looking up"],
    "stressed":    ["stressed", "overwhelmed", "pressure", "deadline", "busy", "hectic", "tense", "swamped"],
    "anxious":     ["anxious", "worried", "nervous", "uneasy", "concerned", "restless", "on edge"],
    "sad":         ["sad", "down", "unhappy", "depressed", "lonely", "blue", "upset", "crying", "heartbroken"],
    "bored":       ["bored", "nothing to do", "dull", "monotonous", "uninterested", "fed up"],
    "tired":       ["tired", "exhausted", "sleepy", "fatigued", "drained", "worn out", "need rest"],
    "relaxed":     ["relaxed", "chill", "peaceful", "calm", "serene", "laid back", "taking it easy"],
    "curious":     ["curious", "wondering", "interested", "want to learn", "exploring", "what if"],
    "focused":     ["focused", "concentrated", "determined", "laser", "in the zone", "working"],
    "intellectual":["intellectual", "think", "deep dive", "learn", "understand", "knowledge"],
    "analytical":  ["analytical", "analyse", "analyze", "data", "numbers", "research"],
    "creative":    ["creative", "create", "art", "design", "imagine", "build", "make"],
    "content":     ["content", "satisfied", "comfortable", "at peace", "settled"],
    "neutral":     ["okay", "fine", "alright", "just browsing", "nothing special"],
}

# Time-of-day → additional category boost
TIME_CATEGORY_BOOST = {
    "morning":   ["news", "health", "finance", "weather"],
    "afternoon": ["sports", "entertainment", "lifestyle", "travel"],
    "evening":   ["entertainment", "movies", "tv", "music", "foodanddrink"],
    "night":     ["entertainment", "movies", "tv", "music"],
}


def get_time_period(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def detect_mood_from_text(text: str) -> Tuple[str, float]:
    """Keyword-based mood detection. Returns (mood_label, confidence)."""
    text_lower = text.lower().strip()
    if not text_lower:
        return "neutral", 0.0

    best_mood = "neutral"
    best_score = 0
    for mood, keywords in MOOD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_mood = mood

    confidence = min(best_score / 3.0, 1.0)
    return best_mood, confidence


def get_mood_info(mood: str) -> Dict:
    """Get full mood metadata including primary/secondary categories."""
    return MOOD_CATEGORY_MAP.get(mood, MOOD_CATEGORY_MAP["neutral"])


def get_mood_categories(mood: str) -> List[str]:
    """Get the primary preferred categories for a mood (no alerts injected here)."""
    info = MOOD_CATEGORY_MAP.get(mood, MOOD_CATEGORY_MAP["neutral"])
    return info["primary"]


def get_mood_suppressed(mood: str) -> List[str]:
    """Get categories that should be suppressed for this mood."""
    info = MOOD_CATEGORY_MAP.get(mood, MOOD_CATEGORY_MAP["neutral"])
    return info.get("suppress", [])


def build_final_categories(mood: str, time_cats: List[str]) -> List[str]:
    """
    Build final merged category list:
    - Primary mood categories first (strongest boost)
    - Alert categories always included (weather, health)
    - Time-based categories appended (if not already present)
    - Suppressed categories moved to the END so they get low mood scores
    """
    info = MOOD_CATEGORY_MAP.get(mood, MOOD_CATEGORY_MAP["neutral"])
    primary = info["primary"]
    secondary = info.get("secondary", [])
    suppress = info.get("suppress", [])

    # Build ordered list: primary → alerts → secondary → time → suppressed last
    final = list(primary)
    for cat in ALERT_CATEGORIES:
        if cat not in final:
            final.append(cat)
    for cat in secondary:
        if cat not in final:
            final.append(cat)
    for cat in time_cats:
        if cat not in final:
            final.append(cat)
    # Suppressed go last — they'll score low in the mood component
    for cat in suppress:
        if cat not in final:
            final.append(cat)

    return final


def infer_mood_and_categories(mood_text: str, hour: int = 12) -> Dict:
    """
    Main mood inference. Returns full analysis dict including
    suppressed categories and alert injection metadata.
    """
    mood, confidence = detect_mood_from_text(mood_text)
    mood_cats = get_mood_categories(mood)
    time_period = get_time_period(hour)
    time_cats = TIME_CATEGORY_BOOST.get(time_period, [])
    final_cats = build_final_categories(mood, time_cats)
    info = get_mood_info(mood)

    return {
        "detected_mood": mood,
        "confidence": confidence,
        "mood_categories": mood_cats,
        "suppressed_categories": info.get("suppress", []),
        "alert_categories": ALERT_CATEGORIES,
        "time_period": time_period,
        "time_categories": time_cats,
        "final_categories": final_cats[:10],
        "mood_label": info.get("label", mood.title()),
        "mood_color": info.get("color", "#86868b"),
    }


def try_llm_mood_inference(mood_text: str) -> Dict:
    """
    Try to use LLM for more accurate mood inference.
    Falls back to rule-based if LLM unavailable.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import PromptTemplate

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
            raise ValueError("No valid GROQ_API_KEY")

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0, max_tokens=200, timeout=5, max_retries=1,
        )

        prompt = PromptTemplate.from_template("""
You are a mood analysis assistant for a news recommendation system.
Given the user's input, infer their mood and suggest news categories.

IMPORTANT: Always include "weather" and "health" in categories — these are safety alerts.
If the user seems stressed/sad/anxious, exclude heavy categories like "finance", "news", "middleeast".

User input: {mood_text}

Available categories: news, sports, finance, foodanddrink, lifestyle, travel, video, weather, health, autos, tv, music, movies, entertainment, middleeast, northamerica

Respond ONLY in JSON:
{{"mood": "one_word_mood", "confidence": 0.0-1.0, "categories": ["cat1","cat2","cat3","cat4","cat5","weather","health"]}}""")

        response = llm.invoke(prompt.invoke({"mood_text": mood_text}))
        start = response.content.find("{")
        end = response.content.rfind("}") + 1
        result = json.loads(response.content[start:end])
        cats = result.get("categories", ["news", "entertainment"])
        # Ensure alerts always present
        for alert in ALERT_CATEGORIES:
            if alert not in cats:
                cats.append(alert)
        return {
            "detected_mood": result.get("mood", "neutral"),
            "confidence": result.get("confidence", 0.5),
            "mood_categories": cats,
            "source": "llm",
        }
    except Exception:
        return None


def get_full_mood_analysis(mood_text: str, hour: int = 12) -> Dict:
    """Full mood analysis pipeline. Tries LLM first, then rule-based fallback."""
    llm_result = try_llm_mood_inference(mood_text) if mood_text.strip() else None
    rule_result = infer_mood_and_categories(mood_text, hour)

    if llm_result:
        time_period = get_time_period(hour)
        time_cats = TIME_CATEGORY_BOOST.get(time_period, [])
        mood = llm_result["detected_mood"]
        final_cats = build_final_categories(mood, time_cats)
        # Override with LLM's category list as primary
        info = get_mood_info(mood)
        return {
            "detected_mood": mood,
            "confidence": llm_result["confidence"],
            "mood_categories": llm_result["mood_categories"],
            "suppressed_categories": info.get("suppress", []),
            "alert_categories": ALERT_CATEGORIES,
            "time_period": time_period,
            "time_categories": time_cats,
            "final_categories": final_cats[:10],
            "mood_label": info.get("label", mood.title()),
            "mood_color": info.get("color", "#86868b"),
            "source": "llm",
        }

    rule_result["source"] = "rule_based"
    return rule_result


# ─── Emoji Mood Shortcuts ─────────────────────────────────────────────────────

EMOJI_MOODS = {
    "😊": ("happy",       ["entertainment", "sports", "lifestyle", "travel", "music"]),
    "😄": ("excited",     ["sports", "entertainment", "travel", "autos", "movies"]),
    "😌": ("relaxed",     ["lifestyle", "travel", "foodanddrink", "music", "health"]),
    "🤔": ("curious",     ["news", "finance", "autos", "health", "entertainment"]),
    "😫": ("stressed",    ["lifestyle", "foodanddrink", "entertainment", "music", "health"]),
    "😢": ("sad",         ["entertainment", "music", "lifestyle", "movies", "tv"]),
    "😴": ("tired",       ["entertainment", "lifestyle", "foodanddrink", "music", "movies"]),
    "💪": ("motivated",   ["finance", "news", "sports", "health", "lifestyle"]),
    "🧠": ("intellectual",["news", "finance", "health", "middleeast", "northamerica"]),
    "😐": ("neutral",     ["news", "entertainment", "sports", "lifestyle", "health"]),
    "😤": ("energetic",   ["sports", "lifestyle", "autos", "travel", "entertainment"]),
    "🎨": ("creative",    ["entertainment", "music", "movies", "lifestyle", "travel"]),
}


def get_emoji_mood(emoji: str) -> Tuple[str, List[str]]:
    """Get mood and primary categories from emoji. Always appends alert categories."""
    if emoji in EMOJI_MOODS:
        mood, cats = EMOJI_MOODS[emoji]
        # Inject alerts if not already present
        for alert in ALERT_CATEGORIES:
            if alert not in cats:
                cats = cats + [alert]
        return mood, cats
    return "neutral", ["news", "entertainment", "sports", "lifestyle", "health", "weather"]


if __name__ == "__main__":
    test_inputs = [
        "I'm feeling really stressed after a long day at work",
        "Today was amazing! Got promoted!",
        "Just woke up, need some coffee",
        "",
        "I'm curious about what's happening in the world",
        "I'm happy and want fun news",
    ]
    for text in test_inputs:
        result = infer_mood_and_categories(text, hour=14)
        print(f"Input: '{text}'")
        print(f"  Mood: {result['detected_mood']} | Primary: {result['mood_categories'][:3]}")
        print(f"  Alerts always shown: {result['alert_categories']}")
        print(f"  Suppressed: {result['suppressed_categories']}")
        print()
