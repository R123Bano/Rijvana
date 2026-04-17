"""
mood_handler.py — Mood Inference & Category Mapping

Adapted from Group-2_Moodflixx/MoodHandling/mood_handling_text.py
Maps user mood input → preferred news categories for personalization.
Falls back to rule-based mapping if LLM is unavailable.
"""

import os
import json
from typing import List, Dict, Tuple

# ─── Rule-Based Mood → Category Mapping ───────────────────────────────────────

MOOD_CATEGORY_MAP = {
    # Positive / High Energy
    "happy": ["entertainment", "sports", "lifestyle", "travel", "music"],
    "excited": ["sports", "entertainment", "travel", "autos", "movies"],
    "energetic": ["sports", "fitness", "autos", "travel", "entertainment"],
    "motivated": ["finance", "news", "sports", "health", "lifestyle"],
    "optimistic": ["lifestyle", "travel", "entertainment", "sports", "finance"],

    # Calm / Neutral
    "relaxed": ["lifestyle", "travel", "foodanddrink", "music", "health"],
    "calm": ["lifestyle", "health", "travel", "foodanddrink", "weather"],
    "content": ["lifestyle", "foodanddrink", "travel", "entertainment", "music"],
    "curious": ["news", "finance", "autos", "health", "entertainment"],
    "focused": ["news", "finance", "health", "autos", "sports"],

    # Negative / Low Energy
    "stressed": ["health", "lifestyle", "foodanddrink", "entertainment", "music"],
    "anxious": ["health", "lifestyle", "entertainment", "music", "foodanddrink"],
    "sad": ["entertainment", "music", "lifestyle", "movies", "tv"],
    "bored": ["entertainment", "sports", "movies", "tv", "travel"],
    "tired": ["entertainment", "lifestyle", "foodanddrink", "music", "movies"],

    # Intellectual
    "intellectual": ["news", "finance", "health", "autos", "entertainment"],
    "analytical": ["finance", "news", "autos", "health", "sports"],
    "creative": ["entertainment", "music", "movies", "lifestyle", "travel"],
}

# Keywords in user text → mood labels
MOOD_KEYWORDS = {
    "happy": ["happy", "great", "wonderful", "amazing", "fantastic", "good day", "awesome", "joyful"],
    "excited": ["excited", "pumped", "thrilled", "can't wait", "hyped", "looking forward"],
    "stressed": ["stressed", "overwhelmed", "pressure", "deadline", "busy", "hectic", "tense"],
    "anxious": ["anxious", "worried", "nervous", "uneasy", "concerned", "restless"],
    "sad": ["sad", "down", "unhappy", "depressed", "lonely", "blue", "upset", "crying"],
    "bored": ["bored", "nothing to do", "dull", "monotonous", "uninterested"],
    "tired": ["tired", "exhausted", "sleepy", "fatigued", "drained", "worn out"],
    "relaxed": ["relaxed", "chill", "peaceful", "calm", "serene", "laid back"],
    "curious": ["curious", "wondering", "interested", "want to learn", "exploring"],
    "motivated": ["motivated", "inspired", "driven", "productive", "accomplished"],
    "energetic": ["energetic", "active", "vibrant", "lively", "full of energy"],
    "focused": ["focused", "concentrated", "determined", "laser", "in the zone"],
}

# Time-of-day → category boost
TIME_CATEGORY_BOOST = {
    "morning": ["news", "health", "finance", "weather"],      # 6-11
    "afternoon": ["sports", "entertainment", "lifestyle", "travel"],  # 12-16
    "evening": ["entertainment", "movies", "tv", "music", "foodanddrink"],  # 17-21
    "night": ["entertainment", "movies", "tv", "music"],       # 22-5
}


def get_time_period(hour: int) -> str:
    """Map hour to time period."""
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def detect_mood_from_text(text: str) -> Tuple[str, float]:
    """
    Simple keyword-based mood detection.
    Returns (mood_label, confidence)
    """
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

    confidence = min(best_score / 3.0, 1.0)  # Normalize
    return best_mood, confidence


def get_mood_categories(mood: str) -> List[str]:
    """Get preferred news categories for a given mood."""
    return MOOD_CATEGORY_MAP.get(mood, ["news", "entertainment", "sports", "lifestyle", "health"])


def infer_mood_and_categories(mood_text: str, hour: int = 12) -> Dict:
    """
    Main mood inference function.
    Takes user's mood text + current hour → returns mood + boosted categories.

    Returns:
        {
            "detected_mood": str,
            "confidence": float,
            "mood_categories": List[str],
            "time_period": str,
            "time_categories": List[str],
            "final_categories": List[str]  # merged & ranked
        }
    """
    # Detect mood
    mood, confidence = detect_mood_from_text(mood_text)

    # Get mood-based categories
    mood_cats = get_mood_categories(mood)

    # Get time-based categories
    time_period = get_time_period(hour)
    time_cats = TIME_CATEGORY_BOOST.get(time_period, [])

    # Merge: mood categories first, then time categories (deduped)
    final_cats = list(mood_cats)
    for cat in time_cats:
        if cat not in final_cats:
            final_cats.append(cat)

    return {
        "detected_mood": mood,
        "confidence": confidence,
        "mood_categories": mood_cats,
        "time_period": time_period,
        "time_categories": time_cats,
        "final_categories": final_cats[:8],
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
            temperature=0,
            max_tokens=200,
            timeout=5,
            max_retries=1,
        )

        prompt = PromptTemplate.from_template("""
You are a mood analysis assistant for a news recommendation system.

Given the user's input about their current state, infer their mood and suggest news categories.

User input: {mood_text}

Available news categories: news, sports, finance, foodanddrink, lifestyle, travel, video, weather, health, autos, tv, music, movies, entertainment

Respond ONLY in this JSON format:
```json
{{
    "mood": "one_word_mood",
    "confidence": 0.0-1.0,
    "categories": ["cat1", "cat2", "cat3", "cat4", "cat5"],
    "reasoning": "Brief explanation"
}}
```""")

        response = llm.invoke(prompt.invoke({"mood_text": mood_text}))
        start = response.content.find("{")
        end = response.content.rfind("}") + 1
        result = json.loads(response.content[start:end])
        return {
            "detected_mood": result.get("mood", "neutral"),
            "confidence": result.get("confidence", 0.5),
            "mood_categories": result.get("categories", ["news", "entertainment"]),
            "reasoning": result.get("reasoning", ""),
            "source": "llm",
        }

    except Exception:
        # Fall back to rule-based
        return None


def get_full_mood_analysis(mood_text: str, hour: int = 12) -> Dict:
    """
    Full mood analysis pipeline. Tries LLM first, then rule-based fallback.
    """
    # Try LLM
    llm_result = try_llm_mood_inference(mood_text) if mood_text.strip() else None

    # Rule-based
    rule_result = infer_mood_and_categories(mood_text, hour)

    if llm_result:
        # Merge LLM result with time context
        time_period = get_time_period(hour)
        time_cats = TIME_CATEGORY_BOOST.get(time_period, [])
        merged_cats = list(llm_result["mood_categories"])
        for cat in time_cats:
            if cat not in merged_cats:
                merged_cats.append(cat)

        return {
            "detected_mood": llm_result["detected_mood"],
            "confidence": llm_result["confidence"],
            "mood_categories": llm_result["mood_categories"],
            "time_period": time_period,
            "time_categories": time_cats,
            "final_categories": merged_cats[:8],
            "source": "llm",
            "reasoning": llm_result.get("reasoning", ""),
        }

    rule_result["source"] = "rule_based"
    rule_result["reasoning"] = f"Detected mood '{rule_result['detected_mood']}' from keywords in input text"
    return rule_result


# ─── Emoji Mood Shortcuts ─────────────────────────────────────────────────────

EMOJI_MOODS = {
    "😊": ("happy", ["entertainment", "sports", "lifestyle", "travel", "music"]),
    "😄": ("excited", ["sports", "entertainment", "travel", "autos", "movies"]),
    "😌": ("relaxed", ["lifestyle", "travel", "foodanddrink", "music", "health"]),
    "🤔": ("curious", ["news", "finance", "autos", "health", "entertainment"]),
    "😫": ("stressed", ["health", "lifestyle", "foodanddrink", "entertainment", "music"]),
    "😢": ("sad", ["entertainment", "music", "lifestyle", "movies", "tv"]),
    "😴": ("tired", ["entertainment", "lifestyle", "foodanddrink", "music", "movies"]),
    "💪": ("motivated", ["finance", "news", "sports", "health", "lifestyle"]),
    "🧠": ("intellectual", ["news", "finance", "health", "autos", "entertainment"]),
    "😐": ("neutral", ["news", "entertainment", "sports", "lifestyle", "health"]),
}


def get_emoji_mood(emoji: str) -> Tuple[str, List[str]]:
    """Get mood and categories from emoji."""
    if emoji in EMOJI_MOODS:
        return EMOJI_MOODS[emoji]
    return "neutral", ["news", "entertainment", "sports", "lifestyle", "health"]


if __name__ == "__main__":
    # Test
    test_inputs = [
        "I'm feeling really stressed after a long day at work",
        "Today was amazing! Got promoted!",
        "Just woke up, need some coffee",
        "",
        "I'm curious about what's happening in the world",
    ]
    for text in test_inputs:
        result = infer_mood_and_categories(text, hour=14)
        print(f"Input: '{text}'")
        print(f"  Mood: {result['detected_mood']} | Categories: {result['final_categories']}")
        print()
