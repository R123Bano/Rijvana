"""
recommendation_engine.py — Hybrid RL-Based News Recommendation Engine

Components:
  1. Contextual Bandit (Thompson Sampling) — RL core
  2. Content-Based Filtering (TF-IDF similarity)
  3. Collaborative Filtering (user-user similarity)
  4. Mood & Time Context Adapters
  5. Cold-Start Handler
"""

import os
import pickle
import time
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Category Constants ───────────────────────────────────────────────────────

CATEGORIES = [
    "news", "sports", "finance", "foodanddrink", "lifestyle",
    "travel", "video", "weather", "health", "autos",
    "tv", "music", "movies", "entertainment", "kids",
    "middleeast", "northamerica"
]
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
NUM_CATEGORIES = len(CATEGORIES)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONTEXTUAL BANDIT — Thompson Sampling per Category
# ═══════════════════════════════════════════════════════════════════════════════

class ContextualBandit:
    """
    Thompson Sampling Contextual Bandit for news category selection.

    Each 'arm' is a news category. The bandit maintains a Beta distribution
    per category, updated by click rewards. Context (user profile, mood, time)
    is used to adjust the prior.
    """

    def __init__(self, num_arms=NUM_CATEGORIES, prior_alpha=1.0, prior_beta=1.0):
        self.num_arms = num_arms
        # Beta distribution parameters per arm
        self.alpha = np.full(num_arms, prior_alpha)  # successes
        self.beta = np.full(num_arms, prior_beta)    # failures
        # Per-user bandit state: {user_id: (alpha_array, beta_array)}
        self.user_bandits = {}

    def get_user_params(self, user_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get or initialize per-user Beta parameters."""
        if user_id not in self.user_bandits:
            self.user_bandits[user_id] = (
                self.alpha.copy(),
                self.beta.copy()
            )
        return self.user_bandits[user_id]

    def sample(self, user_id: str, context: np.ndarray = None) -> np.ndarray:
        """
        Thompson Sampling: draw from Beta(alpha, beta) per arm.
        Returns array of sampled values (higher = prefer that category).
        Context is used to scale the prior.
        """
        alpha, beta = self.get_user_params(user_id)

        # Draw from Beta distribution
        samples = np.random.beta(alpha, beta) 

        # Context modulation: if context provides category preference, boost
        if context is not None and len(context) >= self.num_arms:
            # Context is user's category distribution — blend it in
            context_boost = context[:self.num_arms]
            samples = 0.6 * samples + 0.4 * context_boost

        return samples

    def update(self, user_id: str, arm, reward: float):
        """
        Update Beta distribution for the selected arm.
        arm: can be int index OR string category name.
        reward: 1.0 for click, 0.0 for no-click, can be fractional for dwell time.
        """
        alpha, beta = self.get_user_params(user_id)
        # Convert string category name to index if needed
        if isinstance(arm, str):
            arm_idx = CATEGORY_TO_IDX.get(arm, None)
            if arm_idx is None:
                return  # Unknown category, skip
        else:
            arm_idx = arm
        if reward > 0:
            alpha[arm_idx] += reward
        else:
            beta[arm_idx] += 1.0
        self.user_bandits[user_id] = (alpha, beta)

    def batch_update(self, user_id: str, category_clicks: Dict[str, int], category_impressions: Dict[str, int]):
        """
        Batch update from historical data.
        """
        alpha, beta = self.get_user_params(user_id)
        for cat, clicks in category_clicks.items():
            if cat in CATEGORY_TO_IDX:
                idx = CATEGORY_TO_IDX[cat]
                alpha[idx] += clicks
        for cat, imps in category_impressions.items():
            if cat in CATEGORY_TO_IDX:
                idx = CATEGORY_TO_IDX[cat]
                no_clicks = max(0, imps - category_clicks.get(cat, 0))
                beta[idx] += no_clicks
        self.user_bandits[user_id] = (alpha, beta)

    def get_category_scores(self, user_id: str) -> Dict[str, float]:
        """Get the expected click probability per category."""
        alpha, beta = self.get_user_params(user_id)
        means = alpha / (alpha + beta)
        return {cat: float(means[i]) for i, cat in enumerate(CATEGORIES)}

    def save(self, filepath):
        """Save bandit state."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "beta": self.beta,
                "user_bandits": self.user_bandits,
            }, f)

    def load(self, filepath):
        """Load bandit state."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.alpha = data["alpha"]
            self.beta = data["beta"]
            self.user_bandits = data.get("user_bandits", {})


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONTENT-BASED FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

class ContentBasedFilter:
    """
    TF-IDF based content similarity filter.
    Recommends articles similar to user's reading history.
    """

    def __init__(self, vectorizer, news_ids, tfidf_matrix, news_dict):
        self.vectorizer = vectorizer
        self.news_ids = news_ids
        self.news_id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
        self.tfidf_matrix = tfidf_matrix
        self.news_dict = news_dict

    def get_user_profile_vector(self, history_ids: List[str]) -> np.ndarray:
        """Build user profile as average TF-IDF of their read articles."""
        indices = [self.news_id_to_idx[nid] for nid in history_ids if nid in self.news_id_to_idx]
        if not indices:
            return None
        user_vec = self.tfidf_matrix[indices].mean(axis=0)
        return np.asarray(user_vec).flatten()

    def recommend(self, history_ids: List[str], candidate_ids: List[str] = None, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Recommend articles based on content similarity to user's history.
        Returns [(news_id, score), ...]
        """
        user_vec = self.get_user_profile_vector(history_ids)
        if user_vec is None:
            return []

        # If candidates provided, only score those
        if candidate_ids:
            candidate_indices = [self.news_id_to_idx[nid] for nid in candidate_ids if nid in self.news_id_to_idx]
            if not candidate_indices:
                return []
            candidate_matrix = self.tfidf_matrix[candidate_indices]
            scores = cosine_similarity(user_vec.reshape(1, -1), candidate_matrix).flatten()
            results = [(candidate_ids[i], float(scores[i])) for i in range(len(candidate_ids)) if candidate_ids[i] in self.news_id_to_idx]
        else:
            scores = cosine_similarity(user_vec.reshape(1, -1), self.tfidf_matrix).flatten()
            results = [(self.news_ids[i], float(scores[i])) for i in range(len(scores))]

        # Remove already read
        history_set = set(history_ids)
        results = [(nid, s) for nid, s in results if nid not in history_set]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COLLABORATIVE FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

class CollaborativeFilter:
    """
    User-user collaborative filtering at the category level.
    Uses direct cosine comparison on individual user vectors (no full matrix).
    """

    def __init__(self, user_ids, collab_matrix, user_profiles):
        self.user_ids = user_ids
        self.user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self.collab_matrix = normalize(collab_matrix, norm="l2")
        self.user_profiles = user_profiles

    def get_category_recommendations(self, user_id: str, top_k: int = 10) -> Dict[str, float]:
        """
        Get weighted category preferences from similar users.
        Uses fast single-vector cosine instead of full NxN matrix.
        Returns {category: score}
        """
        if user_id not in self.user_id_to_idx:
            return {}

        user_idx = self.user_id_to_idx[user_id]
        user_vec = self.collab_matrix[user_idx].reshape(1, -1)

        # Compare against a random sample of 2000 users for speed
        sample_size = min(2000, len(self.user_ids))
        sample_indices = np.random.choice(len(self.user_ids), sample_size, replace=False)
        sample_matrix = self.collab_matrix[sample_indices]
        similarities = cosine_similarity(user_vec, sample_matrix).flatten()

        # Get top-k similar
        top_local = np.argsort(similarities)[::-1][:top_k]

        weighted_cats = np.zeros(NUM_CATEGORIES)
        for local_idx in top_local:
            global_idx = sample_indices[local_idx]
            sim_uid = self.user_ids[global_idx]
            sim_score = similarities[local_idx]
            if sim_uid in self.user_profiles and sim_uid != user_id:
                weighted_cats += sim_score * self.user_profiles[sim_uid]["category_dist"]

        # Normalize
        total = weighted_cats.sum()
        if total > 0:
            weighted_cats /= total

        return {CATEGORIES[i]: float(weighted_cats[i]) for i in range(NUM_CATEGORIES)}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HYBRID RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HybridRecommender:
    """
    Main recommender combining RL, content-based, and collaborative signals.

    Scoring: final_score = α×RL + β×content + γ×collab + δ×mood + ε×recency
    Weights adapt based on user's history length (cold-start adaptation).
    """

    def __init__(self):
        self.bandit = None
        self.content_filter = None
        self.collab_filter = None
        self.news_dict = None
        self.news_df = None
        self.user_profiles = None
        self.loaded = False
        self._cached_cat_popularity = None  # Cache popularity scores
        self._news_by_category = None       # Pre-index news by category

    def load_models(self):
        """Load all pre-trained model artifacts."""
        try:
            # Load bandit
            self.bandit = ContextualBandit()
            bandit_path = os.path.join(MODELS_DIR, "bandit_model.pkl")
            if os.path.exists(bandit_path):
                self.bandit.load(bandit_path)

            # Load news data
            with open(os.path.join(MODELS_DIR, "news_dict.pkl"), "rb") as f:
                self.news_dict = pickle.load(f)

            with open(os.path.join(MODELS_DIR, "news_df.pkl"), "rb") as f:
                self.news_df = pickle.load(f)

            # Load TF-IDF
            with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
                vectorizer = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "news_ids.pkl"), "rb") as f:
                news_ids = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
                tfidf_matrix = pickle.load(f)

            self.content_filter = ContentBasedFilter(vectorizer, news_ids, tfidf_matrix, self.news_dict)

            # Load collaborative
            with open(os.path.join(MODELS_DIR, "collab_matrix.pkl"), "rb") as f:
                collab_matrix = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "collab_user_ids.pkl"), "rb") as f:
                collab_user_ids = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "user_profiles.pkl"), "rb") as f:
                self.user_profiles = pickle.load(f)

            self.collab_filter = CollaborativeFilter(collab_user_ids, collab_matrix, self.user_profiles)

            # Pre-compute popularity scores (ONCE) — avoids iterating 50k users per call
            cat_popularity = Counter()
            for up in self.user_profiles.values():
                for cat, cnt in up.get("category_counts", {}).items():
                    cat_popularity[cat] += cnt
            total = sum(cat_popularity.values()) or 1
            self._cached_cat_popularity = {cat: cnt / total for cat, cnt in cat_popularity.items()}

            # Pre-index news by category for fast filtering
            self._news_by_category = defaultdict(list)
            for nid, article in self.news_dict.items():
                self._news_by_category[article["category"]].append(nid)

            self.loaded = True
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def _get_adaptive_weights(self, history_length: int) -> Dict[str, float]:
        """
        Adapt blending weights based on user's history length.
        Cold users → more content-based; warm users → more RL + collab.
        """
        if history_length == 0:
            # Pure cold start
            return {"rl": 0.1, "content": 0.1, "collab": 0.0, "mood": 0.5, "popularity": 0.3}
        elif history_length < 5:
            # Very new user
            return {"rl": 0.15, "content": 0.25, "collab": 0.1, "mood": 0.3, "popularity": 0.2}
        elif history_length < 20:
            # Warming up
            return {"rl": 0.25, "content": 0.3, "collab": 0.2, "mood": 0.15, "popularity": 0.1}
        else:
            # Established user
            return {"rl": 0.35, "content": 0.25, "collab": 0.25, "mood": 0.1, "popularity": 0.05}

    def _get_popularity_scores(self, candidate_ids: List[str]) -> Dict[str, float]:
        """Popularity score using pre-cached category popularity."""
        scores = {}
        pop = self._cached_cat_popularity or {}
        for nid in candidate_ids:
            if nid in self.news_dict:
                cat = self.news_dict[nid]["category"]
                scores[nid] = pop.get(cat, 0.0)
            else:
                scores[nid] = 0.0
        return scores

    def recommend(
        self,
        user_id: str,
        history_ids: List[str] = None,
        mood_categories: List[str] = None,
        excluded_ids: List[str] = None,
        num_recommendations: int = 20,
        category_filter: List[str] = None,
    ) -> List[Dict]:
        """
        Generate personalized news recommendations.

        Args:
            user_id: User identifier
            history_ids: List of news IDs the user has read
            mood_categories: Preferred categories from mood analysis
            excluded_ids: News IDs to exclude (already shown, disliked)
            num_recommendations: Number of articles to return
            category_filter: Only consider these categories (if set)

        Returns:
            List of dicts with keys: news_id, title, category, subcategory, abstract, score, signals
        """
        start_time = time.time()

        if not self.loaded:
            self.load_models()
        if not self.loaded:
            return []

        history_ids = history_ids or []
        mood_categories = mood_categories or []
        excluded_ids = set(excluded_ids or [])
        excluded_ids.update(history_ids)

        # Get adaptive weights
        weights = self._get_adaptive_weights(len(history_ids))

        # ── Build candidate pool (FAST: use pre-indexed categories) ──
        relevant_cats = set(mood_categories) if mood_categories else set()
        if user_id in (self.user_profiles or {}):
            top_cats = [c for c, _ in self.user_profiles[user_id].get("top_categories", [])]
            relevant_cats.update(top_cats[:5])
        if category_filter:
            relevant_cats = set(category_filter)
        if not relevant_cats:
            relevant_cats = {"news", "entertainment", "sports", "lifestyle", "health"}

        # Gather candidates from relevant categories (pre-indexed)
        candidates = []
        for cat in relevant_cats:
            cat_articles = self._news_by_category.get(cat, [])
            candidates.extend(cat_articles)

        # Add some from other categories for diversity
        other_cats = [c for c in CATEGORIES if c not in relevant_cats]
        for cat in other_cats[:5]:
            cat_articles = self._news_by_category.get(cat, [])
            if cat_articles:
                sample = list(np.random.choice(cat_articles, size=min(50, len(cat_articles)), replace=False))
                candidates.extend(sample)

        # Remove excluded
        candidates = [nid for nid in candidates if nid not in excluded_ids]

        # Limit total candidates to ~500 for speed
        if len(candidates) > 500:
            np.random.shuffle(candidates)
            candidates = candidates[:500]

        # ── Score each candidate ──
        scores = {}
        signal_details = {}

        # 1. RL (Bandit) scores — sample category preferences
        user_context = np.zeros(NUM_CATEGORIES)
        if user_id in (self.user_profiles or {}):
            user_context = self.user_profiles[user_id]["category_dist"]

        bandit_samples = self.bandit.sample(user_id, user_context)
        rl_scores = {}
        for nid in candidates:
            cat = self.news_dict[nid]["category"]
            cat_idx = CATEGORY_TO_IDX.get(cat, 0)
            rl_scores[nid] = bandit_samples[cat_idx]

        # 2. Content-based scores (use last 20 history items for speed)
        content_scores = {}
        if history_ids and weights["content"] > 0:
            recent_history = history_ids[-20:]  # Only use recent reads
            content_recs = self.content_filter.recommend(recent_history, candidate_ids=candidates, top_k=len(candidates))
            content_scores = {nid: score for nid, score in content_recs}

        # 3. Collaborative scores
        collab_cat_scores = {}
        if weights["collab"] > 0:
            collab_cats = self.collab_filter.get_category_recommendations(user_id)
            for nid in candidates:
                cat = self.news_dict[nid]["category"]
                collab_cat_scores[nid] = collab_cats.get(cat, 0.0)

        # 4. Mood scores
        mood_scores = {}
        if mood_categories:
            mood_set = set(mood_categories)
            for nid in candidates:
                cat = self.news_dict[nid]["category"]
                if cat in mood_set:
                    # Higher bonus for categories appearing earlier in the mood list
                    rank = mood_categories.index(cat) if cat in mood_categories else len(mood_categories)
                    mood_scores[nid] = 1.0 - (rank / max(len(mood_categories), 1)) * 0.5
                else:
                    mood_scores[nid] = 0.0

        # 5. Popularity scores
        popularity_scores = self._get_popularity_scores(candidates)

        # ── Combine scores ──
        for nid in candidates:
            rl_s = rl_scores.get(nid, 0.0)
            content_s = content_scores.get(nid, 0.0)
            collab_s = collab_cat_scores.get(nid, 0.0)
            mood_s = mood_scores.get(nid, 0.0)
            pop_s = popularity_scores.get(nid, 0.0)

            final_score = (
                weights["rl"] * rl_s +
                weights["content"] * content_s +
                weights["collab"] * collab_s +
                weights["mood"] * mood_s +
                weights["popularity"] * pop_s
            )

            # Small random noise for diversity
            final_score += np.random.uniform(0, 0.02)

            scores[nid] = final_score
            signal_details[nid] = {
                "rl": round(rl_s, 4),
                "content": round(content_s, 4),
                "collab": round(collab_s, 4),
                "mood": round(mood_s, 4),
                "popularity": round(pop_s, 4),
            }

        # ── Rank and diversify ──
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Category-diverse top-K: ensure at least 3 different categories
        results = []
        cat_counts = Counter()
        max_per_category = max(num_recommendations // 3, 3)

        for nid, score in ranked:
            if len(results) >= num_recommendations:
                break
            cat = self.news_dict[nid]["category"]
            if cat_counts[cat] < max_per_category:
                article = self.news_dict[nid]
                results.append({
                    "news_id": nid,
                    "title": article["title"],
                    "category": article["category"],
                    "subcategory": article["subcategory"],
                    "abstract": article["abstract"],
                    "url": article["url"],
                    "score": round(score, 4),
                    "signals": signal_details[nid],
                    "weights_used": weights,
                })
                cat_counts[cat] += 1

        elapsed = time.time() - start_time

        # Add metadata
        for r in results:
            r["latency_ms"] = round(elapsed * 1000, 1)

        return results

    def record_click(self, user_id: str, news_id: str, reward: float = 1.0):
        """Record a click event and update the RL model."""
        if self.loaded and news_id in self.news_dict:
            cat = self.news_dict[news_id]["category"]
            cat_idx = CATEGORY_TO_IDX.get(cat, 0)
            self.bandit.update(user_id, cat_idx, reward)

    def record_skip(self, user_id: str, news_id: str):
        """Record that user skipped/ignored an article."""
        if self.loaded and news_id in self.news_dict:
            cat = self.news_dict[news_id]["category"]
            cat_idx = CATEGORY_TO_IDX.get(cat, 0)
            self.bandit.update(user_id, cat_idx, 0.0)

    def save_bandit(self):
        """Save current bandit state."""
        if self.bandit:
            self.bandit.save(os.path.join(MODELS_DIR, "bandit_model.pkl"))

    def get_user_profile_summary(self, user_id: str) -> Optional[Dict]:
        """Get a summary of a user's profile from the MIND dataset."""
        if self.user_profiles and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return {
                "user_id": user_id,
                "total_clicks": profile["total_clicks"],
                "top_categories": profile["top_categories"],
                "num_sessions": profile["num_sessions"],
                "category_dist": {CATEGORIES[i]: round(float(v), 4) for i, v in enumerate(profile["category_dist"]) if v > 0},
                "avg_active_hour": round(profile["avg_hour"], 1),
            }
        return None

    def get_all_user_ids(self) -> List[str]:
        """Get all user IDs from the MIND dataset."""
        if self.user_profiles:
            return list(self.user_profiles.keys())
        return []

    def get_cold_start_recommendations(self, preferred_categories: List[str] = None, num_recommendations: int = 20) -> List[Dict]:
        """
        Recommendations for users with no history.
        Uses popularity + preferred categories.
        """
        if not self.loaded:
            self.load_models()

        preferred = set(preferred_categories or ["news", "entertainment", "sports", "lifestyle", "health"])

        # Score by: category match + popularity
        all_ids = list(self.news_dict.keys())
        np.random.shuffle(all_ids)

        priority = []
        others = []
        for nid in all_ids:
            article = self.news_dict[nid]
            if article["category"] in preferred:
                priority.append(nid)
            else:
                others.append(nid)

        # Take from priority first, then fill with others
        selected = priority[:num_recommendations * 2] + others[:num_recommendations]
        np.random.shuffle(selected)

        results = []
        cat_counts = Counter()
        for nid in selected:
            if len(results) >= num_recommendations:
                break
            article = self.news_dict[nid]
            cat = article["category"]
            if cat_counts[cat] < 5:
                results.append({
                    "news_id": nid,
                    "title": article["title"],
                    "category": cat,
                    "subcategory": article["subcategory"],
                    "abstract": article["abstract"],
                    "url": article["url"],
                    "score": 1.0 if cat in preferred else 0.5,
                    "signals": {"popularity": 1.0, "cold_start": True},
                    "weights_used": {"popularity": 1.0},
                })
                cat_counts[cat] += 1

        return results


# ─── Singleton Instance ────────────────────────────────────────────────────────

_recommender = None

def get_recommender() -> HybridRecommender:
    """Get or create the singleton recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = HybridRecommender()
        _recommender.load_models()
    return _recommender