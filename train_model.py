"""
train_model.py — Model Training Script

Processes the MIND dataset and trains:
  1. TF-IDF index (content-based)
  2. User profiles (collaborative filtering)
  3. Contextual Bandit (RL — trained via offline replay)
"""

import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict, Counter

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_processor import process_and_save, MODELS_DIR, CATEGORIES, CATEGORY_TO_IDX, NUM_CATEGORIES
from recommendation_engine import ContextualBandit


def train_bandit_offline(behaviors, news_dict, user_profiles, verbose=True):
    """
    Train the Contextual Bandit using offline replay on historical click data.

    Method: For each impression in the training data, treat clicked articles as
    positive rewards and non-clicked as negative. Update the bandit per user.
    """
    if verbose:
        print("\n🎰 Training Contextual Bandit (Thompson Sampling)...")
        print("   Method: Offline reward replay on MIND training behaviors")

    bandit = ContextualBandit(num_arms=NUM_CATEGORIES)

    total_impressions = 0
    total_clicks = 0
    total_skips = 0

    # Process each user's behaviors
    user_behaviors = defaultdict(list)
    for b in behaviors:
        user_behaviors[b["user_id"]].append(b)

    num_users = len(user_behaviors)
    processed = 0

    for user_id, bhvs in user_behaviors.items():
        # Initialize user bandit with their historical category distribution
        if user_id in user_profiles:
            profile = user_profiles[user_id]
            cat_clicks = {}
            cat_impressions = {}
            for cat, cnt in profile.get("category_counts", {}).items():
                cat_clicks[cat] = cnt
                # Estimate impressions as ~3x clicks (typical CTR ~30% for shown articles)
                cat_impressions[cat] = cnt * 3

            bandit.batch_update(user_id, cat_clicks, cat_impressions)

        # Process individual impressions for fine-grained updates
        for bhv in bhvs:
            for news_id, clicked in bhv["impressions"]:
                if news_id in news_dict:
                    cat = news_dict[news_id]["category"]
                    cat_idx = CATEGORY_TO_IDX.get(cat, 0)

                    if clicked == 1:
                        bandit.update(user_id, cat_idx, 1.0)
                        total_clicks += 1
                    else:
                        bandit.update(user_id, cat_idx, 0.0)
                        total_skips += 1

                    total_impressions += 1

        processed += 1
        if verbose and processed % 10000 == 0:
            print(f"   Processed {processed}/{num_users} users...")

    if verbose:
        ctr = total_clicks / max(total_impressions, 1) * 100
        print(f"   ✅ Bandit training complete!")
        print(f"   Total impressions: {total_impressions:,}")
        print(f"   Total clicks: {total_clicks:,}")
        print(f"   Overall CTR: {ctr:.1f}%")
        print(f"   Users with personalized bandits: {len(bandit.user_bandits):,}")

    return bandit


def compute_training_metrics(bandit, user_profiles, news_dict, behaviors):
    """
    Compute and display training metrics.
    """
    print("\n📊 Training Metrics:")

    # 1. Per-category average click probability
    print("\n   Category Click Probabilities (global prior):")
    means = bandit.alpha / (bandit.alpha + bandit.beta)
    for i, cat in enumerate(CATEGORIES):
        if means[i] > 0.01:
            bar = "█" * int(means[i] * 50)
            print(f"   {cat:>15s}: {means[i]:.3f} {bar}")

    # 2. Article coverage
    total_articles = len(news_dict)
    cat_dist = Counter(a["category"] for a in news_dict.values())
    print(f"\n   Total articles in index: {total_articles:,}")
    print(f"   Categories: {len(cat_dist)}")

    # 3. User coverage
    print(f"\n   Total user profiles: {len(user_profiles):,}")
    avg_clicks = np.mean([p["total_clicks"] for p in user_profiles.values()])
    median_clicks = np.median([p["total_clicks"] for p in user_profiles.values()])
    print(f"   Avg clicks per user: {avg_clicks:.1f}")
    print(f"   Median clicks per user: {median_clicks:.1f}")

    # 4. Cold-start stats
    cold_users = sum(1 for p in user_profiles.values() if p["total_clicks"] < 5)
    warm_users = sum(1 for p in user_profiles.values() if p["total_clicks"] >= 5)
    print(f"\n   Cold-start users (<5 clicks): {cold_users:,}")
    print(f"   Warm users (≥5 clicks): {warm_users:,}")


def main():
    total_start = time.time()

    print("=" * 60)
    print("🚀 MIND News Recommendation — Model Training Pipeline")
    print("=" * 60)

    # Step 1: Process raw data
    print("\n" + "─" * 60)
    print("STEP 1: Data Processing")
    print("─" * 60)
    stats = process_and_save(verbose=True)

    # Step 2: Load processed data for bandit training
    print("\n" + "─" * 60)
    print("STEP 2: Loading processed data...")
    print("─" * 60)

    with open(os.path.join(MODELS_DIR, "train_behaviors.pkl"), "rb") as f:
        behaviors = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "news_dict.pkl"), "rb") as f:
        news_dict = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "user_profiles.pkl"), "rb") as f:
        user_profiles = pickle.load(f)

    # Step 3: Train contextual bandit
    print("\n" + "─" * 60)
    print("STEP 3: RL Training (Contextual Bandit)")
    print("─" * 60)
    bandit = train_bandit_offline(behaviors, news_dict, user_profiles, verbose=True)

    # Save bandit
    bandit_path = os.path.join(MODELS_DIR, "bandit_model.pkl")
    bandit.save(bandit_path)
    print(f"   💾 Bandit model saved to: {bandit_path}")

    # Step 4: Compute metrics
    print("\n" + "─" * 60)
    print("STEP 4: Evaluation Metrics")
    print("─" * 60)
    compute_training_metrics(bandit, user_profiles, news_dict, behaviors)

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"✅ Training pipeline complete! ({total_time:.1f}s)")
    print("=" * 60)

    # List saved files
    print(f"\n📁 Model files in {MODELS_DIR}:")
    for fname in sorted(os.listdir(MODELS_DIR)):
        fpath = os.path.join(MODELS_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"   {fname:30s} ({size_mb:.1f} MB)")

    print(f"\n🎯 Ready to run: streamlit run app.py")


if __name__ == "__main__":
    main()
