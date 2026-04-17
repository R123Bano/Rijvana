"""
data_processor.py — MIND Dataset Parser & Feature Engineering

Parses the Microsoft News Dataset (MIND) and produces:
  - News article features (TF-IDF vectors, category mappings)
  - User profiles (category preferences, reading patterns)
  - User-item interaction matrix (for collaborative filtering)
  - Entity embeddings lookup
"""

import os
import csv
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ─── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "MINDsmall_train")
DEV_DIR = os.path.join(BASE_DIR, "MINDsmall_dev")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CATEGORIES = [
    "news", "sports", "finance", "foodanddrink", "lifestyle",
    "travel", "video", "weather", "health", "autos",
    "tv", "music", "movies", "entertainment", "kids",
    "middleeast", "northamerica"
]
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
NUM_CATEGORIES = len(CATEGORIES)

# ─── News Parser ──────────────────────────────────────────────────────────────

def parse_news(filepath):
    """
    Parse news.tsv → dict of {news_id: {...}}
    Columns: NewsID  Category  SubCategory  Title  Abstract  URL  TitleEntities  AbstractEntities
    """
    news = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            news_id = row[0]
            category = row[1].strip().lower() if row[1] else "news"
            subcategory = row[2].strip().lower() if row[2] else ""
            title = row[3].strip() if row[3] else ""
            abstract = row[4].strip() if row[4] else ""
            url = row[5].strip() if row[5] else ""

            # Parse entities if available
            title_entities = []
            abstract_entities = []
            try:
                if len(row) > 6 and row[6]:
                    title_entities = json.loads(row[6])
            except (json.JSONDecodeError, IndexError):
                pass
            try:
                if len(row) > 7 and row[7]:
                    abstract_entities = json.loads(row[7])
            except (json.JSONDecodeError, IndexError):
                pass

            news[news_id] = {
                "news_id": news_id,
                "category": category,
                "subcategory": subcategory,
                "title": title,
                "abstract": abstract,
                "url": url,
                "title_entities": title_entities,
                "abstract_entities": abstract_entities,
                "text": f"{title} {abstract}",
            }
    return news


# ─── Behaviors Parser ─────────────────────────────────────────────────────────

def parse_behaviors(filepath):
    """
    Parse behaviors.tsv → list of behavior dicts
    Columns: ImpressionID  UserID  Timestamp  History  Impressions
    """
    behaviors = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            impression_id = row[0]
            user_id = row[1]

            # Parse timestamp
            timestamp_str = row[2].strip() if row[2] else ""
            try:
                timestamp = datetime.strptime(timestamp_str, "%m/%d/%Y %I:%M:%S %p")
            except (ValueError, TypeError):
                timestamp = datetime.now()

            # Parse history (space-separated news IDs the user clicked before)
            history = row[3].strip().split() if row[3] and row[3].strip() else []

            # Parse impressions: "N12345-1 N67890-0" → [(news_id, clicked)]
            impressions = []
            if row[4] and row[4].strip():
                for imp in row[4].strip().split():
                    parts = imp.split("-")
                    if len(parts) == 2:
                        impressions.append((parts[0], int(parts[1])))

            behaviors.append({
                "impression_id": impression_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "history": history,
                "impressions": impressions,
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
            })
    return behaviors


# ─── User Profile Builder ─────────────────────────────────────────────────────

def build_user_profiles(behaviors, news_dict):
    """
    Build user profiles from their click history.
    Returns {user_id: {category_dist, subcategory_counts, total_clicks,
                       avg_hour, history_ids, ...}}
    """
    user_profiles = {}

    # Group behaviors by user
    user_behaviors = defaultdict(list)
    for b in behaviors:
        user_behaviors[b["user_id"]].append(b)

    for user_id, user_bhvs in user_behaviors.items():
        # Collect all clicked news IDs from history and positive impressions
        all_clicked = set()
        category_counts = Counter()
        subcategory_counts = Counter()
        hours = []
        days = []

        for bhv in user_bhvs:
            # From history
            for nid in bhv["history"]:
                all_clicked.add(nid)
                if nid in news_dict:
                    category_counts[news_dict[nid]["category"]] += 1
                    if news_dict[nid]["subcategory"]:
                        subcategory_counts[news_dict[nid]["subcategory"]] += 1

            # From positive impressions
            for nid, clicked in bhv["impressions"]:
                if clicked == 1:
                    all_clicked.add(nid)
                    if nid in news_dict:
                        category_counts[news_dict[nid]["category"]] += 1
                        if news_dict[nid]["subcategory"]:
                            subcategory_counts[news_dict[nid]["subcategory"]] += 1

            hours.append(bhv["hour"])
            days.append(bhv["day_of_week"])

        total = sum(category_counts.values()) or 1
        category_dist = np.zeros(NUM_CATEGORIES)
        for cat, cnt in category_counts.items():
            if cat in CATEGORY_TO_IDX:
                category_dist[CATEGORY_TO_IDX[cat]] = cnt / total

        user_profiles[user_id] = {
            "user_id": user_id,
            "category_dist": category_dist,
            "category_counts": dict(category_counts),
            "subcategory_counts": dict(subcategory_counts),
            "total_clicks": len(all_clicked),
            "history_ids": list(all_clicked),
            "avg_hour": np.mean(hours) if hours else 12.0,
            "most_active_hours": Counter(hours).most_common(3),
            "most_active_days": Counter(days).most_common(3),
            "top_categories": category_counts.most_common(5),
            "num_sessions": len(user_bhvs),
        }

    return user_profiles


# ─── TF-IDF Vectorizer ────────────────────────────────────────────────────────

def build_tfidf_index(news_dict, max_features=5000):
    """
    Build TF-IDF vectors for all news articles.
    Returns (vectorizer, news_id_list, tfidf_matrix)
    """
    news_ids = list(news_dict.keys())
    texts = [news_dict[nid]["text"] for nid in news_ids]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_matrix = normalize(tfidf_matrix, norm="l2")

    return vectorizer, news_ids, tfidf_matrix


# ─── Entity Embeddings ─────────────────────────────────────────────────────────

def load_entity_embeddings(filepath):
    """
    Load entity_embedding.vec → dict of {entity_id: np.array}
    """
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            entity_id = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[entity_id] = vec
    return embeddings


# ─── Collaborative Filtering Matrix ───────────────────────────────────────────

def build_user_item_matrix(user_profiles, news_ids):
    """
    Build a sparse-ish user-item interaction matrix.
    Returns (user_id_list, matrix) where matrix[i,j] = 1 if user_i clicked news_j
    """
    news_id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    user_ids = list(user_profiles.keys())

    # Use a smaller representation — category-level interaction
    # Full 50k×50k matrix would be too large
    matrix = np.zeros((len(user_ids), NUM_CATEGORIES), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        matrix[i] = user_profiles[uid]["category_dist"]

    return user_ids, matrix


# ─── Main Processing Pipeline ─────────────────────────────────────────────────

def process_and_save(verbose=True):
    """
    Full processing pipeline: parse data, build features, save to models/
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Parse news
    if verbose:
        print("📰 Parsing news articles...")
    train_news = parse_news(os.path.join(TRAIN_DIR, "news.tsv"))
    dev_news = parse_news(os.path.join(DEV_DIR, "news.tsv"))
    # Merge (dev may have additional articles)
    all_news = {**train_news, **dev_news}
    if verbose:
        print(f"   Found {len(all_news)} unique news articles")

    # 2. Parse behaviors
    if verbose:
        print("👤 Parsing user behaviors...")
    train_behaviors = parse_behaviors(os.path.join(TRAIN_DIR, "behaviors.tsv"))
    if verbose:
        print(f"   Found {len(train_behaviors)} behavior records")

    # 3. Build user profiles
    if verbose:
        print("🧠 Building user profiles...")
    user_profiles = build_user_profiles(train_behaviors, all_news)
    if verbose:
        print(f"   Built profiles for {len(user_profiles)} users")

    # 4. Build TF-IDF index
    if verbose:
        print("📊 Building TF-IDF index...")
    vectorizer, news_ids, tfidf_matrix = build_tfidf_index(all_news)
    if verbose:
        print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")

    # 5. Build collaborative matrix
    if verbose:
        print("🤝 Building collaborative filtering matrix...")
    user_ids, collab_matrix = build_user_item_matrix(user_profiles, news_ids)
    if verbose:
        print(f"   Collab matrix shape: {collab_matrix.shape}")

    # 6. Load entity embeddings
    entity_emb_path = os.path.join(TRAIN_DIR, "entity_embedding.vec")
    entity_embeddings = {}
    if os.path.exists(entity_emb_path):
        if verbose:
            print("🔗 Loading entity embeddings...")
        entity_embeddings = load_entity_embeddings(entity_emb_path)
        if verbose:
            print(f"   Loaded {len(entity_embeddings)} entity embeddings")

    # 7. Save everything
    if verbose:
        print("💾 Saving processed data...")

    with open(os.path.join(MODELS_DIR, "news_dict.pkl"), "wb") as f:
        pickle.dump(all_news, f)
    with open(os.path.join(MODELS_DIR, "user_profiles.pkl"), "wb") as f:
        pickle.dump(user_profiles, f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "news_ids.pkl"), "wb") as f:
        pickle.dump(news_ids, f)
    with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(MODELS_DIR, "collab_matrix.pkl"), "wb") as f:
        pickle.dump(collab_matrix, f)
    with open(os.path.join(MODELS_DIR, "collab_user_ids.pkl"), "wb") as f:
        pickle.dump(user_ids, f)
    with open(os.path.join(MODELS_DIR, "train_behaviors.pkl"), "wb") as f:
        pickle.dump(train_behaviors, f)

    # Save a compact news dataframe for the UI
    news_df = pd.DataFrame([
        {
            "news_id": v["news_id"],
            "category": v["category"],
            "subcategory": v["subcategory"],
            "title": v["title"],
            "abstract": v["abstract"],
            "url": v["url"],
        }
        for v in all_news.values()
    ])
    news_df.to_pickle(os.path.join(MODELS_DIR, "news_df.pkl"))

    if verbose:
        print("✅ Data processing complete!")
        print(f"   Files saved to: {MODELS_DIR}")

    return {
        "num_news": len(all_news),
        "num_users": len(user_profiles),
        "num_behaviors": len(train_behaviors),
        "tfidf_shape": tfidf_matrix.shape,
    }


if __name__ == "__main__":
    stats = process_and_save(verbose=True)
    print("\n📈 Summary:")
    for k, v in stats.items():
        print(f"   {k}: {v}")