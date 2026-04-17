# 📰 NewsLens — Hyper-Personalized News Recommendation Engine

> **Group 12 — IIT Mandi Hackathon**  
> AI-driven personalized news content recommendation using Reinforcement Learning

---

## 🎯 Problem Statement

Build an AI-driven system for personalized news content recommendation that:
- Models user preferences using real-time and historical data
- Adapts recommendations using contextual signals (mood, time, recent activity)
- Generates dynamic, ranked recommendations that evolve with user interaction
- Demonstrates cold-start handling for new users
- Achieves < 2 second response time

---

## 🧠 Our Approach

### Hybrid RL-Based Recommendation Engine

We use a **Contextual Bandit with Thompson Sampling** as the core RL component, combined with content-based and collaborative filtering:

```
final_score = α × RL_score + β × content_score + γ × collab_score + δ × mood_bonus + ε × popularity
```

| Component | Method | Purpose |
|-----------|--------|---------|
| **RL Agent** | Thompson Sampling (Beta distribution per category) | Explore-exploit balance based on click rewards |
| **Content-Based** | TF-IDF on article text + cosine similarity | Match articles to user's reading history |
| **Collaborative** | User-user similarity on category vectors | Leverage similar users' preferences |
| **Mood Adapter** | Keyword detection + LLM inference (fallback) | Map mood → preferred categories |
| **Cold-Start** | Popularity-based + onboarding preferences | Handle users with no history |

### Adaptive Weight Blending

Weights automatically adjust based on user's history length:

| User Type | RL | Content | Collab | Mood | Popularity |
|-----------|-----|---------|--------|------|------------|
| Cold Start (0 clicks) | 0.10 | 0.10 | 0.00 | 0.50 | 0.30 |
| New (< 5 clicks) | 0.15 | 0.25 | 0.10 | 0.30 | 0.20 |
| Warming (< 20 clicks) | 0.25 | 0.30 | 0.20 | 0.15 | 0.10 |
| Established (20+ clicks) | 0.35 | 0.25 | 0.25 | 0.10 | 0.05 |

---

## 📊 Dataset

**MIND — Microsoft News Dataset** (Small version)
- Source: https://msnews.github.io/
- **50,000 users** with reading history
- **51,282 news articles** across 17 categories
- **156,965 behavior records** with click/impression data
- Entity embeddings from Wikidata knowledge graph

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd grp12_hackathon
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python3 train_model.py
```

This will:
- Parse the MIND dataset (news.tsv + behaviors.tsv)
- Build TF-IDF vectors for all articles
- Create user profiles from click history
- Train the Contextual Bandit via offline replay
- Save all model artifacts to `models/`

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

### 4. (Optional) Add LLM-based Mood Analysis

Add your Groq API key to `.env`:
```
GROQ_API_KEY="your_key_here"
```

---

## 🖥️ Features

### 🏠 Dashboard
- Real-time context display (mood, time, user type)
- Category preference visualization (polar chart)
- Session activity tracking

### 📰 Recommendations
- Personalized news feed with ranked articles
- Mood input (text + emoji selector)
- Category filtering
- Like/Dislike buttons that update the RL model in real-time
- Scoring signal breakdown (RL, content, collab, mood)

### 👤 User Profile
- Browse 50,000 MIND users or create new profiles
- Category distribution charts
- RL bandit state visualization
- Preference customization

### 📊 Analytics
- Article distribution pie chart
- Recommendation diversity metrics
- Latency tracking (< 2s target)
- Before/after personalization comparison
- RL agent learning curve

---

## 📁 Project Structure

```
grp12_hackathon/
├── app.py                    # Streamlit web application (4 pages)
├── data_processor.py         # MIND dataset parser & feature engineering
├── recommendation_engine.py  # Hybrid RL recommendation engine
├── train_model.py            # Model training pipeline
├── mood_handler.py           # Mood inference & category mapping
├── database.py               # SQLite database layer
├── styles.css                # Custom dark theme CSS
├── requirements.txt          # Python dependencies
├── .env                      # API keys (optional)
├── README.md                 # This file
└── models/                   # Trained model artifacts (auto-generated)
```

---

## 🔬 Personalization Signals

1. **Click History** — Articles the user has read (from MIND dataset + live session)
2. **Mood** — Detected from text input or emoji selection
3. **Time-of-Day** — Morning/afternoon/evening/night context
4. **Category Preference** — Learned via RL (Thompson Sampling)
5. **Similar Users** — Collaborative filtering for preference discovery

---

## ⚡ Performance

- **Recommendation latency**: < 500ms typical (well under 2s requirement)
- **Model training**: ~60-90 seconds for full MIND dataset
- **Memory**: ~200MB for loaded model artifacts

---

## 🏗️ Architecture

```
┌─────────────────────────────┐
│     Streamlit Web App       │
│  (Dashboard | Recs | etc.)  │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│   Hybrid Recommendation     │
│         Engine               │
│  ┌────┐ ┌──────┐ ┌──────┐  │
│  │ RL │ │ TFIDF│ │Collab│  │
│  └──┬─┘ └──┬───┘ └──┬───┘  │
│     └───┬───┘        │      │
│    Final Scoring + Diversity │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│     MIND Dataset             │
│  50k users • 51k articles   │
│  Behaviors • Embeddings     │
└─────────────────────────────┘
```

---

## 📚 References

- Wu, F. et al. "MIND: A Large-scale Dataset for News Recommendation." ACL 2020.
- Adapted from Group-2_Moodflixx (IIT Mandi Hackathon — Movie Recommendation)
