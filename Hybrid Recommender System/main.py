import pandas as pd
import time
import numpy as np
from src.recommender_engine import HybridEngine
from src.evaluation_metrics import calculate_metrics
from sklearn.model_selection import train_test_split
 

# ---------------- LOAD DATA ----------------
print("Loading datasets...")
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# ---------------- SPLIT TRAIN/TEST ----------------
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# ---------------- INITIALIZE ENGINE ----------------
engine = HybridEngine(alpha=0.6, k_neighbors=20, hybrid_mode="weighted", similarity="pearson")

# ---------------- TRAIN CF & CB ----------------
print("Fitting Collaborative Filtering...")
engine.fit_collaborative(train_df)

print("Fitting Content-Based features...")
engine.fit_content(movies)

# ---------------- COMPLEXITY ANALYSIS ----------------
print("\n--- Complexity Analysis ---")
engine.complexity_analysis()

# ---------------- SELECT TEST USER ----------------
user_id = 2150
user_test_items = test_df[test_df['userId'] == user_id]['movieId'].tolist()
if not user_test_items:
    user_test_items = [train_df[train_df['userId'] == user_id]['movieId'].iloc[0]]

# ---------------- RECOMMEND ----------------
start_time = time.time()
recs = engine.get_recommendations(user_id, n=10)
end_time = time.time()

# ---------------- OUTPUT ----------------
print(f"\nTop 10 Recommendations for User {user_id}:")
for i, movie_id in enumerate(recs):
    title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
    print(f"{i+1}. {title}")

# ---------------- EVALUATION ----------------
metrics = calculate_metrics(recs, user_test_items)
print("\n--- Evaluation Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ---------------- PERFORMANCE ----------------
runtime = end_time - start_time
print(f"\nRuntime: {runtime:.4f} seconds")
if runtime < 1.0:
    print("Requirement Met: Latency under 1 second.")
else:
    print("Warning: Latency exceeded 1 second.")

# This is the main script to run the hybrid recommender system, evaluate it, and analyze performance