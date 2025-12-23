import pandas as pd
import time
from src.recommender_engine import HybridEngine
from src.evaluation_metrics import calculate_metrics

# 1. DATA PREPROCESSING 
print("Loading MovieLens 1M dataset...") 
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# 2. INITIALIZE ENGINE
engine = HybridEngine(alpha=0.6)

# 3. TRAINING ALGORITHMS
print("Building Collaborative Filtering Graph...")
engine.fit_collaborative(ratings)

print("Extracting Content TF-IDF Features...")
engine.fit_content(movies)

# 4. TESTING & PERFORMANCE 
user_id = 50
start_time = time.time()
recs = engine.get_recommendations(user_id, n=10)
end_time = time.time()

# 5. OUTPUT RESULTS
print(f"\nTop 10 Recommendations for User {user_id}:")
for i, movie_id in enumerate(recs):
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"{i+1}. {title}")

runtime = end_time - start_time
print(f"\n--- Performance Summary ---")
print(f"Runtime: {runtime:.4f} seconds") # Target: < 1 second 

if runtime < 1.0:
    print("Requirement Met: Latency is under 1 second.")
else:
    print("Warning: Latency exceeded 1 second.")