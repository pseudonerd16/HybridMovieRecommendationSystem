import pandas as pd
import numpy as np
import os

# Create data directory
os.makedirs('data', exist_ok=True)

print("Generating 1,000,000 ratings with 20+ features...")

# 1. Generate Movies (approx 5,000 movies)
n_movies = 5000
# Full list of genres for pipe-separated string
genres_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
               'Documentary', 'Drama', 'Fantasy', 'Horror', 'Sci-Fi', 'Thriller']

def generate_genre_string():
    # Randomly pick 1 to 3 genres and join them with |
    selected = np.random.choice(genres_list, np.random.randint(1, 4), replace=False)
    return "|".join(selected)

movies = pd.DataFrame({
    'movieId': range(1, n_movies + 1),
    'title': [f"Movie {i} (2024)" for i in range(1, n_movies + 1)],
    'genres': [generate_genre_string() for _ in range(n_movies)], 
    'budget': np.random.randint(1, 200, n_movies) * 1000000,
    'runtime': np.random.randint(80, 180, n_movies),
    'release_year': np.random.randint(1990, 2025, n_movies),
    'is_sequel': np.random.choice([0, 1], n_movies, p=[0.8, 0.2]),
    'avg_critic_score': np.random.uniform(2, 10, n_movies).round(1),
    'director_id': np.random.randint(1, 500, n_movies),
    'production_cost': np.random.randint(50, 500, n_movies) * 10000,
    'has_subtitles': np.random.choice([0, 1], n_movies)
})

# One-hot encode genres (adds 12 more features to reach the 20+ target)
for g in genres_list:
    movies[f'genre_{g}'] = movies['genres'].apply(lambda x: 1 if g in x else 0)

# 2. Generate Users (approx 10,000 users)
n_users = 10000
users = pd.DataFrame({
    'userId': range(1, n_users + 1),
    'age': np.random.randint(18, 70, n_users),
    'gender': np.random.choice([0, 1], n_users),  # 0: M, 1: F
    'is_premium': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
    'user_region': np.random.randint(1, 10, n_users)
})

# 3. Generate Ratings (1,000,000 rows)
n_ratings = 1000000
ratings = pd.DataFrame({
    'userId': np.random.randint(1, n_users + 1, n_ratings),
    'movieId': np.random.randint(1, n_movies + 1, n_ratings),
    'rating': np.random.randint(1, 6, n_ratings),  # Target Variable (1-5)
    'timestamp': np.random.randint(1500000000, 1700000000, n_ratings),
    'watch_duration_pct': np.random.uniform(0.1, 1.0, n_ratings).round(2),
    'device_type': np.random.choice([1, 2, 3], n_ratings), # 1: Mobile, 2: TV, 3: Web
    'day_of_week': np.random.randint(0, 7, n_ratings),
    'internet_speed_mbps': np.random.randint(5, 200, n_ratings),
    'user_session_id': range(1, n_ratings + 1)
})

# Save files
movies.to_csv('data/movies.csv', index=False)
ratings.to_csv('data/ratings.csv', index=False)

print(f"Success! Generated {len(ratings)} samples.")
print(f"Movies columns: {len(movies.columns)} | Ratings columns: {len(ratings.columns)}")
print(f"Total features across files: {len(movies.columns) + len(ratings.columns) - 2}")
print(f"Ratings File Size: {os.path.getsize('data/ratings.csv') / (1024*1024):.2f} MB")