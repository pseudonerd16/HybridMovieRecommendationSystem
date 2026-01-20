import pandas as pd
import numpy as np
import os

# ---------------- CREATE DATA DIRECTORY ----------------
os.makedirs('data', exist_ok=True)

# ---------------- MOVIES ----------------
n_movies = 5000
genres_list = ['Action','Adventure','Animation','Children','Comedy','Crime',
               'Documentary','Drama','Fantasy','Horror','Sci-Fi','Thriller']

def generate_genres():
    return "|".join(np.random.choice(genres_list, np.random.randint(1,4), replace=False))

movies = pd.DataFrame({
    'movieId': range(1, n_movies+1),
    'title': [f"Movie {i}" for i in range(1, n_movies+1)],
    'genres': [generate_genres() for _ in range(n_movies)],
    'budget': np.random.randint(1,200, n_movies) * 1_000_000,
    'runtime': np.random.randint(80,180,n_movies),
    'release_year': np.random.randint(1990,2025,n_movies),
    'avg_critic_score': np.round(np.random.uniform(2,10,n_movies),1),
    'director_id': np.random.randint(1,500,n_movies),
    'is_sequel': np.random.choice([0,1],n_movies,p=[0.8,0.2]),
})

# ---------------- USERS ----------------
n_users = 10000
users = pd.DataFrame({
    'userId': range(1,n_users+1),
    'age': np.random.randint(18,70,n_users),
    'gender': np.random.choice([0,1],n_users),
    'is_premium': np.random.choice([0,1],n_users,p=[0.7,0.3]),
    'region': np.random.randint(1,10,n_users)
})

# ---------------- RATINGS ----------------
n_ratings = 1_000_000
ratings = pd.DataFrame({
    'userId': np.random.randint(1,n_users+1,n_ratings),
    'movieId': np.random.randint(1,n_movies+1,n_ratings),
    'rating': np.random.randint(1,6,n_ratings),
    'timestamp': np.random.randint(1500000000,1700000000,n_ratings),
})

# ---------------- SAVE FILES ----------------
movies.to_csv('data/movies.csv',index=False)
users.to_csv('data/users.csv',index=False)
ratings.to_csv('data/ratings.csv',index=False)

print(f"Generated {len(movies)} movies, {len(users)} users, {len(ratings)} ratings.")

# This module generates synthetic data for movies, users, and ratings