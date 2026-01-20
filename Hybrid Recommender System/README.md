# Hybrid Movie Recommendation System - README
**Student Name:** Anurag Roychowdhury (ZDA25M004)  
**Student Name:** Sreejita Roy (ZDA25M008)\
**Course:** Z5007: Programming and Data Structures  
**Institution:** IIT Madras Zanzibar  

# Project overview
A hybrid movie recommendation engine combining collaborative filtering and content‑based methods.\
Features: bipartite graph, inverted index, TF‑IDF + numerical features, precomputed k‑NN neighbor lists, bipartite projection and PageRank, two‑hop candidate expansion, weighted/switching hybrid scoring, online updates, and evaluation (Precision, Recall, NDCG).

# Quick start (3 steps)
- Clone the repo and open the project root.\
- Install dependencies (see Dependencies).\
- Prepare data (data/movies.csv, data/ratings.csv) and run main.py.

# Dependencies
Recommended Python version: 3.8–3.11

Python packages:
- numpy
- pandas
- scikit-learn

# Install with pip:

python -m venv venv\
source venv/bin/activate       # macOS / Linux\
venv\Scripts\activate           # Windows PowerShell

pip install --upgrade pip\
pip install numpy pandas scikit-learn 

# File layout (important files)

main.py — example entry point: loads data, fits engines, prints recommendations, metrics, runtime.\
src/recommender_engine.py — HybridEngine implementation (CF, CB, hybrid, graph ops).\
src/evaluation_metrics.py — Precision, Recall, NDCG implementations.\
src/generate_data.py — synthetic data generator (optional).\
data/movies.csv — item metadata (genres, numeric features).\
data/ratings.csv — user ratings (userId, movieId, rating, timestamp).

# Prepare dataset
MovieLens 1M or your CSVs must be placed in data/ with these columns:\
- movies.csv: movieId, title, genres, plus optional numeric columns used by the engine (budget, runtime, release_year, avg_critic_score).\
- ratings.csv: userId, movieId, rating, timestamp.

# Usage examples

1. Run the default pipeline\
This runs training (fit CF & CB), optional precomputation, prints top‑10 recommendations for a sample user, evaluation metrics, and runtime.
python main.py
2. Example engine initialization (if editing main.py)\
from src.recommender_engine import HybridEngine <br />
<br />engine = HybridEngine(\
    alpha=0.6,\
    k_neighbors=40,\
    hybrid_mode="weighted",      # "weighted" or "switching"\
    similarity="cosine",         # "cosine" or "pearson"\
    candidate_pool_size=200,\
    precompute_knn=True\
)

3. Generate recommendations for a user\
Inside main.py or interactive session:\
recs = engine.get_recommendations(user_id=50, n=10)\
print("Top 10:", recs)
4. Evaluate on multiple users\
Use the evaluation module to compute Precision@10, Recall@10, NDCG@10 across a sample of test users. Example pattern in main.py:\
from src.evaluation_metrics import calculate_metrics

# sample test_users from test_df
<pre>
 metrics_list = []
 for u in test_users:
     recs = engine.get_recommendations(u, n=10)
     actual = test_df[test_df['userId'] == u]['movieId'].tolist()
     metrics_list.append(calculate_metrics(recs, actual))
</pre>

# Compute averages

Tuning knobs (no output change)\
candidate_pool_size — smaller reduces latency, may slightly reduce accuracy (try 100–300).\
k_neighbors — neighbors per item (20–80 typical).\
similarity — "cosine" is slightly faster than "pearson".\
precompute_knn — set True to precompute neighbors (recommended for production).\
hybrid_mode — "weighted" or "switching"; dynamic alpha is used for cold‑start.

Example: to reduce latency, set candidate_pool_size=100 and similarity="cosine"


# Troubleshooting

1. ModuleNotFoundError or import errors

Ensure you run main.py from the project root so src is on the Python path.

Activate the virtual environment before running scripts.

2. AttributeError: 'HybridEngine' object has no attribute '...'

Make sure you are using the latest src/recommender_engine.py file.

Restart your Python session to clear cached imports: exit() then python main.py.

3. Slow runtime (>1s per request)

Confirm you precomputed neighbors (precompute_knn=True) and ran fit_content() and fit_collaborative() before timing.

Reduce candidate_pool_size (e.g., 200 → 100) and k_neighbors.

Use similarity="cosine" and ensure NumPy is using optimized BLAS (install numpy from wheels or use mkl builds).

4. Memory errors when computing similarity

Full I×I similarity matrix can be large. Use precompute_knn=True with store_matrix=False so only top‑k neighbors are kept.

Reduce vocabulary size (limit TF terms) or use sparse representations for very large datasets.

5. Missing columns in movies.csv

The engine expects genres and may use optional numeric columns. If missing, add default values or update fit_content() to handle absent columns (the implementation already uses .get() for numeric fields).

6. Evaluation metrics are low

Check train/test split and ensure test users have held-out items.

Tune alpha, candidate_pool_size, and k_neighbors.

Run cross‑validation or sample more users for stable estimates.


##  Folder Structure
- `src/`: Contains the core logic.
    - `recommender_engine.py`: The main algorithm class.
    - `generate_data.py`: Script to generate the 1M sample dataset.
    - `evaluation_metrics.py`: Script for Precision, Recall, and NDCG math.
- `data/`: Directory where genrated CSV files are stored.
- `main.py`: The execution script for demonstration.