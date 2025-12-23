import numpy as np
import heapq
from collections import defaultdict

class HybridEngine:
    def __init__(self, alpha=0.5):
        # Data structures defined in proposal 5.1
        self.user_item_graph = defaultdict(list) # Adjacency list 
        self.item_metadata = {} # Hash table for TF-IDF/features 
        self.alpha = alpha # Weight for hybrid scoring
        self.user_profiles = {} # User interest vectors
        self.item_ids = []

    # --- ALGORITHM 1: COLLABORATIVE FILTERING (USER-BASED) ---
    def fit_collaborative(self, ratings_df):
        """Builds an adjacency list graph from user-item interactions."""
        for _, row in ratings_df.iterrows():
            u, i, r = int(row['userId']), int(row['movieId']), float(row['rating'])
            self.user_item_graph[u].append((i, r))
        self.item_ids = list(ratings_df['movieId'].unique())

    def _get_user_similarity(self, u1, u2):
        """Calculates manual Pearson/Cosine similarity between two users."""
        items1 = dict(self.user_item_graph[u1])
        items2 = dict(self.user_item_graph[u2])
        common = set(items1.keys()) & set(items2.keys())
        
        if not common: return 0
        
        # Dot product calculation from scratch
        sum_sq1 = sum(items1[i]**2 for i in common)
        sum_sq2 = sum(items2[i]**2 for i in common)
        dot_product = sum(items1[i] * items2[i] for i in common)
        
        return dot_product / (np.sqrt(sum_sq1) * np.sqrt(sum_sq2))

    # --- ALGORITHM 2: CONTENT-BASED FILTERING (TF-IDF FROM SCRATCH) ---
    def fit_content(self, movies_df):
        """Manual implementation of TF-IDF vectorization for movie genres/descriptions."""
        # Step 1: Tokenization and term frequency
        doc_counts = defaultdict(int)
        tfidf_matrix = {}
        
        for _, row in movies_df.iterrows():
            m_id = int(row['movieId'])
            terms = row['genres'].lower().split('|')
            self.item_metadata[m_id] = terms
            
            tf = defaultdict(int)
            for term in terms:
                tf[term] += 1
                doc_counts[term] += 1
            tfidf_matrix[m_id] = tf

        # Step 2: Calculate IDF and finalize vectors
        n_docs = len(movies_df)
        for m_id, tf_dict in tfidf_matrix.items():
            vector = {}
            for term, count in tf_dict.items():
                idf = np.log(n_docs / (1 + doc_counts[term]))
                vector[term] = (count / len(self.item_metadata[m_id])) * idf
            self.item_metadata[m_id] = vector

    # --- ALGORITHM 3: HYBRID SCORING & TOP-N RANKING ---
    def get_recommendations(self, target_user, n=10):
        """Combines CF and CB scores and uses a Heap for O(M log N) ranking."""
        scores = {}
        
        # Cold-Start Check 
        is_cold_start = target_user not in self.user_item_graph
        
        for m_id in self.item_ids:
            # Skip if already rated
            if not is_cold_start and m_id in dict(self.user_item_graph[target_user]):
                continue
                
            cf_score = self._calculate_cf_score(target_user, m_id) if not is_cold_start else 0
            cb_score = self._calculate_cb_score(target_user, m_id)
            
            # Weighted Combination 
            # Use 100% CB if Cold Start 
            current_alpha = 0 if is_cold_start else self.alpha
            scores[m_id] = (current_alpha * cf_score) + ((1 - current_alpha) * cb_score)

        # Ranking Module: Max-Heap to select Top-N 
        # Store as (-score, id) to use as min-heap for max values
        heap = []
        for m_id, score in scores.items():
            heapq.heappush(heap, (-score, m_id))
            
        return [heapq.heappop(heap)[1] for _ in range(min(n, len(heap)))]

    def _calculate_cb_score(self, user_id, movie_id):
        # Logic for content similarity 
        return np.random.random() # Placeholder for vector dot product logic

    def _calculate_cf_score(self, user_id, movie_id):
        # Logic for user-based neighborhood 
        return np.random.random() # Placeholder for neighbor weighted average

class MultiFeatureEngine:
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.movie_features = {}

    def fit_features(self, movies_df):
        """Maps all 20+ movie features into a structured lookup table."""
        # Convert all columns except ID to a feature vector
        feature_cols = [c for c in movies_df.columns if c != 'movieId']
        for _, row in movies_df.iterrows():
            self.movie_features[int(row['movieId'])] = row[feature_cols].values

    def calculate_content_similarity(self, m1, m2):
        """Manual Cosine Similarity across all 20+ features."""
        v1 = self.movie_features.get(m1)
        v2 = self.movie_features.get(m2)
        if v1 is None or v2 is None: return 0
        
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0