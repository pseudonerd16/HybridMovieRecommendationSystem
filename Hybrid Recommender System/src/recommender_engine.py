import math
import heapq
from collections import defaultdict, Counter
import numpy as np

class HybridEngine:
    """
    Robust Hybrid Recommender Engine (drop-in replacement).

    Public API expected by main.py:
      - fit_collaborative(ratings_df)
      - fit_content(movies_df)
      - get_recommendations(user_id, n=10)
      - complexity_analysis()
      - add_rating(user_id, item_id, rating)
      - update_item_vector(item_id, new_sparse_vector)

    This implementation is defensive: it handles missing columns, empty datasets,
    and ensures get_recommendations returns a sensible list (possibly popularity fallback).
    """

    def __init__(self,
                 alpha=0.6,
                 k_neighbors=20,
                 hybrid_mode="weighted",
                 similarity="cosine",
                 candidate_pool_size=200,
                 precompute_knn=True):
        # Bipartite graph structures
        self.user_item_graph = defaultdict(list)   # user_id -> list of (item_id, rating)
        self.item_users = defaultdict(list)        # item_id -> list of user_id
        self.item_popularity = Counter()           # item_id -> count

        # Content structures
        self.term_to_idx = {}                      # term -> column index
        self.item_id_to_idx = {}                   # item_id -> row index
        self.idx_to_item_id = {}                   # row index -> item_id
        self.item_matrix = np.zeros((0, 0), dtype=np.float32)  # dense item vectors
        self.inverted_index = defaultdict(set)     # term -> set(item_idx)
        self.avg_vector = None

        # User content vectors and caches
        self.user_content_vectors = {}             # user_id -> dense vector
        self.user_ratings_cache = {}               # user_id -> (item_idx_array, rating_array)

        # Precomputed neighbors
        self.item_neighbors = {}                   # item_idx -> list of (neighbor_idx, sim)

        # Config
        self.alpha = alpha
        self.k = k_neighbors
        self.hybrid_mode = hybrid_mode
        self.similarity = similarity
        self.candidate_pool_size = candidate_pool_size
        self.precompute_knn = precompute_knn

        # Internal flags
        self._knn_built = False

    # ---------------- FIT COLLABORATIVE ----------------
    def fit_collaborative(self, ratings_df):
        """
        Build bipartite adjacency lists and popularity counts.
        ratings_df expected to have columns: userId, movieId, rating
        """
        self.user_item_graph.clear()
        self.item_users.clear()
        self.item_popularity.clear()
        self.user_ratings_cache.clear()

        if ratings_df is None or len(ratings_df) == 0:
            return

        for _, row in ratings_df.iterrows():
            try:
                u = int(row['userId'])
                i = int(row['movieId'])
                r = float(row['rating'])
            except Exception:
                continue
            self.user_item_graph[u].append((i, r))
            self.item_users[i].append(u)
            self.item_popularity[i] += 1

        # Build id-based cache for now; indices conversion happens after fit_content
        for u, pairs in self.user_item_graph.items():
            item_ids = np.array([p[0] for p in pairs], dtype=np.int32)
            ratings = np.array([p[1] for p in pairs], dtype=np.float32)
            self.user_ratings_cache[u] = (item_ids, ratings)

        # Mark neighbors stale
        self._knn_built = False

    # ---------------- FIT CONTENT ----------------
    def fit_content(self, movies_df):
        """
        Build item vectors (TF-IDF-like on genres + numeric features).
        movies_df expected to have at least: movieId, title, genres (genres pipe-separated).
        Optional numeric columns: budget, runtime, release_year, avg_critic_score
        """
        # Reset content structures
        self.term_to_idx.clear()
        self.item_id_to_idx.clear()
        self.idx_to_item_id.clear()
        self.inverted_index.clear()
        self.item_matrix = np.zeros((0, 0), dtype=np.float32)
        self.avg_vector = None

        if movies_df is None or len(movies_df) == 0:
            # still convert any existing user caches to empty indices
            for u in list(self.user_ratings_cache.keys()):
                self.user_ratings_cache[u] = (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
            return

        # Step 1: collect term frequencies and doc counts
        tf_store = {}
        term_doc_count = defaultdict(int)
        item_ids = []
        for _, row in movies_df.iterrows():
            try:
                m_id = int(row['movieId'])
            except Exception:
                continue
            item_ids.append(m_id)
            genres = row.get('genres', '')
            if not isinstance(genres, str):
                genres = ''
            terms = [t.strip().lower() for t in genres.split('|') if t.strip()]
            tf = defaultdict(int)
            for t in terms:
                tf[t] += 1
            tf_store[m_id] = tf
            for t in set(terms):
                term_doc_count[t] += 1

        n_docs = max(1, len(item_ids))

        # Step 2: build sparse vectors including numeric features
        sparse_vectors = {}
        for _, row in movies_df.iterrows():
            try:
                m_id = int(row['movieId'])
            except Exception:
                continue
            vec = {}
            # TF-IDF for genres
            for term, count in tf_store.get(m_id, {}).items():
                idf = math.log((n_docs + 1) / (1 + term_doc_count[term]))
                vec[term] = count * idf
            # numeric features with safe defaults
            try:
                vec['budget'] = float(row.get('budget', 0.0)) / 1e7
            except Exception:
                vec['budget'] = 0.0
            try:
                vec['runtime'] = float(row.get('runtime', 0.0)) / 100.0
            except Exception:
                vec['runtime'] = 0.0
            try:
                vec['release_year'] = (float(row.get('release_year', 1990.0)) - 1990.0) / 35.0
            except Exception:
                vec['release_year'] = 0.0
            try:
                vec['critic_score'] = float(row.get('avg_critic_score', 0.0)) / 10.0
            except Exception:
                vec['critic_score'] = 0.0
            sparse_vectors[m_id] = vec

        # Step 3: build vocabulary
        all_terms = set()
        for v in sparse_vectors.values():
            all_terms.update(v.keys())
        self.term_to_idx = {term: idx for idx, term in enumerate(sorted(all_terms))}
        dim = len(self.term_to_idx)

        # Step 4: build dense matrix and mappings
        n_items = len(sparse_vectors)
        if n_items == 0:
            self.item_matrix = np.zeros((0, dim), dtype=np.float32)
            self.avg_vector = np.zeros((dim,), dtype=np.float32)
            return

        self.item_matrix = np.zeros((n_items, dim), dtype=np.float32)
        for row_idx, (m_id, vec) in enumerate(sorted(sparse_vectors.items())):
            self.item_id_to_idx[m_id] = row_idx
            self.idx_to_item_id[row_idx] = m_id
            for term, val in vec.items():
                col = self.term_to_idx[term]
                self.item_matrix[row_idx, col] = float(val)
            for term in vec.keys():
                self.inverted_index[term].add(row_idx)

        # Step 5: average vector
        if self.item_matrix.size > 0:
            self.avg_vector = np.mean(self.item_matrix, axis=0)
        else:
            self.avg_vector = np.zeros((dim,), dtype=np.float32)

        # Step 6: convert user_ratings_cache item ids -> indices where possible
        for u, (item_ids_arr, ratings_arr) in list(self.user_ratings_cache.items()):
            idxs = []
            ratings_mapped = []
            for iid, r in zip(item_ids_arr.tolist(), ratings_arr.tolist()):
                if iid in self.item_id_to_idx:
                    idxs.append(self.item_id_to_idx[iid])
                    ratings_mapped.append(r)
            if len(idxs) == 0:
                self.user_ratings_cache[u] = (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
            else:
                self.user_ratings_cache[u] = (np.array(idxs, dtype=np.int32), np.array(ratings_mapped, dtype=np.float32))

        # Step 7: precompute user content vectors
        self.user_content_vectors = {}
        for u in self.user_item_graph.keys():
            self._build_user_content_vector(u)

        # Step 8: optionally precompute neighbors
        if self.precompute_knn:
            self.build_item_neighbors(k=self.k)
            self._knn_built = True

    # ---------------- BUILD USER CONTENT VECTOR ----------------
    def _build_user_content_vector(self, user_id):
        """
        Weighted average of item vectors rated by user.
        """
        if self.item_matrix.size == 0:
            vec = np.zeros((0,), dtype=np.float32)
            self.user_content_vectors[user_id] = vec
            self.user_ratings_cache.setdefault(user_id, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
            return vec

        pairs = self.user_item_graph.get(user_id, [])
        idxs = []
        ratings = []
        for (item_id, rating) in pairs:
            if item_id in self.item_id_to_idx:
                idxs.append(self.item_id_to_idx[item_id])
                ratings.append(float(rating))
        if len(idxs) == 0:
            vec = np.zeros(self.item_matrix.shape[1], dtype=np.float32)
            self.user_content_vectors[user_id] = vec
            self.user_ratings_cache.setdefault(user_id, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
            return vec

        idxs = np.array(idxs, dtype=np.int32)
        ratings = np.array(ratings, dtype=np.float32)
        item_vecs = self.item_matrix[idxs]  # (m, dim)
        weighted = (ratings[:, None] * item_vecs).sum(axis=0)
        denom = ratings.sum() if ratings.sum() != 0 else 1.0
        vec = (weighted / denom).astype(np.float32)
        self.user_content_vectors[user_id] = vec
        self.user_ratings_cache[user_id] = (idxs, ratings)
        return vec

    # ---------------- SIMILARITY HELPERS ----------------
    def _cosine_sim(self, A, B):
        """
        Compute cosine similarity between rows of A and rows of B.
        A: (n_a, dim) or (dim,)
        B: (n_b, dim) or (dim,)
        Returns matrix (n_a, n_b)
        """
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        A_norm = np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = np.linalg.norm(B, axis=1, keepdims=True)
        A_norm[A_norm == 0] = 1.0
        B_norm[B_norm == 0] = 1.0
        sims = (A @ B.T) / (A_norm * B_norm.T)
        return sims

    def _pearson_sim(self, A, B):
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        A_centered = A - A.mean(axis=1, keepdims=True)
        B_centered = B - B.mean(axis=1, keepdims=True)
        return self._cosine_sim(A_centered, B_centered)

    def _similarity_matrix(self, A, B):
        if self.similarity == "cosine":
            return self._cosine_sim(A, B)
        else:
            return self._pearson_sim(A, B)

    # ---------------- BUILD ITEM NEIGHBORS ----------------
    def build_item_neighbors(self, k=None):
        """
        Build top-k neighbors for each item using cosine/pearson similarity.
        Stores item_neighbors as item_idx -> list of (neighbor_idx, sim).
        """
        if k is None:
            k = self.k
        if self.item_matrix.size == 0:
            self.item_neighbors = {}
            self._knn_built = True
            return

        I = self.item_matrix.shape[0]
        # compute normalized rows for cosine to speed up
        if self.similarity == "cosine":
            norms = np.linalg.norm(self.item_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = self.item_matrix / norms
            sim_mat = normed @ normed.T
        else:
            centered = self.item_matrix - self.item_matrix.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = centered / norms
            sim_mat = normed @ normed.T

        np.fill_diagonal(sim_mat, 0.0)
        self.item_neighbors = {}
        k_eff = min(k, I - 1) if I > 1 else 0
        for i in range(I):
            row = sim_mat[i]
            if k_eff > 0:
                top_idx = np.argpartition(-row, k_eff)[:k_eff]
                top_idx_sorted = top_idx[np.argsort(-row[top_idx])]
            else:
                top_idx_sorted = np.argsort(-row)
            neighbors = [(int(j), float(row[j])) for j in top_idx_sorted if row[j] > 0]
            self.item_neighbors[i] = neighbors
        self._knn_built = True

    # ---------------- CANDIDATE GENERATION ----------------
    def _generate_candidates(self, user_id):
        """
        Candidate generation using:
          - inverted index from user's top terms
          - neighbors of user's rated items (precomputed)
          - two-hop expansion (user->item->user->item)
          - popularity fallback
        """
        if self.item_matrix.size == 0:
            return np.array([], dtype=np.int32)

        candidate_set = set()

        # Use user content vector top terms
        user_vec = self.user_content_vectors.get(user_id)
        if user_vec is not None and user_vec.size > 0 and not np.all(user_vec == 0):
            top_term_count = min(8, len(self.term_to_idx))
            if top_term_count > 0:
                top_term_idxs = np.argsort(-np.abs(user_vec))[:top_term_count]
                idx_to_term = {v: k for k, v in self.term_to_idx.items()}
                for t_idx in top_term_idxs:
                    term = idx_to_term.get(int(t_idx))
                    if term is None:
                        continue
                    candidate_set.update(self.inverted_index.get(term, set()))
                    if len(candidate_set) >= self.candidate_pool_size:
                        break

        # Neighbors of rated items
        rated_idxs, _ = self.user_ratings_cache.get(user_id, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
        for ridx in rated_idxs.tolist():
            neighs = self.item_neighbors.get(int(ridx), [])
            for n_idx, _ in neighs:
                candidate_set.add(int(n_idx))
                if len(candidate_set) >= self.candidate_pool_size:
                    break
            if len(candidate_set) >= self.candidate_pool_size:
                break

        # Two-hop expansion: user -> items -> users -> items
        # limited to small budget to avoid explosion
        two_hop_limit = max(50, self.candidate_pool_size // 4)
        first_items = [iid for iid, _ in self.user_item_graph.get(user_id, []) if iid in self.item_id_to_idx]
        first_idxs = [self.item_id_to_idx[iid] for iid in first_items]
        second_users = set()
        for idx in first_idxs:
            item_id = self.idx_to_item_id.get(idx)
            if item_id is None:
                continue
            for u in self.item_users.get(item_id, []):
                second_users.add(u)
                if len(second_users) >= two_hop_limit:
                    break
            if len(second_users) >= two_hop_limit:
                break
        for u in second_users:
            for iid, _ in self.user_item_graph.get(u, []):
                if iid in self.item_id_to_idx:
                    candidate_set.add(self.item_id_to_idx[iid])
                if len(candidate_set) >= self.candidate_pool_size:
                    break
            if len(candidate_set) >= self.candidate_pool_size:
                break

        # Fill with popular items if needed
        if len(candidate_set) < self.candidate_pool_size:
            for iid, _ in self.item_popularity.most_common(self.candidate_pool_size * 2):
                if iid in self.item_id_to_idx:
                    candidate_set.add(self.item_id_to_idx[iid])
                if len(candidate_set) >= self.candidate_pool_size:
                    break

        # Remove items user already rated
        candidate_set.difference_update(set(rated_idxs.tolist()))

        return np.array(sorted(candidate_set), dtype=np.int32)

    # ---------------- CF SCORING (USING NEIGHBORS) ----------------
    def _score_cf_using_neighbors(self, user_id, candidate_idxs):
        """
        For each candidate item, use precomputed neighbors to compute weighted score.
        """
        if candidate_idxs.size == 0:
            return np.array([], dtype=np.float32)

        rated_idxs, ratings = self.user_ratings_cache.get(user_id, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
        if rated_idxs.size == 0:
            return np.zeros(candidate_idxs.shape[0], dtype=np.float32)

        rated_map = {int(idx): float(r) for idx, r in zip(rated_idxs.tolist(), ratings.tolist())}
        scores = np.zeros(candidate_idxs.shape[0], dtype=np.float32)
        for pos, cidx in enumerate(candidate_idxs.tolist()):
            neighs = self.item_neighbors.get(int(cidx), [])
            num = 0.0
            den = 0.0
            for n_idx, sim in neighs:
                if int(n_idx) in rated_map:
                    r = rated_map[int(n_idx)]
                    num += sim * r
                    den += sim
            if den > 0:
                scores[pos] = num / den
            else:
                scores[pos] = 0.0
        return scores

    # ---------------- CB SCORING ----------------
    def _score_cb(self, user_id, candidate_idxs):
        if candidate_idxs.size == 0:
            return np.array([], dtype=np.float32)
        user_vec = self.user_content_vectors.get(user_id)
        if user_vec is None:
            user_vec = self._build_user_content_vector(user_id)
        if user_vec.size == 0:
            return np.zeros(candidate_idxs.shape[0], dtype=np.float32)
        cand_vecs = self.item_matrix[candidate_idxs]
        sims = self._similarity_matrix(cand_vecs, user_vec).reshape(-1)
        sims = np.maximum(sims, 0.0)
        return sims.astype(np.float32)

    # ---------------- NORMALIZATION ----------------
    def _normalize_scores(self, scores):
        if scores.size == 0:
            return scores
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        if np.isclose(min_v, max_v):
            return np.full_like(scores, 0.5, dtype=np.float32)
        return ((scores - min_v) / (max_v - min_v)).astype(np.float32)

    # ---------------- ALPHA ----------------
    def _get_alpha(self, user_id):
        n_ratings = len(self.user_item_graph.get(user_id, []))
        if n_ratings < 3:
            return 0.0
        elif n_ratings < 20:
            return 0.6
        else:
            return 0.8

    # ---------------- RECOMMEND ----------------
    def get_recommendations(self, user_id, n=10):
        """
        Return top-n item_ids for user_id.
        Ensures a fallback to popularity/content if necessary.
        """
        # If no items known, return empty list
        if self.item_matrix.size == 0:
            return []

        # Ensure user vector exists
        if user_id not in self.user_content_vectors:
            self._build_user_content_vector(user_id)

        # Ensure neighbors built
        if not self._knn_built:
            try:
                self.build_item_neighbors(k=self.k)
            except Exception:
                # fallback: empty neighbors
                self.item_neighbors = {}
            self._knn_built = True

        candidate_idxs = self._generate_candidates(user_id)
        if candidate_idxs.size == 0:
            # fallback: top popular items not seen by user
            recs = []
            seen = set([iid for iid, _ in self.user_item_graph.get(user_id, [])])
            for iid, _ in self.item_popularity.most_common(n * 5):
                if iid not in seen:
                    recs.append(iid)
                if len(recs) >= n:
                    break
            return recs[:n]

        # Compute CB and CF scores
        cb_scores = self._score_cb(user_id, candidate_idxs)
        cf_scores = self._score_cf_using_neighbors(user_id, candidate_idxs)

        # Normalize
        cb_norm = self._normalize_scores(cb_scores)
        cf_norm = self._normalize_scores(cf_scores)

        # Hybrid combination
        is_cold = len(self.user_item_graph.get(user_id, [])) == 0
        alpha = 0.0 if is_cold else self._get_alpha(user_id)

        if self.hybrid_mode == "weighted":
            hybrid_scores = alpha * cf_norm + (1.0 - alpha) * cb_norm
        elif self.hybrid_mode == "switching":
            hybrid_scores = cf_norm if alpha >= 0.5 else cb_norm
        else:
            hybrid_scores = alpha * cf_norm + (1.0 - alpha) * cb_norm

        if hybrid_scores.size == 0:
            return []

        top_k = min(n, hybrid_scores.size)
        top_idxs_local = np.argpartition(-hybrid_scores, top_k - 1)[:top_k]
        top_sorted_local = top_idxs_local[np.argsort(-hybrid_scores[top_idxs_local])]
        top_candidate_idxs = candidate_idxs[top_sorted_local]
        top_item_ids = [self.idx_to_item_id[int(idx)] for idx in top_candidate_idxs]
        return top_item_ids

    # ---------------- ONLINE UPDATES ----------------
    def add_rating(self, user_id, item_id, rating):
        """
        Add rating and update caches. Marks neighbors stale for lazy rebuild.
        """
        self.user_item_graph[user_id].append((item_id, float(rating)))
        self.item_users[item_id].append(user_id)
        self.item_popularity[item_id] += 1

        # Update user_ratings_cache: convert to indices if possible
        if item_id in self.item_id_to_idx:
            item_idx = self.item_id_to_idx[item_id]
            if user_id in self.user_ratings_cache:
                idxs, ratings = self.user_ratings_cache[user_id]
                idxs = np.append(idxs, item_idx)
                ratings = np.append(ratings, float(rating))
                self.user_ratings_cache[user_id] = (idxs, ratings)
            else:
                self.user_ratings_cache[user_id] = (np.array([item_idx], dtype=np.int32),
                                                   np.array([float(rating)], dtype=np.float32))
        else:
            self.user_ratings_cache.setdefault(user_id, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))

        # Recompute user content vector
        self._build_user_content_vector(user_id)
        self._knn_built = False

    def update_item_vector(self, item_id, new_sparse_vector):
        """
        Update or add an item vector (sparse dict). Marks neighbors stale.
        """
        # Add new terms to vocabulary if needed
        new_terms = set(new_sparse_vector.keys()) - set(self.term_to_idx.keys())
        if new_terms:
            current_dim = len(self.term_to_idx)
            for t in sorted(new_terms):
                self.term_to_idx[t] = current_dim
                current_dim += 1
            # expand item_matrix columns
            if self.item_matrix.size == 0:
                self.item_matrix = np.zeros((0, current_dim), dtype=np.float32)
            else:
                n_items, old_dim = self.item_matrix.shape
                new_mat = np.zeros((n_items, current_dim), dtype=np.float32)
                new_mat[:, :old_dim] = self.item_matrix
                self.item_matrix = new_mat

        dim = len(self.term_to_idx)
        vec = np.zeros((dim,), dtype=np.float32)
        for term, val in new_sparse_vector.items():
            if term in self.term_to_idx:
                vec[self.term_to_idx[term]] = float(val)

        if item_id in self.item_id_to_idx:
            idx = self.item_id_to_idx[item_id]
            # expand columns if needed
            if self.item_matrix.shape[1] < dim:
                n_items = self.item_matrix.shape[0]
                new_mat = np.zeros((n_items, dim), dtype=np.float32)
                new_mat[:, :self.item_matrix.shape[1]] = self.item_matrix
                self.item_matrix = new_mat
            self.item_matrix[idx] = vec
        else:
            new_idx = self.item_matrix.shape[0]
            if self.item_matrix.size == 0:
                self.item_matrix = vec.reshape(1, -1)
            else:
                # expand columns if needed
                if self.item_matrix.shape[1] < dim:
                    n_items = self.item_matrix.shape[0]
                    new_mat = np.zeros((n_items, dim), dtype=np.float32)
                    new_mat[:, :self.item_matrix.shape[1]] = self.item_matrix
                    self.item_matrix = new_mat
                self.item_matrix = np.vstack([self.item_matrix, vec])
            self.item_id_to_idx[item_id] = new_idx
            self.idx_to_item_id[new_idx] = item_id

        # update inverted index
        for term, col in self.term_to_idx.items():
            if vec[col] != 0.0:
                self.inverted_index[term].add(self.item_id_to_idx[item_id])
            else:
                if self.item_id_to_idx[item_id] in self.inverted_index.get(term, set()):
                    self.inverted_index[term].discard(self.item_id_to_idx[item_id])

        # recompute avg vector
        if self.item_matrix.shape[0] > 0:
            self.avg_vector = np.mean(self.item_matrix, axis=0)
        else:
            self.avg_vector = np.zeros((dim,), dtype=np.float32)

        # mark neighbors stale
        self._knn_built = False

    # ---------------- COMPLEXITY ANALYSIS ----------------
    def complexity_analysis(self):
        print("Collaborative Filtering:")
        print("- Build bipartite graph: O(R) time, O(R) space (R = #ratings)")
        print("- CF scoring (using precomputed neighbors): O(C * k) per user (C=candidate pool, k=neighbors)")

        print("\nContent-Based Filtering:")
        print("- TF-IDF vectorization: O(I * T) time (I = items, T = avg terms per item)")
        print("- Dense item matrix: O(I * D) space (D = feature dim)")

        print("\nHybrid Recommendation:")
        print("- Candidate generation: O(C) (inverted index + two-hop + neighbors + popularity)")
        print("- Scoring per user: O(C * (D + k)) with vectorized CB and neighbor-based CF")
        print("- Space: O(I*D + U*D + I*k) for item/user vectors and neighbor lists")

        print("\nGraph-specific operations:")
        print("- Bipartite projection (item graph): O(sum_u r_u^2) time")
        print("- Top-k neighbor lists: stored as O(I * k) space")

        print("\nData Structures Used:")
        print("- Graphs: bipartite adjacency lists (user_item_graph, item_users)")
        print("- Hash Tables: dicts for mappings and inverted index")
        print("- Priority selection: argpartition/heapq for top-N selection")

# This module implements a robust hybrid recommender engine