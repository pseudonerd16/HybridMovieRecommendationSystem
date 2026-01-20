# Hybrid Movie Recommendation System - Milestone 2
**Student Name:** Anurag Roychowdhury (ZDA25M004)  
**Student Name:** Sreejita Roy (ZDA25M008) 
**Course:** Z5007: Programming and Data Structures  
**Institution:** IIT Madras Zanzibar  

##  Project Overview
This project implements a high-performance Hybrid Recommendation System from scratch. It combines **Collaborative Filtering** (using Bipartite Graphs/ Adjacency lists) and **Content-Based Filtering** (using high-dimensional feature vectors) to provide personalized movie suggestions.

##  Data Structures Implemented (Milestone 2 Requirements)
1.  **Bipartite Graph (Adjacency List):** Stores User-Movie interactions for Collaborative Filtering.
2.  **Hash Tables:** Used for "O(1)" retrieval of movie metadata and pre-computed popularity scores.
3.  **Max-Heap:** Implemented for the Ranking Module to ensure Top-N selection is computationally efficient ("O(M \log N)").

##  Folder Structure
- `src/`: Contains the core logic.
    - `recommender_engine.py`: The main algorithm class.
    - `generate_data.py`: Script to generate the 1M sample dataset.
    - `evaluation_metrics.py`: Script for Precision, Recall, and NDCG math.
- `data/`: Directory where genrated CSV files are stored.
- `main.py`: The execution script for demonstration.

##  How to Run
### 1. Environment Setup
Ensure you have Python 3.8+ installed. You will need `numpy` and `pandas`.
pip install numpy pandas
### 2. Run the data generator to create 1,000,000 ratings and 20+ features.
python src/generate_data.py
### 3. Execute the main script to build the graph, train the engine, and see sample recommendations.
python main.py
```bash