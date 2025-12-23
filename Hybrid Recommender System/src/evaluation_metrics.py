import numpy as np

def calculate_metrics(recommended_ids, actual_ids):
    rec_set = set(recommended_ids)
    act_set = set(actual_ids)
    
    # Precision @ N
    precision = len(rec_set & act_set) / len(recommended_ids) if recommended_ids else 0
    # Recall @ N
    recall = len(rec_set & act_set) / len(actual_ids) if actual_ids else 0
    
    # NDCG calculation
    dcg = 0
    for i, m_id in enumerate(recommended_ids):
        if m_id in act_set:
            dcg += 1 / np.log2(i + 2)
    
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(act_set), len(recommended_ids))))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return {"Precision": precision, "Recall": recall, "NDCG": ndcg}