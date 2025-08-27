import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_score(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    Returns similarity score in percentage.
    """
    try:
        # Reshape for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)

        # Compute cosine similarity
        sim = cosine_similarity(emb1, emb2)[0][0]

        # Convert to percentage
        return float(sim * 100)
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return 0.0
