"""
Retrieve the top-k most relevant document chunks for a query, using
cosine similarity over the TF-IDF vectors built by ingest.py.
"""

import json
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"


class Retriever:
    def __init__(self):
        self.vectorizer = joblib.load(INDEX_DIR / "vectorizer.joblib")
        self.matrix = joblib.load(INDEX_DIR / "tfidf_matrix.joblib")
        with open(INDEX_DIR / "chunks.json") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05):
        """
        Return the top_k chunks most similar to the query, each with its
        similarity score. Chunks below min_score are dropped — a low max
        score across the board usually means the answer isn't in the
        knowledge base at all, which chat.py uses to say "I don't know"
        instead of forcing an answer from irrelevant context.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in ranked:
            if scores[i] >= min_score:
                results.append({**self.chunks[i], "score": float(scores[i])})
        return results


if __name__ == "__main__":
    retriever = Retriever()
    test_queries = [
        "What is the difference between precision and recall?",
        "How does k-fold cross-validation work?",
        "What's a good recipe for chocolate cake?",  # should retrieve nothing useful
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q)
        if not results:
            print("  (no relevant chunks found)")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']} (chunk {r['chunk_id']})")
