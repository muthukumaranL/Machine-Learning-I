"""Daily AI/ML Mini Project — 2026-09-04

Tiny retrieval engine using TF-IDF + cosine similarity.
This demonstrates the retrieval step that sits underneath many search and RAG systems,
without requiring an API key or external dataset.

Run:
    pip install scikit-learn
    python daily-ai-ml/2026-09-04_tfidf_search.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float


class TinyRetriever:
    def __init__(self, documents: Sequence[str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty")

        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.document_matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            SearchResult(text=self.documents[index], score=float(scores[index]))
            for index in ranked_indices
        ]


def main() -> None:
    documents = [
        "Retrieval augmented generation combines search with language model generation.",
        "PyTorch is widely used to train and deploy deep learning models.",
        "Vector databases store embeddings for similarity search.",
        "FastAPI is useful for serving machine learning models as web APIs.",
        "Model evaluation should use metrics that match the business objective.",
    ]

    retriever = TinyRetriever(documents)
    query = "How does retrieval work in RAG?"

    print(f"Query: {query}\n")
    for rank, result in enumerate(retriever.search(query), start=1):
        print(f"{rank}. {result.score:.3f} — {result.text}")


if __name__ == "__main__":
    main()
