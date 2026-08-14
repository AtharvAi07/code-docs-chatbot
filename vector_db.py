"""
Vector database for storing and searching embeddings.
Uses FAISS (Facebook AI Similarity Search) - free, fast, runs locally.
"""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from pathlib import Path


class VectorDB:
    """Store and search embeddings using FAISS."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            print("No chunks to add.")
            return
        
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        print(f"Added {len(chunks)} chunks. Total in database: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if len(self.chunks) == 0:
            print("Database is empty. Add chunks first.")
            return []
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['distance'] = float(distances[0][i])
                chunk.pop('embedding', None)
                results.append(chunk)
        
        return results
    
    def save(self, filepath: str = "vector_db"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{filepath}.index")
        
        chunks_to_save = []
        for chunk in self.chunks:
            chunk_copy = chunk.copy()
            chunk_copy.pop('embedding', None)
            chunks_to_save.append(chunk_copy)
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(chunks_to_save, f)
        
        print(f"Saved database to {filepath}.index and {filepath}.pkl")
    
    def load(self, filepath: str = "vector_db"):
        index_path = f"{filepath}.index"
        pkl_path = f"{filepath}.pkl"
        
        if not Path(index_path).exists() or not Path(pkl_path).exists():
            print(f"No saved database found at {filepath}")
            return False
        
        self.index = faiss.read_index(index_path)
        
        with open(pkl_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded database: {len(self.chunks)} chunks")
        return True
    
    def stats(self):
        print(f"Vector DB Stats:")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  Embedding dimension: {self.dimension}")
        if self.chunks:
            repos = set(chunk.get('repo', 'unknown') for chunk in self.chunks)
            print(f"  Repos indexed: {repos}")


if __name__ == "__main__":
    from embeddings import EmbeddingGenerator
    
    print("=== Testing Vector Database ===")
    
    generator = EmbeddingGenerator()
    
    test_chunks = [
        {"content": "Flask is a lightweight Python web framework", "filename": "flask_intro.md", "repo": "flask"},
        {"content": "Django follows the model-view-template pattern", "filename": "django_intro.md", "repo": "django"},
        {"content": "FastAPI uses Python type hints for validation", "filename": "fastapi_intro.md", "repo": "fastapi"},
        {"content": "Flask routes are defined using @app.route decorator", "filename": "flask_routes.md", "repo": "flask"},
    ]
    
    texts = [chunk['content'] for chunk in test_chunks]
    embeddings = generator.embed_batch(texts)
    for i, chunk in enumerate(test_chunks):
        chunk['embedding'] = embeddings[i]
    
    db = VectorDB(dimension=384)
    db.add_chunks(test_chunks)
    db.stats()
    
    print("=== Testing Search ===")
    query = "How do I create a route in Flask?"
    print(f"Query: {query}")
    
    query_embedding = generator.embed_text(query)
    results = db.search(query_embedding, top_k=2)
    
    print("Top results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['filename']} (distance: {result['distance']:.4f})")
        print(f"   Content: {result['content']}")
    
    print("=== Testing Save/Load ===")
    db.save("test_db")
    
    new_db = VectorDB(dimension=384)
    new_db.load("test_db")
    new_db.stats()
