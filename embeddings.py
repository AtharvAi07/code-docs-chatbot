"""
Generate embeddings for code/documentation chunks.
Uses HuggingFace sentence-transformers (free, no API key required).
Converts text into vectors for semantic search.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        print(f"Embedding {len(chunks)} chunks...")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embed_batch(texts, batch_size=32)
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        print(f"Embedded {len(chunks)} chunks")
        return chunks


def prepare_embeddings(files: List[Dict], chunker) -> List[Dict]:
    all_chunks = []
    for file in files:
        file_chunks = chunker.chunk_file(
            file['content'],
            file['name'],
            file['type']
        )
        for chunk in file_chunks:
            chunk['source_url'] = file.get('url', '')
            chunk['repo'] = file.get('repo', 'unknown')
        all_chunks.extend(file_chunks)
    
    generator = EmbeddingGenerator()
    all_chunks = generator.embed_chunks(all_chunks)
    
    return all_chunks


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    test_text = "def hello_world(): print('Hello')"
    embedding = generator.embed_text(test_text)
    
    print(f"Test embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    test_texts = [
        "Flask is a lightweight web framework",
        "Django is a full-featured web framework",
        "FastAPI is a modern async framework"
    ]
    batch_embeddings = generator.embed_batch(test_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
