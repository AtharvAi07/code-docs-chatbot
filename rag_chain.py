"""
RAG chain: combines vector search + Groq LLM to answer questions.
This is the core logic that makes the chatbot actually useful.
"""

import os
from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv

from embeddings import EmbeddingGenerator
from vector_db import VectorDB

load_dotenv()


class RAGChain:
    """Combines retrieval (vector search) with generation (Groq LLM)."""
    
    def __init__(self, vector_db: VectorDB, embedding_generator: EmbeddingGenerator):
        self.vector_db = vector_db
        self.embedder = embedding_generator
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Make sure your .env file has: "
                "GROQ_API_KEY=your_key_here"
            )
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_db.search(query_embedding, top_k=top_k)
        return results
    
    def build_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant documentation found."
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = f"{chunk.get('repo', 'unknown')}/{chunk.get('filename', 'unknown')}"
            context_parts.append(
                f"[Source {i+1}: {source}]\n{chunk['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a helpful coding assistant that answers questions about "
            "code and documentation. Use ONLY the provided context to answer. "
            "If the context doesn't contain enough information, say so clearly. "
            "Always mention which source(s) you used. Be concise and technical."
        )
        
        user_prompt = f"""Context from documentation/code:
{context}

Question: {query}

Answer the question using the context above. Cite sources like [Source 1] when relevant."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def answer(self, query: str, top_k: int = 5) -> Dict:
        chunks = self.retrieve(query, top_k=top_k)
        
        if not chunks:
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "sources": [],
                "chunks_used": []
            }
        
        context = self.build_context(chunks)
        answer_text = self.generate(query, context)
        
        sources = list(set(
            f"{chunk.get('repo', 'unknown')}/{chunk.get('filename', 'unknown')}"
            for chunk in chunks
        ))
        
        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": chunks
        }


if __name__ == "__main__":
    print("=== Testing RAG Chain ===")
    
    generator = EmbeddingGenerator()
    
    test_chunks = [
        {"content": "Flask is a lightweight Python web framework. Install it with pip install flask.", 
         "filename": "flask_intro.md", "repo": "flask"},
        {"content": "Flask routes are defined using the @app.route decorator. Example: @app.route(/home)", 
         "filename": "flask_routes.md", "repo": "flask"},
        {"content": "Django follows the model-view-template (MVT) architectural pattern.", 
         "filename": "django_intro.md", "repo": "django"},
        {"content": "FastAPI uses Python type hints for automatic request validation and API docs.", 
         "filename": "fastapi_intro.md", "repo": "fastapi"},
    ]
    
    texts = [chunk['content'] for chunk in test_chunks]
    embeddings = generator.embed_batch(texts)
    for i, chunk in enumerate(test_chunks):
        chunk['embedding'] = embeddings[i]
    
    db = VectorDB(dimension=384)
    db.add_chunks(test_chunks)
    
    rag = RAGChain(vector_db=db, embedding_generator=generator)
    
    query = "How do I define a route in Flask?"
    print(f"Question: {query}")
    
    result = rag.answer(query, top_k=2)
    
    print(f"Answer: {result['answer']}")
    print(f"Sources used: {result['sources']}")
