"""
Smart chunking strategy for code and documentation.
Preserves context and relationships in code.
"""

from typing import List, Dict
import re

class SmartChunker:
    """Intelligently chunk code and documentation."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_file(self, content: str, filename: str, file_type: str) -> List[Dict]:
        if file_type == 'code':
            return self._chunk_code(content, filename)
        else:
            return self._chunk_documentation(content, filename)
    
    def _chunk_code(self, content: str, filename: str) -> List[Dict]:
        if filename.endswith('.py'):
            return self._chunk_python(content, filename)
        else:
            return self._chunk_generic(content, filename)
    
    def _chunk_python(self, content: str, filename: str) -> List[Dict]:
        chunks = []
        lines = content.split("\n")
        
        current_chunk = []
        current_context = ""
        
        for i, line in enumerate(lines):
            if line.strip().startswith(("def ", "class ", "async def")):
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "filename": filename,
                        "start_line": i - len(current_chunk),
                        "end_line": i,
                        "context": current_context,
                    })
                
                current_context = line.strip()[:50]
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
                if len("\n".join(current_chunk)) > self.chunk_size:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "filename": filename,
                        "start_line": i - len(current_chunk),
                        "end_line": i,
                        "context": current_context,
                    })
                    current_chunk = []
        
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "filename": filename,
                "start_line": len(lines) - len(current_chunk),
                "end_line": len(lines),
                "context": current_context,
            })
        
        return chunks
    
    def _chunk_generic(self, content: str, filename: str) -> List[Dict]:
        chunks = []
        words = content.split()
        
        current_chunk = []
        for i, word in enumerate(words):
            current_chunk.append(word)
            
            if len(" ".join(current_chunk)) > self.chunk_size:
                chunks.append({
                    "content": " ".join(current_chunk),
                    "filename": filename,
                    "start_line": i - len(current_chunk),
                    "end_line": i,
                    "context": filename,
                })
                current_chunk = []
        
        if current_chunk:
            chunks.append({
                "content": " ".join(current_chunk),
                "filename": filename,
                "start_line": len(words) - len(current_chunk),
                "end_line": len(words),
                "context": filename,
            })
        
        return chunks
    
    def _chunk_documentation(self, content: str, filename: str) -> List[Dict]:
        chunks = []
        sections = re.split(r"^(#{1,6}\s+.*?)$", content, flags=re.MULTILINE)
        
        current_chunk = []
        current_header = "Introduction"
        
        for section in sections:
            if section.strip().startswith("#"):
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "filename": filename,
                        "context": current_header,
                    })
                current_header = section.strip()
                current_chunk = [section]
            else:
                current_chunk.append(section)
                
                if len("\n".join(current_chunk)) > self.chunk_size:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "filename": filename,
                        "context": current_header,
                    })
                    current_chunk = []
        
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "filename": filename,
                "context": current_header,
            })
        
        return chunks
