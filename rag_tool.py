import json
import os
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

class RAGTool:
    """
    RAG Tool for PDF processing with embedding caching
    - Loads PDF and creates embeddings
    - Saves/loads embeddings from JSON
    - Provides semantic search for queries
    """
    
    def __init__(self, api_key: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize RAG Tool
        
        Args:
            api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = []
        
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load and extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from PDF
        """
        
        print(f"📄 Loading PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(f"✅ Loaded {len(reader.pages)} pages")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        print(f"✂️  Chunking text (chunk_size={chunk_size}, overlap={overlap})")
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embeddings
        """
        print(f"🔄 Creating embeddings for {len(texts)} chunks...")
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        print("✅ Embeddings created")
        return embeddings
    
    def save_embeddings(self, pdf_path: str, embeddings_dir: str = "tmp/embeddings"):
        """
        Save chunks and embeddings to JSON
        
        Args:
            pdf_path: Original PDF path (used for naming)
            embeddings_dir: Directory to save embeddings
        """
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Create filename based on PDF name
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path = os.path.join(embeddings_dir, f"{pdf_name}_embeddings.json")
        
        data = {
            "pdf_path": pdf_path,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "model": self.embedding_model
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Embeddings saved to: {json_path}")
        return json_path
    
    def load_embeddings(self, pdf_path: str, embeddings_dir: str = "tmp/embeddings") -> bool:
        """
        Load embeddings from JSON if exists
        
        Args:
            pdf_path: Original PDF path
            embeddings_dir: Directory where embeddings are saved
            
        Returns:
            True if loaded successfully, False otherwise
        """
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path = os.path.join(embeddings_dir, f"{pdf_name}_embeddings.json")
        
        if not os.path.exists(json_path):
            print(f"❌ No cached embeddings found for {pdf_name}")
            return False
        
        print(f"📂 Loading cached embeddings from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        
        print(f"✅ Loaded {len(self.chunks)} chunks from cache")
        return True
    
    def process_pdf(self, pdf_path: str, force_recreate: bool = False, 
                    chunk_size: int = 500, overlap: int = 50):
        """
        Process PDF: Load from cache or create new embeddings
        
        Args:
            pdf_path: Path to PDF file
            force_recreate: Force recreation of embeddings even if cached
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        """
        # Try to load from cache first
        if not force_recreate and self.load_embeddings(pdf_path):
            return
        
        # Extract text from PDF
        text = self.load_pdf(pdf_path)
        
        # Chunk the text
        self.chunks = self.chunk_text(text, chunk_size, overlap)
        
        # Create embeddings
        self.embeddings = self.create_embeddings(self.chunks)
        
        # Save to cache
        self.save_embeddings(pdf_path)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for query
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most relevant chunks
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.chunks or not self.embeddings:
            raise ValueError("No embeddings loaded. Please process a PDF first.")
        
        print(f"🔍 Searching for: '{query}'")
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Calculate similarities
        query_vec = np.array(query_embedding).reshape(1, -1)
        embeddings_matrix = np.array(self.embeddings)
        similarities = cosine_similarity(query_vec, embeddings_matrix)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.chunks[i], float(similarities[i])) for i in top_indices]
        
        print(f"✅ Found {len(results)} relevant chunks")
        return results
    
    def answer_query(self, query: str, top_k: int = 3, model: str = "gpt-4o-mini") -> str:
        """
        Answer query using RAG
        
        Args:
            query: User query
            top_k: Number of context chunks to use
            model: OpenAI model for answering
            
        Returns:
            Answer text
        """
        # Get relevant chunks
        results = self.search(query, top_k)
        
        # Build context
        context = "\n\n".join([f"Context {i+1}:\n{chunk}" 
                               for i, (chunk, _) in enumerate(results)])
        
        # Create prompt
        prompt = f"""Based on the following context from a PDF document, answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        print(f"💬 Generating answer using {model}...")
        
        # Get answer from GPT
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        print("✅ Answer generated\n")
        
        return answer

