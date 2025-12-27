"""
preprocess.py
-------------
Handles data loading, text preprocessing, embedding generation, and vector indexing
for the RAG (Retrieval-Augmented Generation) pipeline.

Key responsibilities:
1. Load questions.json knowledge base
2. Create text chunks with metadata
3. Generate embeddings using sentence-transformers
4. Build FAISS vector index for similarity search
5. Provide retrieval functionality
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional


class KnowledgeBase:
    """
    Manages the interview questions knowledge base with RAG capabilities.
    Implements vector search for context retrieval.
    """
    
    def __init__(self, data_path: str = "data/questions.json", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the knowledge base with embeddings and vector index.
        
        Args:
            data_path: Path to questions.json file
            model_name: Sentence transformer model for embeddings
        """
        self.data_path = data_path
        self.model_name = model_name
        
        # Load embedding model (lightweight and fast)
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        # Storage for questions and embeddings
        self.questions = []
        self.chunks = []  # Text chunks with metadata
        self.embeddings = None
        self.index = None  # FAISS index
        
        # Load and process data
        self._load_data()
        self._create_chunks()
        self._build_index()
        
        print(f"Knowledge base initialized with {len(self.chunks)} chunks")
    
    def _load_data(self):
        """Load questions from JSON file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.questions = data.get('questions', [])
        
        print(f"Loaded {len(self.questions)} questions from knowledge base")
    
    def _create_chunks(self):
        """
        Create text chunks from questions for embedding.
        Each chunk contains question + answer with metadata.
        """
        for q in self.questions:
            # Primary chunk: full question with context
            full_text = f"Topic: {q['topic']} | Difficulty: {q['difficulty']}\n"
            full_text += f"Question: {q['question']}\n"
            full_text += f"Answer: {q['ideal_answer']}"
            
            chunk = {
                'text': full_text,
                'question_id': q['id'],
                'topic': q['topic'],
                'difficulty': q['difficulty'],
                'question': q['question'],
                'ideal_answer': q['ideal_answer'],
                'key_points': q.get('key_points', [])
            }
            self.chunks.append(chunk)
            
            # Additional chunk: key points for better retrieval
            if 'key_points' in q and q['key_points']:
                key_points_text = f"Topic: {q['topic']} | Key Concepts: "
                key_points_text += ", ".join(q['key_points'])
                key_points_text += f"\nRelated to: {q['question']}"
                
                self.chunks.append({
                    'text': key_points_text,
                    'question_id': q['id'],
                    'topic': q['topic'],
                    'difficulty': q['difficulty'],
                    'question': q['question'],
                    'ideal_answer': q['ideal_answer'],
                    'key_points': q['key_points'],
                    'is_key_points': True
                })
    
    def _build_index(self):
        """
        Generate embeddings and build FAISS index for fast similarity search.
        """
        print("Generating embeddings for all chunks...")
        
        # Extract text from all chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings (batch processing for efficiency)
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index (using inner product for cosine similarity)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine after normalization
        self.index.add(self.embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve_context(
        self, 
        query: str, 
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve most relevant context chunks using RAG.
        
        Args:
            query: User's query or conversation context
            topic: Filter by topic (Python, ML, SQL, etc.)
            difficulty: Filter by difficulty level
            top_k: Number of top results to return
        
        Returns:
            List of relevant chunks with similarity scores
        """
        # Generate query embedding
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index (get more than top_k for filtering)
        search_k = min(top_k * 5, len(self.chunks))
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Retrieve chunks and apply filters
        results = []
        for idx, score in zip(indices[0], distances[0]):
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = float(score)
            
            # Apply topic filter
            if topic and chunk['topic'].lower() != topic.lower():
                continue
            
            # Apply difficulty filter
            if difficulty and chunk['difficulty'].lower() != difficulty.lower():
                continue
            
            results.append(chunk)
            
            # Stop when we have enough results
            if len(results) >= top_k:
                break
        
        # If filtering removed all results, return unfiltered top results
        if not results:
            for idx, score in zip(indices[0][:top_k], distances[0][:top_k]):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
        
        return results
    
    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """
        Retrieve a specific question by ID.
        
        Args:
            question_id: Question identifier
        
        Returns:
            Question dictionary or None if not found
        """
        for q in self.questions:
            if q['id'] == question_id:
                return q
        return None
    
    def get_questions_by_filters(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[Dict]:
        """
        Get questions matching topic and/or difficulty.
        
        Args:
            topic: Topic filter
            difficulty: Difficulty filter
        
        Returns:
            List of matching questions
        """
        filtered = self.questions
        
        if topic:
            filtered = [q for q in filtered if q['topic'].lower() == topic.lower()]
        
        if difficulty:
            filtered = [q for q in filtered if q['difficulty'].lower() == difficulty.lower()]
        
        return filtered
    
    def get_random_question(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get a random question based on filters.
        
        Args:
            topic: Topic filter
            difficulty: Difficulty filter
        
        Returns:
            Random question or None
        """
        questions = self.get_questions_by_filters(topic, difficulty)
        
        if not questions:
            return None
        
        import random
        return random.choice(questions)
    
    def get_all_topics(self) -> List[str]:
        """Get list of unique topics in knowledge base."""
        return list(set(q['topic'] for q in self.questions))
    
    def get_all_difficulties(self) -> List[str]:
        """Get list of unique difficulty levels."""
        return list(set(q['difficulty'] for q in self.questions))


# Global instance (initialized once)
_kb_instance = None


def get_knowledge_base() -> KnowledgeBase:
    """
    Get or create singleton instance of KnowledgeBase.
    Ensures embeddings are loaded only once.
    
    Returns:
        KnowledgeBase instance
    """
    global _kb_instance
    
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    
    return _kb_instance


# Convenience functions for external use
def retrieve_relevant_context(
    query: str,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    top_k: int = 3
) -> List[Dict]:
    """
    Convenience function to retrieve context.
    Main entry point for RAG retrieval.
    """
    kb = get_knowledge_base()
    return kb.retrieve_context(query, topic, difficulty, top_k)


def get_question(
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    random: bool = True
) -> Optional[Dict]:
    """
    Get a question based on filters.
    """
    kb = get_knowledge_base()
    
    if random:
        return kb.get_random_question(topic, difficulty)
    else:
        questions = kb.get_questions_by_filters(topic, difficulty)
        return questions[0] if questions else None


# Initialize on module import for faster subsequent requests
# Comment out if you want lazy loading
# get_knowledge_base()