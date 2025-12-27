"""
evaluator.py
------------
Evaluates user answers against ideal answers using semantic similarity
and keyword matching. Provides detailed feedback and scoring.

Key responsibilities:
1. Semantic similarity scoring using embeddings
2. Keyword coverage analysis
3. Generate constructive feedback
4. Identify strengths and areas for improvement
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import re


class AnswerEvaluator:
    """
    Evaluates user answers and provides feedback.
    Uses both semantic similarity and keyword matching.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize evaluator with embedding model.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.encoder = SentenceTransformer(model_name)
    
    def _calculate_semantic_similarity(self, user_answer: str, ideal_answer: str) -> float:
        """
        Calculate semantic similarity between user and ideal answers.
        
        Args:
            user_answer: User's response
            ideal_answer: Expected answer
        
        Returns:
            Similarity score (0-1)
        """
        if not user_answer.strip():
            return 0.0
        
        # Generate embeddings
        embeddings = self.encoder.encode([user_answer, ideal_answer])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def _extract_keywords(self, text: str) -> set:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
        
        Returns:
            Set of lowercase keywords
        """
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'it', 'that', 'this', 'these', 'those', 'can', 'will', 'would',
            'should', 'could', 'may', 'might', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'very', 'much', 'more', 'most'
        }
        
        keywords = {w for w in words if len(w) > 2 and w not in stop_words}
        
        return keywords
    
    def _calculate_keyword_coverage(
        self,
        user_answer: str,
        ideal_answer: str,
        key_points: List[str]
    ) -> Dict:
        """
        Calculate how many key concepts are covered.
        
        Args:
            user_answer: User's response
            ideal_answer: Expected answer
            key_points: List of key concepts to cover
        
        Returns:
            Dictionary with coverage metrics
        """
        user_keywords = self._extract_keywords(user_answer)
        ideal_keywords = self._extract_keywords(ideal_answer)
        key_points_keywords = set()
        for kp in key_points:
            key_points_keywords.update(self._extract_keywords(kp))
        
        # Calculate coverage
        ideal_covered = len(user_keywords.intersection(ideal_keywords))
        ideal_total = len(ideal_keywords)
        
        key_points_covered = len(user_keywords.intersection(key_points_keywords))
        key_points_total = len(key_points_keywords)
        
        # Coverage scores
        ideal_coverage = ideal_covered / ideal_total if ideal_total > 0 else 0
        key_points_coverage = key_points_covered / key_points_total if key_points_total > 0 else 0
        
        # Identify missing key points
        missing_concepts = []
        for kp in key_points:
            kp_lower = kp.lower()
            if kp_lower not in user_answer.lower():
                missing_concepts.append(kp)
        
        return {
            'ideal_coverage': ideal_coverage,
            'key_points_coverage': key_points_coverage,
            'missing_concepts': missing_concepts,
            'covered_keywords': list(user_keywords.intersection(ideal_keywords))
        }
    
    def _determine_score_category(self, overall_score: float) -> str:
        """
        Categorize score into performance levels.
        
        Args:
            overall_score: Overall score (0-100)
        
        Returns:
            Category string
        """
        if overall_score >= 85:
            return "Excellent"
        elif overall_score >= 70:
            return "Good"
        elif overall_score >= 55:
            return "Fair"
        elif overall_score >= 40:
            return "Needs Improvement"
        else:
            return "Weak"
    
    def _generate_feedback(
        self,
        overall_score: float,
        semantic_score: float,
        keyword_coverage: Dict,
        user_answer: str
    ) -> Dict:
        """
        Generate detailed feedback based on evaluation.
        
        Args:
            overall_score: Overall score (0-100)
            semantic_score: Semantic similarity score
            keyword_coverage: Keyword coverage metrics
            user_answer: User's response
        
        Returns:
            Feedback dictionary
        """
        category = self._determine_score_category(overall_score)
        
        feedback = {
            'category': category,
            'strengths': [],
            'improvements': [],
            'missing_points': keyword_coverage['missing_concepts'],
            'summary': ''
        }
        
        # Strengths
        if semantic_score >= 0.7:
            feedback['strengths'].append("Strong understanding of the core concepts")
        
        if keyword_coverage['key_points_coverage'] >= 0.6:
            feedback['strengths'].append("Covered most key points effectively")
        
        if len(user_answer.split()) >= 30:
            feedback['strengths'].append("Provided detailed explanation")
        
        if not feedback['strengths']:
            feedback['strengths'].append("Showed effort in attempting the question")
        
        # Improvements
        if semantic_score < 0.5:
            feedback['improvements'].append("Work on understanding the fundamental concepts more deeply")
        
        if keyword_coverage['key_points_coverage'] < 0.4:
            feedback['improvements'].append("Include more specific technical terms and key concepts")
        
        if len(keyword_coverage['missing_concepts']) > 0:
            feedback['improvements'].append(
                f"Address these missing concepts: {', '.join(keyword_coverage['missing_concepts'][:3])}"
            )
        
        if len(user_answer.split()) < 20:
            feedback['improvements'].append("Provide more detailed explanations with examples")
        
        # Summary
        if category == "Excellent":
            feedback['summary'] = "Outstanding answer! You demonstrated strong mastery of the topic."
        elif category == "Good":
            feedback['summary'] = "Good answer with solid understanding. Minor improvements possible."
        elif category == "Fair":
            feedback['summary'] = "Decent attempt, but some key concepts need more clarity."
        elif category == "Needs Improvement":
            feedback['summary'] = "You have a basic grasp, but several important points are missing."
        else:
            feedback['summary'] = "This answer needs significant improvement. Review the concepts carefully."
        
        return feedback
    
    def evaluate(
        self,
        user_answer: str,
        ideal_answer: str,
        key_points: List[str] = None
    ) -> Dict:
        """
        Comprehensive evaluation of user answer.
        
        Args:
            user_answer: User's response
            ideal_answer: Expected answer
            key_points: List of key concepts (optional)
        
        Returns:
            Evaluation dictionary with scores and feedback
        """
        if not user_answer or not user_answer.strip():
            return {
                'overall_score': 0,
                'semantic_score': 0,
                'keyword_score': 0,
                'feedback': {
                    'category': 'No Answer',
                    'strengths': [],
                    'improvements': ['Please provide an answer to the question'],
                    'missing_points': key_points or [],
                    'summary': 'No answer was provided.'
                }
            }
        
        key_points = key_points or []
        
        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(user_answer, ideal_answer)
        
        # Calculate keyword coverage
        keyword_coverage = self._calculate_keyword_coverage(
            user_answer,
            ideal_answer,
            key_points
        )
        
        # Weighted overall score
        keyword_score = (
            keyword_coverage['ideal_coverage'] * 0.5 +
            keyword_coverage['key_points_coverage'] * 0.5
        )
        
        overall_score = (semantic_score * 0.6 + keyword_score * 0.4) * 100
        
        # Generate feedback
        feedback = self._generate_feedback(
            overall_score,
            semantic_score,
            keyword_coverage,
            user_answer
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'semantic_score': round(semantic_score * 100, 2),
            'keyword_score': round(keyword_score * 100, 2),
            'feedback': feedback
        }
    
    def get_hints(
        self,
        question: str,
        ideal_answer: str,
        key_points: List[str]
    ) -> List[str]:
        """
        Generate hints for a question.
        
        Args:
            question: The interview question
            ideal_answer: Expected answer
            key_points: Key concepts to cover
        
        Returns:
            List of hint strings
        """
        hints = []
        
        if key_points and len(key_points) > 0:
            hints.append(f"Think about these key concepts: {', '.join(key_points[:2])}")
        
        # Extract first sentence of ideal answer as a hint
        sentences = ideal_answer.split('.')
        if len(sentences) > 0 and sentences[0].strip():
            hints.append(f"Consider starting with: {sentences[0].strip()}...")
        
        hints.append("Try to explain both the 'what' and the 'why'")
        
        return hints


# Global evaluator instance
_evaluator_instance = None


def get_evaluator() -> AnswerEvaluator:
    """
    Get or create singleton evaluator instance.
    
    Returns:
        AnswerEvaluator instance
    """
    global _evaluator_instance
    
    if _evaluator_instance is None:
        _evaluator_instance = AnswerEvaluator()
    
    return _evaluator_instance


def evaluate_answer(
    user_answer: str,
    ideal_answer: str,
    key_points: List[str] = None
) -> Dict:
    """
    Convenience function for answer evaluation.
    
    Args:
        user_answer: User's response
        ideal_answer: Expected answer
        key_points: Key concepts list
    
    Returns:
        Evaluation results
    """
    evaluator = get_evaluator()
    return evaluator.evaluate(user_answer, ideal_answer, key_points)


def get_question_hints(
    question: str,
    ideal_answer: str,
    key_points: List[str]
) -> List[str]:
    """
    Get hints for a question.
    
    Args:
        question: Interview question
        ideal_answer: Expected answer
        key_points: Key concepts
    
    Returns:
        List of hints
    """
    evaluator = get_evaluator()
    return evaluator.get_hints(question, ideal_answer, key_points)