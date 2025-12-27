"""
logic.py
--------
Core chatbot logic implementing RAG-based conversational interview preparation.
Manages conversation flow, prompt construction, and LLM interaction.

Key responsibilities:
1. Conversation state management
2. RAG context injection into prompts
3. LLM interaction (OpenAI API)
4. Interview flow orchestration
5. Response generation
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# For OpenAI API calls
try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed. Install with: pip install openai")
    OpenAI = None

from chatbot.preprocess import (
    get_knowledge_base,
    retrieve_relevant_context
)

from chatbot.evaluator import (
    evaluate_answer, 
    get_question_hints
)

class InterviewBot:
    """
    Main chatbot class implementing RAG-based interview preparation.
    Acts as a strict but helpful interviewer.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the interview bot.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.model = model
        
        # Initialize OpenAI client
        if OpenAI is None:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        # Initialize knowledge base
        self.kb = get_knowledge_base()
        
        # Conversation state
        self.reset_conversation()
    
    def reset_conversation(self):
        """Reset conversation state."""
        self.conversation_history = []
        self.current_question = None
        self.current_topic = None
        self.current_difficulty = None
        self.questions_asked = []
        self.session_scores = []
        self.awaiting_answer = False
    
    def set_preferences(self, topic: Optional[str] = None, difficulty: Optional[str] = None):
        """
        Set interview preferences.
        
        Args:
            topic: Topic to focus on (Python, ML, SQL, etc.)
            difficulty: Difficulty level (Beginner, Intermediate, Advanced)
        """
        self.current_topic = topic
        self.current_difficulty = difficulty
    
    def _build_system_prompt(self) -> str:
        """
        Construct system prompt for the LLM with RAG context.
        
        Returns:
            System prompt string
        """
        base_prompt = """You are an experienced technical interviewer specializing in data science interviews. 

Your role:
- Act as a strict but encouraging interviewer
- Ask interview questions one at a time
- Evaluate answers thoughtfully
- Provide constructive feedback
- Adapt difficulty based on performance
- Stay professional and supportive

Guidelines:
- Be concise in your responses (2-4 sentences typically)
- Don't provide full answers unless explicitly asked
- Focus on guiding the candidate to think critically
- Use the provided context from the knowledge base
- Acknowledge good points and identify gaps
"""
        
        # Add topic/difficulty context if set
        if self.current_topic or self.current_difficulty:
            base_prompt += f"\n\nCurrent session focus:\n"
            if self.current_topic:
                base_prompt += f"- Topic: {self.current_topic}\n"
            if self.current_difficulty:
                base_prompt += f"- Difficulty: {self.current_difficulty}\n"
        
        return base_prompt
    
    def _get_rag_context(self, user_message: str, top_k: int = 2) -> str:
        """
        Retrieve relevant context from knowledge base using RAG.
        
        Args:
            user_message: User's latest message
            top_k: Number of relevant chunks to retrieve
        
        Returns:
            Formatted context string
        """
        # Build query from recent conversation
        query_parts = [user_message]
        
        # Add recent conversation for better context
        for msg in self.conversation_history[-4:]:
            if msg['role'] == 'user':
                query_parts.append(msg['content'])
        
        query = " ".join(query_parts)
        
        # Retrieve relevant chunks
        results = retrieve_relevant_context(
            query=query,
            topic=self.current_topic,
            difficulty=self.current_difficulty,
            top_k=top_k
        )
        
        # Format context
        if not results:
            return "No specific context retrieved."
        
        context_parts = []
        for i, chunk in enumerate(results, 1):
            context_parts.append(f"[Context {i}]")
            context_parts.append(f"Topic: {chunk['topic']} | Difficulty: {chunk['difficulty']}")
            context_parts.append(f"Q: {chunk['question']}")
            context_parts.append(f"Key Points: {', '.join(chunk['key_points'])}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """
        Call OpenAI API with conversation messages.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            LLM response text
        """
        if not self.client:
            # Fallback response when no API key
            return "I'm ready to help you prepare for interviews! However, I need an OpenAI API key to function properly. Please set the OPENAI_API_KEY environment variable."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Error: {e}")
            return "I'm having trouble connecting to my language model. Please check your API key and try again."
    
    def ask_question(self) -> Dict:
        """
        Ask a new interview question based on preferences.
        
        Returns:
            Response dictionary with question details
        """
        # Get a question from knowledge base
        question = self.kb.get_random_question(
            topic=self.current_topic,
            difficulty=self.current_difficulty
        )
        
        if not question:
            return {
                'type': 'error',
                'message': 'No questions available for the selected criteria.'
            }
        
        # Store current question
        self.current_question = question
        self.awaiting_answer = True
        self.questions_asked.append(question['id'])
        
        # Format question message
        difficulty_emoji = {
            'Beginner': '🟢',
            'Intermediate': '🟡',
            'Advanced': '🔴'
        }
        
        emoji = difficulty_emoji.get(question['difficulty'], '⚪')
        
        message = f"{emoji} **{question['topic']} Question** (Difficulty: {question['difficulty']})\n\n"
        message += question['question']
        
        # Add to conversation
        self.conversation_history.append({
            'role': 'assistant',
            'content': message
        })
        
        return {
            'type': 'question',
            'message': message,
            'question_id': question['id'],
            'topic': question['topic'],
            'difficulty': question['difficulty']
        }
    
    def _evaluate_user_answer(self, user_answer: str) -> Dict:
        """
        Evaluate user's answer to current question.
        
        Args:
            user_answer: User's answer text
        
        Returns:
            Evaluation results
        """
        if not self.current_question:
            return None
        
        evaluation = evaluate_answer(
            user_answer=user_answer,
            ideal_answer=self.current_question['ideal_answer'],
            key_points=self.current_question.get('key_points', [])
        )
        
        # Store score
        self.session_scores.append(evaluation['overall_score'])
        
        return evaluation
    
    def _format_feedback(self, evaluation: Dict) -> str:
        """
        Format evaluation feedback into a readable message.
        
        Args:
            evaluation: Evaluation dictionary
        
        Returns:
            Formatted feedback string
        """
        feedback = evaluation['feedback']
        score = evaluation['overall_score']
        
        # Score emoji
        if score >= 85:
            score_emoji = '🌟'
        elif score >= 70:
            score_emoji = '✅'
        elif score >= 55:
            score_emoji = '👍'
        else:
            score_emoji = '📚'
        
        message = f"{score_emoji} **Score: {score}/100** ({feedback['category']})\n\n"
        message += f"**{feedback['summary']}**\n\n"
        
        if feedback['strengths']:
            message += "**Strengths:**\n"
            for strength in feedback['strengths']:
                message += f"✓ {strength}\n"
            message += "\n"
        
        if feedback['improvements']:
            message += "**Areas for Improvement:**\n"
            for improvement in feedback['improvements']:
                message += f"→ {improvement}\n"
            message += "\n"
        
        if feedback['missing_points']:
            message += "**Key Concepts to Review:**\n"
            for point in feedback['missing_points'][:3]:
                message += f"• {point}\n"
        
        return message
    
    def process_message(self, user_message: str) -> Dict:
        """
        Process user message and generate response.
        Main entry point for conversation.
        
        Args:
            user_message: User's message
        
        Returns:
            Response dictionary
        """
        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Check if we're awaiting an answer
        if self.awaiting_answer and self.current_question:
            # Evaluate the answer
            evaluation = self._evaluate_user_answer(user_message)
            feedback_message = self._format_feedback(evaluation)
            
            # Add ideal answer summary
            feedback_message += f"\n**Model Answer (Summary):**\n"
            ideal = self.current_question['ideal_answer']
            # Show first 2 sentences
            sentences = ideal.split('.')[:2]
            feedback_message += '. '.join(sentences) + '...\n\n'
            
            feedback_message += "Ready for the next question? (Type 'next question' or ask me anything!)"
            
            self.awaiting_answer = False
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': feedback_message
            })
            
            return {
                'type': 'evaluation',
                'message': feedback_message,
                'evaluation': evaluation
            }
        
        # Check for special commands
        user_lower = user_message.lower().strip()
        
        if any(cmd in user_lower for cmd in ['next question', 'ask question', 'another question']):
            return self.ask_question()
        
        if 'hint' in user_lower and self.current_question:
            hints = get_question_hints(
                self.current_question['question'],
                self.current_question['ideal_answer'],
                self.current_question.get('key_points', [])
            )
            hint_message = "**Hint:**\n" + "\n".join(f"💡 {h}" for h in hints[:2])
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': hint_message
            })
            
            return {
                'type': 'hint',
                'message': hint_message
            }
        
        if 'score' in user_lower or 'performance' in user_lower:
            if self.session_scores:
                avg_score = sum(self.session_scores) / len(self.session_scores)
                stats_message = f"**Session Statistics:**\n"
                stats_message += f"📊 Questions Answered: {len(self.session_scores)}\n"
                stats_message += f"📈 Average Score: {avg_score:.1f}/100\n"
                stats_message += f"🎯 Highest Score: {max(self.session_scores):.1f}/100\n"
            else:
                stats_message = "You haven't answered any questions yet! Ready to start?"
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': stats_message
            })
            
            return {
                'type': 'stats',
                'message': stats_message
            }
        
        # General conversation with RAG
        rag_context = self._get_rag_context(user_message)
        
        # Build messages for LLM
        llm_messages = [
            {'role': 'system', 'content': self._build_system_prompt()},
            {'role': 'system', 'content': f"Relevant knowledge base context:\n{rag_context}"}
        ]
        
        # Add recent conversation history (last 6 messages)
        llm_messages.extend(self.conversation_history[-6:])
        
        # Get LLM response
        bot_response = self._call_llm(llm_messages)
        
        # Add to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': bot_response
        })
        
        return {
            'type': 'conversation',
            'message': bot_response
        }
    
    def get_session_summary(self) -> Dict:
        """
        Get summary of current session.
        
        Returns:
            Summary dictionary
        """
        return {
            'questions_asked': len(self.questions_asked),
            'average_score': sum(self.session_scores) / len(self.session_scores) if self.session_scores else 0,
            'topic': self.current_topic,
            'difficulty': self.current_difficulty
        }


# Global bot instance management
_bot_instances = {}


def get_bot(session_id: str = 'default') -> InterviewBot:
    """
    Get or create bot instance for session.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        InterviewBot instance
    """
    if session_id not in _bot_instances:
        _bot_instances[session_id] = InterviewBot()
    
    return _bot_instances[session_id]


def reset_bot(session_id: str = 'default'):
    """
    Reset bot session.
    
    Args:
        session_id: Session identifier
    """
    if session_id in _bot_instances:
        _bot_instances[session_id].reset_conversation()