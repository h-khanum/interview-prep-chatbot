"""
app.py
------
Flask application serving the RAG-based interview chatbot.
Provides REST API endpoints and serves the frontend.

Routes:
- GET  /           : Serve main UI
- POST /chat       : Process chat messages
- POST /start      : Start new session with preferences
- GET  /topics     : Get available topics
- GET  /reset      : Reset conversation
"""

import os
import json
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from datetime import datetime
import uuid

from chatbot.logic import get_bot, reset_bot
from chatbot.preprocess import get_knowledge_base


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)  # Enable CORS for API calls


# ========== Helper Functions ==========

def get_session_id():
    """Get or create session ID for user."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def format_response(success: bool, data: dict = None, error: str = None):
    """
    Standardize API response format.
    
    Args:
        success: Whether request was successful
        data: Response data
        error: Error message if failed
    
    Returns:
        JSON response dictionary
    """
    response = {'success': success}
    
    if success and data:
        response['data'] = data
    elif not success and error:
        response['error'] = error
    
    return jsonify(response)


# ========== Routes ==========

@app.route('/')
def index():
    """Serve main chat interface."""
    return render_template('index.html')


@app.route('/api/topics', methods=['GET'])
def get_topics():
    """
    Get available topics and difficulty levels.
    
    Returns:
        JSON with topics and difficulties
    """
    try:
        kb = get_knowledge_base()
        
        return format_response(True, {
            'topics': kb.get_all_topics(),
            'difficulties': kb.get_all_difficulties()
        })
    except Exception as e:
        return format_response(False, error=str(e))


@app.route('/api/start', methods=['POST'])
def start_session():
    """
    Start new interview session with user preferences.
    
    Expected JSON:
        {
            "topic": "Python",  # optional
            "difficulty": "Intermediate"  # optional
        }
    
    Returns:
        Welcome message and first question
    """
    try:
        data = request.get_json() or {}
        
        topic = data.get('topic')
        difficulty = data.get('difficulty')
        
        # Get bot for this session
        sid = get_session_id()
        bot = get_bot(sid)
        
        # Reset conversation
        bot.reset_conversation()
        
        # Set preferences
        bot.set_preferences(topic=topic, difficulty=difficulty)
        
        # Generate welcome message
        welcome_parts = ["Welcome to your interview prep session! 👋"]
        
        if topic:
            welcome_parts.append(f"**Topic:** {topic}")
        if difficulty:
            welcome_parts.append(f"**Level:** {difficulty}")
        
        welcome_parts.append("\nI'll act as your interviewer and help you practice. Ready to start?")
        
        welcome_message = "\n".join(welcome_parts)
        
        # Ask first question
        question_response = bot.ask_question()
        
        return format_response(True, {
            'welcome_message': welcome_message,
            'question': question_response,
            'session_id': sid
        })
        
    except Exception as e:
        print(f"Error in start_session: {e}")
        return format_response(False, error=str(e))


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Process user message and return bot response.
    
    Expected JSON:
        {
            "message": "user message here"
        }
    
    Returns:
        Bot response with type and content
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return format_response(False, error='Message is required')
        
        user_message = data['message'].strip()
        
        if not user_message:
            return format_response(False, error='Message cannot be empty')
        
        # Get bot instance
        sid = get_session_id()
        bot = get_bot(sid)
        
        # Process message
        response = bot.process_message(user_message)
        
        # Add timestamp
        response['timestamp'] = datetime.now().isoformat()
        
        return format_response(True, response)
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return format_response(False, error=str(e))


@app.route('/api/hint', methods=['POST'])
def get_hint():
    """
    Get hint for current question.
    
    Returns:
        Hint message
    """
    try:
        sid = get_session_id()
        bot = get_bot(sid)
        
        if not bot.current_question:
            return format_response(False, error='No active question to provide hint for')
        
        from chatbot.evaluator import get_question_hints
        
        hints = get_question_hints(
            bot.current_question['question'],
            bot.current_question['ideal_answer'],
            bot.current_question.get('key_points', [])
        )
        
        hint_message = "💡 **Hint:**\n" + hints[0] if hints else "No hint available."
        
        return format_response(True, {
            'type': 'hint',
            'message': hint_message,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_hint: {e}")
        return format_response(False, error=str(e))


@app.route('/api/next', methods=['POST'])
def next_question():
    """
    Ask next question.
    
    Returns:
        New question
    """
    try:
        sid = get_session_id()
        bot = get_bot(sid)
        
        response = bot.ask_question()
        response['timestamp'] = datetime.now().isoformat()
        
        return format_response(True, response)
        
    except Exception as e:
        print(f"Error in next_question: {e}")
        return format_response(False, error=str(e))


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get session statistics.
    
    Returns:
        Session summary with scores
    """
    try:
        sid = get_session_id()
        bot = get_bot(sid)
        
        summary = bot.get_session_summary()
        summary['timestamp'] = datetime.now().isoformat()
        
        return format_response(True, summary)
        
    except Exception as e:
        print(f"Error in get_stats: {e}")
        return format_response(False, error=str(e))


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """
    Reset conversation and start fresh.
    
    Returns:
        Success confirmation
    """
    try:
        sid = get_session_id()
        reset_bot(sid)
        
        return format_response(True, {
            'message': 'Session reset successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in reset_session: {e}")
        return format_response(False, error=str(e))


@app.route('/api/ideal-answer', methods=['GET'])
def get_ideal_answer():
    """
    Get ideal answer for current question (for reference).
    
    Returns:
        Ideal answer text
    """
    try:
        sid = get_session_id()
        bot = get_bot(sid)
        
        if not bot.current_question:
            return format_response(False, error='No active question')
        
        return format_response(True, {
            'ideal_answer': bot.current_question['ideal_answer'],
            'key_points': bot.current_question.get('key_points', []),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_ideal_answer: {e}")
        return format_response(False, error=str(e))


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return format_response(True, {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


# ========== Error Handlers ==========

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return format_response(False, error='Endpoint not found'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return format_response(False, error='Internal server error'), 500


# ========== Main ==========

if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("=" * 60)
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running the application:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("=" * 60)
    
    # Initialize knowledge base on startup
    print("Initializing knowledge base...")
    try:
        kb = get_knowledge_base()
        print(f"✓ Knowledge base loaded successfully")
        print(f"  - Topics: {', '.join(kb.get_all_topics())}")
        print(f"  - Questions: {len(kb.questions)}")
    except Exception as e:
        print(f"✗ Error loading knowledge base: {e}")
    
    # Run Flask app
    print("\nStarting Flask server...")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
