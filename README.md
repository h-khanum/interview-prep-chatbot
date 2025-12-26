# AI-Powered Interview Preparation Chatbot for Data Science Students

## 📋 Project Overview

This is a comprehensive **RAG (Retrieval-Augmented Generation)** based chatbot designed to help data science students prepare for technical interviews. The system acts as an intelligent interviewer that asks questions, evaluates answers, provides feedback, and adapts to the user's skill level.

**Key Features:**
- 🤖 AI-powered conversational interviewer
- 🎯 Topic-specific practice (Python, ML, Deep Learning, SQL, Statistics, A/B Testing)
- 📊 Real-time answer evaluation with semantic analysis
- 💡 Contextual hints and guidance
- 📈 Performance tracking and analytics
- 🎨 Modern, responsive UI

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                    (Web Browser)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP/JSON API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK SERVER                              │
│                     (app.py)                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  REST API Endpoints:                                 │   │
│  │  • /api/start  - Initialize session                  │   │
│  │  • /api/chat   - Process messages                    │   │
│  │  • /api/hint   - Request hints                       │   │
│  │  • /api/next   - Get next question                   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                  CHATBOT LOGIC                               │
│                   (logic.py)                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Conversation management                           │   │
│  │  • Prompt construction                               │   │
│  │  • RAG context injection                             │   │
│  │  • OpenAI API integration                            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────┬──────────────────────────────────┬─────────────────┘
         │                                   │
         │                                   │
    ┌────▼─────────┐                   ┌────▼──────────┐
    │   RAG LAYER  │                   │  EVALUATOR    │
    │ (preprocess) │                   │ (evaluator.py)│
    └────┬─────────┘                   └────┬──────────┘
         │                                   │
    ┌────▼──────────────────────────────────▼────┐
    │         KNOWLEDGE BASE                      │
    │        (questions.json)                     │
    │                                             │
    │  • Interview questions                      │
    │  • Topics & difficulty levels               │
    │  • Ideal answers                            │
    │  • Key concepts                             │
    └─────────────────────────────────────────────┘
         │
    ┌────▼──────────────┐
    │  VECTOR INDEX     │
    │   (FAISS)         │
    │                   │
    │  • Embeddings     │
    │  • Similarity     │
    │    search         │
    └───────────────────┘
```

---

## 🔄 RAG Pipeline Explanation

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances LLM responses by retrieving relevant information from a knowledge base before generating answers. This approach combines:

1. **Information Retrieval** - Finding relevant context
2. **Language Generation** - Creating natural responses

### Our RAG Implementation

#### Step 1: Data Preprocessing (`preprocess.py`)
```
questions.json → Text Chunks → Embeddings → FAISS Index
```

- Load interview questions from JSON
- Create searchable text chunks with metadata
- Generate vector embeddings using `sentence-transformers`
- Build FAISS index for fast similarity search

#### Step 2: Query Processing (`logic.py`)
```
User Query → Query Embedding → Vector Search → Top-K Results
```

- Convert user message to vector embedding
- Search FAISS index for similar content
- Retrieve top-k most relevant question chunks
- Filter by topic/difficulty if specified

#### Step 3: Context Injection (`logic.py`)
```
Retrieved Context + User Message → LLM Prompt → Response
```

- Inject retrieved context into system prompt
- Add conversation history
- Send to OpenAI API
- Generate contextually-aware response

#### Step 4: Answer Evaluation (`evaluator.py`)
```
User Answer + Ideal Answer → Similarity Score + Feedback
```

- Calculate semantic similarity
- Analyze keyword coverage
- Generate detailed feedback
- Provide constructive suggestions

---

## 🛠️ Technologies Used

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework for REST API
- **OpenAI API** - LLM for conversation
- **sentence-transformers** - Text embeddings (all-MiniLM-L6-v2)
- **FAISS** - Fast similarity search
- **scikit-learn** - Evaluation metrics

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Tailwind CSS** - Utility-first styling
- **JavaScript (Vanilla)** - Frontend logic
- **Material Symbols** - Icon library

### AI/ML Components
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: GPT-3.5-turbo (configurable)
- **Vector Store**: FAISS (IndexFlatIP)
- **Similarity Metric**: Cosine similarity

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set OpenAI API Key
```bash
# On Windows
set OPENAI_API_KEY=your-api-key-here

# On macOS/Linux
export OPENAI_API_KEY=your-api-key-here
```

### Step 5: Initialize Knowledge Base
The knowledge base is automatically initialized on first run. The system will:
- Load `data/questions.json`
- Generate embeddings (takes 1-2 minutes)
- Build FAISS index

### Step 6: Run Application
```bash
python app.py
```

The server will start at: `http://localhost:5000`

---

## 📖 Usage Guide

### Starting a Session

1. **Open** `http://localhost:5000` in your browser
2. **Select** topic and difficulty level (optional)
3. **Click** "Start Interview"

### Interacting with the Bot

**Commands:**
- Type your answer naturally
- "next question" - Get a new question
- "hint" - Request a hint
- "score" or "performance" - View statistics

**Features:**
- Real-time answer evaluation
- Detailed feedback with strengths/improvements
- Semantic similarity scoring
- Keyword coverage analysis

### API Endpoints

#### POST `/api/start`
Start new session with preferences
```json
{
  "topic": "Python",
  "difficulty": "Intermediate"
}
```

#### POST `/api/chat`
Send message
```json
{
  "message": "List comprehensions create lists in memory..."
}
```

#### POST `/api/hint`
Request hint for current question

#### POST `/api/next`
Get next question

#### GET `/api/stats`
Get session statistics

#### POST `/api/reset`
Reset conversation

---

## 📂 Project Structure

```
chatbot/
│
├── app.py              # Flask server & API routes
├── logic.py            # Chatbot logic & RAG implementation
├── preprocess.py       # Data loading & vector indexing
├── evaluator.py        # Answer evaluation & feedback
│
├── data/
│   └── questions.json  # Knowledge base (18 questions)
│
├── templates/
│   └── index.html      # Frontend UI
│
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 🎯 Key Features Explained

### 1. RAG-Based Context Retrieval
- Uses semantic search to find relevant questions
- Injects context into LLM prompts
- Filters by topic and difficulty
- Top-k retrieval (default: 2-3 chunks)

### 2. Intelligent Answer Evaluation
- **Semantic Similarity** (60% weight)
  - Uses cosine similarity on embeddings
  - Measures conceptual understanding
  
- **Keyword Coverage** (40% weight)
  - Checks for key concepts
  - Identifies missing points
  
- **Feedback Generation**
  - Categorizes performance (Excellent → Weak)
  - Lists strengths and improvements
  - Suggests specific concepts to review

### 3. Adaptive Interview Flow
- Tracks conversation history
- Maintains session state
- Supports multi-turn conversations
- Provides contextual responses

### 4. Performance Analytics
- Questions answered counter
- Average score calculation
- Per-session tracking
- Topic-wise analytics

---

## 🔬 Academic Concepts

### Machine Learning Techniques Used

1. **Text Embeddings**
   - Transforms text into 384-dimensional vectors
   - Captures semantic meaning
   - Model: all-MiniLM-L6-v2

2. **Vector Similarity Search**
   - FAISS library for efficient search
   - Inner product similarity (cosine after normalization)
   - O(n) search complexity with IndexFlatIP

3. **Natural Language Processing**
   - Tokenization and keyword extraction
   - Stop word filtering
   - Semantic similarity calculation

4. **Evaluation Metrics**
   - Cosine similarity for semantic matching
   - Precision-based keyword coverage
   - Weighted scoring algorithm

---

## 🚀 Future Improvements

1. **Enhanced RAG**
   - Multi-query retrieval
   - Reranking retrieved chunks
   - Hybrid search (keyword + semantic)

2. **Advanced Evaluation**
   - Fine-tuned evaluation model
   - Multi-dimensional scoring
   - Rubric-based assessment

3. **Features**
   - Voice input/output
   - Code execution sandbox
   - Spaced repetition system
   - Personalized learning paths

4. **Infrastructure**
   - User authentication
   - Database for persistence
   - Progress visualization
   - Export session reports

5. **AI Enhancements**
   - Fine-tuned domain-specific models
   - Multi-modal inputs (code, diagrams)
   - Adaptive difficulty adjustment
   - Collaborative filtering recommendations

---

## 📝 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required: OpenAI API key
FLASK_ENV=development           # Optional: Flask environment
FLASK_DEBUG=1                   # Optional: Debug mode
```

### Configurable Parameters

In `logic.py`:
```python
model = "gpt-3.5-turbo"         # LLM model
temperature = 0.7               # Response creativity
max_tokens = 500                # Response length
```

In `preprocess.py`:
```python
model_name = "all-MiniLM-L6-v2" # Embedding model
top_k = 3                       # Retrieved chunks
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "No OpenAI API key" error**
- Set `OPENAI_API_KEY` environment variable
- Verify key is valid on OpenAI dashboard

**2. Slow first startup**
- Normal: downloading embedding model (~80MB)
- Subsequent starts are fast (model cached)

**3. "Module not found" errors**
- Run: `pip install -r requirements.txt`
- Check Python version >= 3.8

**4. FAISS installation issues**
- Use: `pip install faiss-cpu`
- For GPU: `pip install faiss-gpu`

---

## 👥 Contributors

- **Your Name** - Developer & Researcher
- **University Name** - Academic Institution
- **Course Code** - Semester Project

---

## 📄 License

This project is developed for academic purposes as part of a university semester project.

---

## 🙏 Acknowledgments

- OpenAI for GPT models
- Hugging Face for sentence-transformers
- Facebook AI for FAISS
- Flask community for web framework

---

## 📧 Contact

For questions or issues, please contact:
- Email: your.email@university.edu
- GitHub: your-github-username

---

**Last Updated:** December 2024  
**Version:** 1.0.0  
**Status:** ✅ Production Ready