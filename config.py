import os
from dotenv import load_dotenv
load_dotenv()

# APIs
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")

# Models
GROQ_MODEL            = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_VISION_MODEL   = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash")
WHISPER_MODEL         = os.getenv("WHISPER_MODEL", "tiny")
EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG
FAISS_INDEX_PATH      = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
KNOWLEDGE_BASE_PATH   = os.getenv("KNOWLEDGE_BASE_PATH", "rag/knowledge_base")
TOP_K_RESULTS         = int(os.getenv("TOP_K_RESULTS", 4))

# Memory
MEMORY_DB_PATH        = os.getenv("MEMORY_DB_PATH", "data/memory.json")
FEEDBACK_DB_PATH      = os.getenv("FEEDBACK_DB_PATH", "data/feedback/feedback.json")
