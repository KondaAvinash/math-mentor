# 🧮 Math Mentor — JEE Math Solver

Multimodal AI app solving JEE-style math problems using RAG + Multi-Agent + HITL + Memory.

## Stack
- **Image OCR:** Google Gemini Vision API (reads math perfectly)
- **LLM:** Groq API — LLaMA-3.1-8B (fast, free)
- **RAG:** FAISS + sentence-transformers (local)
- **Audio:** Whisper tiny (local)
- **UI:** Streamlit
- **Deploy:** HuggingFace Spaces

## Setup

### 1. Get free API keys
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure .env
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Build RAG index (run once)
```bash
python -m rag.ingest
```

### 5. Run app
```bash
streamlit run app.py
```

## 5 Agents
1. **Parser Agent** — converts raw input to structured JSON
2. **Router Agent** — classifies problem and picks strategy
3. **Solver Agent** — solves using RAG + Groq LLaMA
4. **Verifier Agent** — checks correctness, triggers HITL
5. **Explainer Agent** — student-friendly explanation

## HITL Triggers
- OCR confidence < 55%
- ASR confidence < 50%
- Parser detects ambiguity
- Verifier confidence < 60%
- User forces review via toggle
