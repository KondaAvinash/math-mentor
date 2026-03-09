"""
Memory Store — saves solved problems and feedback for self-learning.
Uses JSON + embedding similarity to find similar past problems.
"""
import json
import os
import uuid
from datetime import datetime
from config import MEMORY_DB_PATH, FEEDBACK_DB_PATH

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _load_db(path: str) -> list:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_db(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_to_memory(
    raw_input: str,
    parsed_problem: dict,
    solver_result: dict,
    verifier_result: dict,
    input_source: str = "text",
):
    """Save a solved problem to memory."""
    db = _load_db(MEMORY_DB_PATH)
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "input_source": input_source,
        "raw_input": raw_input[:500],
        "problem_text": parsed_problem.get("problem_text", ""),
        "topic": parsed_problem.get("topic", ""),
        "subtopic": parsed_problem.get("subtopic", ""),
        "mcq_options": parsed_problem.get("mcq_options", {}),
        "final_answer": verifier_result.get("verified_answer", ""),
        "confidence": verifier_result.get("confidence", 0),
        "is_correct": verifier_result.get("is_correct", True),
        "user_feedback": None,
    }
    db.append(entry)
    _save_db(MEMORY_DB_PATH, db)
    return entry["id"]


def save_feedback(problem_id: str, is_correct: bool, comment: str = ""):
    """Save user feedback for a problem."""
    # Update memory
    db = _load_db(MEMORY_DB_PATH)
    for entry in db:
        if entry.get("id") == problem_id:
            entry["user_feedback"] = {"correct": is_correct, "comment": comment}
            entry["is_correct"] = is_correct
    _save_db(MEMORY_DB_PATH, db)

    # Save to feedback log
    feedback_db = _load_db(FEEDBACK_DB_PATH)
    feedback_db.append({
        "problem_id": problem_id,
        "timestamp": datetime.now().isoformat(),
        "is_correct": is_correct,
        "comment": comment,
    })
    _save_db(FEEDBACK_DB_PATH, feedback_db)


def find_similar_problems(query: str, top_k: int = 3) -> list:
    """Find similar past problems using embedding similarity."""
    import numpy as np
    db = _load_db(MEMORY_DB_PATH)
    if not db:
        return []
    try:
        model = _get_embed_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        texts = [e.get("problem_text", "") for e in db]
        vecs = model.encode(texts, normalize_embeddings=True)
        scores = np.dot(vecs, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            if scores[i] > 0.70:
                entry = db[i].copy()
                entry["similarity"] = float(scores[i])
                results.append(entry)
        return results
    except Exception:
        return []


def get_stats() -> dict:
    """Get memory statistics."""
    db = _load_db(MEMORY_DB_PATH)
    feedback_db = _load_db(FEEDBACK_DB_PATH)
    topics = {}
    for e in db:
        t = e.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
    return {
        "total_solved": len(db),
        "with_feedback": len(feedback_db),
        "topics": topics,
    }
