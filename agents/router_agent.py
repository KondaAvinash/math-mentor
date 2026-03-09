"""
Agent 2: Intent Router Agent
Classifies problem type and decides solving strategy.
"""
from utils.llm_client import call_llm_json

SYSTEM = """You are a math problem router for JEE problems.
Classify the problem and choose the best solving strategy.
Return ONLY valid JSON."""

STRATEGIES = {
    "algebra": ["symbolic_solve", "direct_compute"],
    "probability": ["combinatorics", "direct_compute"],
    "calculus": ["differentiation", "integration", "limits"],
    "linear_algebra": ["matrix_ops", "determinant", "eigenvalue"],
}


def run_router_agent(parsed_problem: dict) -> dict:
    topic = parsed_problem.get("topic", "algebra")
    subtopic = parsed_problem.get("subtopic", "general")
    problem = parsed_problem.get("problem_text", "")
    what = parsed_problem.get("what_to_find", "")

    prompt = f"""Route this JEE math problem to the right solving strategy.

Problem: {problem}
Topic: {topic}
Subtopic: {subtopic}
What to find: {what}

Return JSON:
{{
  "topic": "{topic}",
  "subtopic": "{subtopic}",
  "difficulty": "easy | medium | hard",
  "strategy": "primary_strategy | fallback_strategy",
  "requires_python": true or false,
  "requires_rag": true,
  "reasoning": "one line why this strategy"
}}"""

    result = call_llm_json(prompt, SYSTEM)

    defaults = {
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": "medium",
        "strategy": "direct_compute",
        "requires_python": False,
        "requires_rag": True,
        "reasoning": "default routing",
    }
    for k, v in defaults.items():
        result.setdefault(k, v)
    return result
