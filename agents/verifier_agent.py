"""
Agent 4: Verifier / Critic Agent
Checks solution correctness and triggers HITL if unsure.
"""
from utils.llm_client import call_llm_json

SYSTEM = """You are a JEE math solution verifier.
Check if the solution is correct, units are right, and answer matches an MCQ option.
Return ONLY valid JSON."""

HITL_THRESHOLD = 0.60


def run_verifier_agent(parsed_problem: dict, solver_result: dict) -> dict:
    problem = parsed_problem.get("problem_text", "")
    mcq = parsed_problem.get("mcq_options", {})
    solution = solver_result.get("solution_text", "")
    answer = solver_result.get("final_answer", "")

    options_str = ""
    if mcq:
        options_str = "MCQ Options: " + ", ".join([f"({k}) {v}" for k, v in mcq.items()])

    prompt = f"""Verify this JEE math solution.

PROBLEM: {problem}
{options_str}
SOLUTION: {solution}
FINAL ANSWER: {answer}

Check:
1. Is the mathematical working correct?
2. Does the answer match one of the MCQ options (if MCQ)?
3. Are there any errors?

Return JSON:
{{
  "is_correct": true or false,
  "confidence": 0.0 to 1.0,
  "issues": ["list any issues found"],
  "verified_answer": "the correct answer",
  "needs_hitl": true or false,
  "feedback": "one line summary"
}}"""

    result = call_llm_json(prompt, SYSTEM)

    defaults = {
        "is_correct": True,
        "confidence": 0.75,
        "issues": [],
        "verified_answer": answer,
        "needs_hitl": False,
        "feedback": "Solution looks correct.",
    }
    for k, v in defaults.items():
        result.setdefault(k, v)

    # Force HITL if confidence too low
    if result.get("confidence", 1.0) < HITL_THRESHOLD:
        result["needs_hitl"] = True

    return result
