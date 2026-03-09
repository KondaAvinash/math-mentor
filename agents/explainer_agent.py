"""
Agent 5: Explainer / Tutor Agent
Produces student-friendly step-by-step explanation.
"""
from utils.llm_client import call_llm

SYSTEM = """You are a friendly JEE math tutor explaining to a student.
Use simple language. Show each step clearly.
Use bullet points for steps. Keep it under 200 words."""


def run_explainer_agent(parsed_problem: dict, solver_result: dict, verifier_result: dict) -> dict:
    problem = parsed_problem.get("problem_text", "")
    solution = solver_result.get("solution_text", "")
    answer = verifier_result.get("verified_answer", solver_result.get("final_answer", ""))
    topic = parsed_problem.get("topic", "")
    subtopic = parsed_problem.get("subtopic", "")

    prompt = f"""Explain this JEE solution to a student in simple words.

PROBLEM: {problem}
TOPIC: {topic} / {subtopic}
SOLUTION WORKING: {solution}
FINAL ANSWER: {answer}

Write a clear explanation:
1. What concept is being used
2. Step by step what we did
3. Why the answer is {answer}

Keep it simple, friendly, under 200 words."""

    explanation = call_llm(prompt, SYSTEM, temperature=0.2, max_tokens=400)

    return {
        "explanation": explanation,
        "concept_used": f"{topic} — {subtopic}",
        "final_answer": answer,
    }
