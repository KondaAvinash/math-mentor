"""
Agent 3: Solver Agent
Solves math using RAG context + Groq LLaMA.
Keeps response concise and picks correct MCQ option.
"""
import math
import re
from rag.retriever import get_retriever
from utils.llm_client import call_llm

SYSTEM = """You are a concise JEE math solver.
Rules:
1. Solve step by step — maximum 8 steps
2. If MCQ: compute answer then match to given options
3. Never say "see above" — always give the actual value
4. End EVERY response with: ANSWER: (letter) value
5. No rambling, no exploring wrong paths"""


def _safe_eval(code: str) -> tuple:
    allowed = {
        "math": math, "abs": abs, "round": round, "pow": pow,
        "sum": sum, "min": min, "max": max, "int": int,
        "float": float, "range": range, "len": len, "list": list,
    }
    try:
        local_vars = {}
        lines = [l.strip() for l in code.split("\n") if l.strip() and not l.startswith("#")]
        results = []
        for line in lines:
            try:
                r = eval(line, {"__builtins__": {}}, {**allowed, **local_vars})
                if r is not None:
                    results.append(f"{line} = {r}")
            except SyntaxError:
                exec(line, {"__builtins__": {}}, {**allowed, **local_vars})
        return True, "\n".join(results) or "OK"
    except Exception as e:
        return False, str(e)


def run_solver_agent(parsed_problem: dict, routing: dict, retrieved_context: str = "") -> dict:
    retriever = get_retriever()
    query = f"{parsed_problem.get('topic', '')} {parsed_problem.get('problem_text', '')}"
    chunks = retriever.retrieve(query, top_k=3)
    context_str = retriever.format_context(chunks)
    sources = [{"source": c["source"], "score": round(c["score"], 3)} for c in chunks]

    problem = parsed_problem.get("problem_text", "")
    mcq = parsed_problem.get("mcq_options", {})

    options_str = ""
    if mcq:
        options_str = "\nMCQ OPTIONS (you MUST pick one):\n"
        for k, v in mcq.items():
            options_str += f"  ({k}) {v}\n"

    prompt = f"""Solve this JEE problem. Be direct and concise.

PROBLEM: {problem}
{options_str}
RELEVANT FORMULAS:
{context_str}

Solve now in max 8 steps. End with: ANSWER: (letter) value"""

    response = call_llm(prompt, SYSTEM, temperature=0.0, max_tokens=700)

    # Run Python blocks if any
    py_blocks = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
    python_results = []
    for block in py_blocks:
        ok, result = _safe_eval(block)
        python_results.append({"code": block.strip(), "result": result, "success": ok})

    # Extract final answer
    final_answer = ""
    for line in reversed(response.split("\n")):
        if line.strip().upper().startswith("ANSWER:"):
            final_answer = line.split(":", 1)[1].strip()
            break

    if not final_answer:
        if mcq:
            final_answer = f"See solution — options: {mcq}"
        else:
            final_answer = "See solution above."

    return {
        "solution_text": response,
        "final_answer": final_answer,
        "python_results": python_results,
        "retrieved_sources": sources,
        "retrieved_chunks": chunks,
    }
