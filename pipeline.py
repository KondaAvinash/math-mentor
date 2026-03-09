"""
Pipeline — orchestrates all 5 agents in sequence.
"""
from agents.parser_agent import run_parser_agent
from agents.router_agent import run_router_agent
from agents.solver_agent import run_solver_agent
from agents.verifier_agent import run_verifier_agent
from agents.explainer_agent import run_explainer_agent
from memory.memory_store import save_to_memory, find_similar_problems


def run_pipeline(
    raw_input: str,
    input_source: str = "text",
    ocr_confidence: float = 1.0,
    asr_confidence: float = 1.0,
    force_hitl: bool = False,
) -> dict:
    """
    Run the full 5-agent pipeline.
    Returns complete result dict with agent traces.
    """
    trace = []

    # Check memory for similar problems first
    similar = find_similar_problems(raw_input)

    # Agent 1: Parse
    trace.append({"agent": "Parser", "status": "running"})
    parsed = run_parser_agent(raw_input, input_source)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {
        "topic": parsed.get("topic"),
        "subtopic": parsed.get("subtopic"),
        "needs_clarification": parsed.get("needs_clarification"),
    }

    # Check HITL triggers
    hitl_reason = None
    if force_hitl:
        hitl_reason = "User requested review"
    elif ocr_confidence < 0.55 and input_source == "ocr":
        hitl_reason = f"Low OCR confidence ({ocr_confidence:.0%})"
    elif asr_confidence < 0.50 and input_source == "audio":
        hitl_reason = f"Low ASR confidence ({asr_confidence:.0%})"
    elif parsed.get("needs_clarification"):
        hitl_reason = parsed.get("clarification_request", "Problem needs clarification")

    # Agent 2: Route
    trace.append({"agent": "Router", "status": "running"})
    routing = run_router_agent(parsed)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {
        "strategy": routing.get("strategy"),
        "difficulty": routing.get("difficulty"),
    }

    # Agent 3: Solve
    trace.append({"agent": "Solver", "status": "running"})
    solver_result = run_solver_agent(parsed, routing)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {
        "final_answer": solver_result.get("final_answer"),
        "sources_used": len(solver_result.get("retrieved_sources", [])),
    }

    # Agent 4: Verify
    trace.append({"agent": "Verifier", "status": "running"})
    verifier_result = run_verifier_agent(parsed, solver_result)
    trace[-1]["status"] = "done"
    trace[-1]["output"] = {
        "confidence": verifier_result.get("confidence"),
        "is_correct": verifier_result.get("is_correct"),
    }

    # Trigger HITL if verifier is not confident
    if verifier_result.get("needs_hitl") and not hitl_reason:
        hitl_reason = f"Verifier confidence {verifier_result.get('confidence', 0):.0%} < threshold"

    # Agent 5: Explain
    trace.append({"agent": "Explainer", "status": "running"})
    explainer_result = run_explainer_agent(parsed, solver_result, verifier_result)
    trace[-1]["status"] = "done"

    # Save to memory
    problem_id = save_to_memory(raw_input, parsed, solver_result, verifier_result, input_source)

    return {
        "problem_id": problem_id,
        "parsed_problem": parsed,
        "routing": routing,
        "solver_result": solver_result,
        "verifier_result": verifier_result,
        "explainer_result": explainer_result,
        "agent_trace": trace,
        "hitl_triggered": hitl_reason is not None,
        "hitl_reason": hitl_reason,
        "similar_problems": similar,
    }
