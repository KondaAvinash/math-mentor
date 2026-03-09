"""
Agent 1: Parser Agent
Converts raw OCR/ASR/text input into structured math problem JSON.
"""
from utils.llm_client import call_llm_json

SYSTEM = """You are a JEE math problem parser.
Input may come from OCR (image) or speech and may have minor errors.
Your job: extract and structure the problem cleanly.
Return ONLY valid JSON — no explanation, no markdown."""


def run_parser_agent(raw_input: str, input_source: str = "text") -> dict:
    prompt = f"""Parse this JEE math problem into structured JSON.

Input source: {input_source}
Raw input:
\"\"\"{raw_input}\"\"\"

Extract:
1. Clean problem statement (fix obvious OCR errors)
2. MCQ options if present — extract (A)(B)(C)(D) values EXACTLY
3. Topic: algebra | probability | calculus | linear_algebra
4. Subtopic: sequences_and_series | quadratic | limits | matrices | permutations | etc
5. What needs to be found
6. Whether clarification is needed

Return this JSON:
{{
  "problem_text": "clean problem statement",
  "mcq_options": {{"A": "val", "B": "val", "C": "val", "D": "val"}},
  "topic": "algebra",
  "subtopic": "sequences_and_series",
  "variables": ["n", "d"],
  "constraints": [],
  "given_values": {{}},
  "what_to_find": "describe what to find",
  "needs_clarification": false,
  "clarification_request": ""
}}

If no MCQ options found, set mcq_options to {{}}.
If input is unclear or ambiguous, set needs_clarification to true."""

    result = call_llm_json(prompt, SYSTEM)

    # Apply defaults for missing keys
    defaults = {
        "problem_text": raw_input,
        "mcq_options": {},
        "topic": "algebra",
        "subtopic": "general",
        "variables": [],
        "constraints": [],
        "given_values": {},
        "what_to_find": "solve the problem",
        "needs_clarification": False,
        "clarification_request": "",
    }
    for k, v in defaults.items():
        result.setdefault(k, v)
    return result
