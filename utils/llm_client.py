"""
Groq LLM Client — fast LLaMA via Groq API.
Used by all 5 agents.
"""
import json
import time
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

_client = None

def get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def call_llm(prompt, system_prompt="", temperature=0.1, max_tokens=1024):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            response = get_client().chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(15)
                continue
            return f"[LLM Error] {str(e)}"


def call_llm_json(prompt, system_prompt="", temperature=0.1):
    system_prompt += "\n\nRespond ONLY with valid JSON. No markdown fences, no explanation."
    raw = call_llm(prompt, system_prompt, temperature, max_tokens=600)
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except Exception:
                pass
        return {"error": "JSON parse failed", "raw": raw}