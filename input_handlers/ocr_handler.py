"""
OCR Handler — uses Groq Vision (LLaMA-4 Scout) to read math from images.
No Gemini needed. Groq is free and fast.
"""
import base64
import io
from PIL import Image
from config import GROQ_API_KEY

LOW_CONFIDENCE_THRESHOLD = 0.55

VISION_PROMPT = """You are a math OCR assistant for JEE exam problems.
Look at this image carefully and extract the COMPLETE math problem.

Extract:
1. Full question text word for word
2. All numbers in sequences exactly (e.g. 3, 7, 11, ..., 113)
3. MCQ options EXACTLY as written: (A)... (B)... (C)... (D)...
4. All mathematical symbols

Return ONLY the extracted text. Nothing else. No explanation."""


def image_to_base64(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w > 1024:
        scale = 1024.0 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_text_from_image(image_bytes: bytes) -> tuple:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        b64 = image_to_base64(image_bytes)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.0,
        )
        extracted = response.choices[0].message.content.strip()
        confidence = 0.92 if extracted and len(extracted) > 10 else 0.3
        return extracted, confidence
    except Exception as e:
        return f"[Vision Error] {str(e)}", 0.0


def preprocess_math_text(raw: str) -> str:
    import re
    text = raw
    for old, new in [("²","^2"),("³","^3"),("×","*"),("÷","/"),("√","sqrt"),("π","pi")]:
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text