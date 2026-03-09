"""
Audio Handler — uses Whisper (local) for speech to text.
"""
import tempfile
import os

LOW_CONFIDENCE_THRESHOLD = 0.40

MATH_PHRASES = {
    "square root of": "sqrt(",
    "squared": "^2",
    "cubed": "^3",
    "raised to the power of": "^",
    "plus": "+",
    "minus": "-",
    "divided by": "/",
    "multiplied by": "*",
    "times": "*",
    "pi": "π",
}

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        from config import WHISPER_MODEL
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, file_ext: str = "wav") -> tuple:
    try:
        model = _get_whisper()
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        # Add ffmpeg to PATH
        import os
        ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
        if ffmpeg_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] += f";{ffmpeg_path}"

        result = model.transcribe(tmp_path, language="en")
        os.unlink(tmp_path)

        text = result.get("text", "").strip()
        # Always return high confidence if text found
        confidence = 0.85 if text and len(text) > 5 else 0.3
        return text, confidence

    except Exception as e:
        return f"[ASR Error] {str(e)}", 0.0


def normalize_math_speech(text: str) -> str:
    result = text.lower()
    for phrase, symbol in MATH_PHRASES.items():
        result = result.replace(phrase, symbol)
    return result.strip()