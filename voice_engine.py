"""
VoxAgent - Shared voice engine
Reusable functions for transcription, LLM reasoning, and speech synthesis.
Used by both the Stage 1 CLI script and the Stage 2 FastAPI WebSocket server.
"""

import os
import tempfile

import edge_tts
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

VOICE = "en-GB-RyanNeural"  # JARVIS-style British voice (alt: en-GB-ThomasNeural)
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are JARVIS, a calm, formal, dryly witty AI assistant. "
    "Address the user as 'sir' or 'ma'am'. Keep responses concise, "
    "understated, and intelligent. Never use filler phrases like "
    "'great question' or 'I'd be happy to help'. Get straight to the point."
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str:
    """Save incoming audio bytes to a temp file and transcribe with faster-whisper.
    faster-whisper decodes via PyAV (bundled ffmpeg), so webm/opus from the
    browser's MediaRecorder works directly without extra conversion."""
    tmp_path = tempfile.mktemp(suffix=suffix)
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        segments, _ = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments)
    finally:
        os.remove(tmp_path)

    return text


def get_response(user_text: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content


async def synthesize(text: str, voice: str = VOICE) -> bytes:
    """Generate speech audio (mp3 bytes) from text using edge-tts."""
    tmp_path = tempfile.mktemp(suffix=".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)

    return audio_bytes