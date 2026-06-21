"""
VoxAgent - Shared voice engine
Reusable functions for transcription, LLM reasoning, and speech synthesis.
Used by both the Stage 1 CLI script and the Stage 2 FastAPI WebSocket server.
"""

import os
import tempfile

import edge_tts
from faster_whisper import WhisperModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from tools import get_weather, web_search

load_dotenv()

VOICE = "en-GB-RyanNeural"  
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are JARVIS, a calm, formal, dryly witty AI assistant. "
    "Address the user as 'sir' or 'ma'am'. Keep responses concise, "
    "understated, and intelligent. Never use filler phrases like "
    "'great question' or 'I'd be happy to help'. Get straight to the point. "
    "You have access to real tools for weather and web search — use them "
    "whenever the user asks about current conditions, facts, or anything "
    "you cannot answer reliably from memory alone. Do not guess when a "
    "tool can give you the real answer."
)

whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

# Stage 3: LangGraph ReAct agent, reasons about whether to call a tool before replying
llm = ChatGroq(model=GROQ_MODEL, temperature=0.4)
agent = create_react_agent(llm, tools=[get_weather, web_search])


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str:
    """Save incoming audio bytes to a temp file and transcribe with faster-whisper.
    faster-whisper decodes via PyAV (bundled ffmpeg), so webm/opus from the
    browser's MediaRecorder works directly without extra conversion."""
    tmp_path = tempfile.mktemp(suffix=suffix)
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        segments, _ = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            initial_prompt=(
                "The user is in Kerala, India. Places mentioned may include "
                "Palakkad, Perintalmanna, Kochi, Kozhikode, Thrissur, Bengaluru, "
                "Chennai, Mumbai, Delhi."
            ),
        )
        text = " ".join(segment.text.strip() for segment in segments)
    finally:
        os.remove(tmp_path)

    return text


def get_response(user_text: str, history: list) -> str:
    """Run the message through the LangGraph agent, which decides whether to
    call a tool (weather, web search) before producing a final reply."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=user_text))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


async def synthesize(text: str, voice: str = VOICE) -> bytes:
    """Generate speech audio (mp3 bytes) from text using edge-tts."""
    tmp_path = tempfile.mktemp(suffix=".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)

    return audio_bytes