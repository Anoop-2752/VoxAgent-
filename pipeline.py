"""
VoxAgent - Stage 1: Offline voice pipeline
Record -> Transcribe (faster-whisper) -> Reason (Groq LLaMA 3.3-70B) -> Speak (edge-tts)

No WebSockets yet, no streaming. Just proving every component talks to
every other component correctly before adding real-time complexity.
"""

import os
import asyncio
import tempfile
import wave

import sounddevice as sd
import pygame
import edge_tts
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
VOICE = "en-GB-RyanNeural"  # JARVIS-style British voice (alt: en-GB-ThomasNeural)
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are JARVIS, a calm, formal, dryly witty AI assistant. "
    "Address the user as 'sir' or 'ma'am'. Keep responses concise, "
    "understated, and intelligent. Never use filler phrases like "
    "'great question' or 'I'd be happy to help'. Get straight to the point."
)

# ---- Setup ----
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
pygame.mixer.init()


def record_audio(duration=RECORD_SECONDS, samplerate=SAMPLE_RATE):
    print(f"\nListening... (speak now, {duration}s)")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    print("Recording done.")

    tmp_path = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return tmp_path


def transcribe(audio_path):
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)
    print(f"You said: {text}")
    return text


def get_response(user_text, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
    )
    reply = completion.choices[0].message.content
    print(f"JARVIS: {reply}")
    return reply


async def speak(text, voice=VOICE):
    tmp_path = tempfile.mktemp(suffix=".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_path)

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.unload()  # release file handle (Windows needs this before delete)
    os.remove(tmp_path)


def main():
    print("=== VoxAgent Stage 1: Offline Pipeline ===")
    print("Press Enter to record, Ctrl+C to quit.\n")

    history = []
    while True:
        try:
            input("Press Enter to talk to JARVIS...")
        except KeyboardInterrupt:
            print("\nGoodbye, sir.")
            break

        audio_path = record_audio()
        user_text = transcribe(audio_path)
        os.remove(audio_path)

        if not user_text.strip():
            print("No speech detected, try again.")
            continue

        reply = get_response(user_text, history)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        asyncio.run(speak(reply))


if __name__ == "__main__":
    main()