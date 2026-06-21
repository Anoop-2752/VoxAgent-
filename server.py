"""
VoxAgent - Stage 2: Real-time voice loop over WebSockets
Browser mic -> WebSocket -> transcribe -> reason -> speak -> WebSocket -> browser playback
"""

import json
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from voice_engine import transcribe_bytes, get_response, synthesize

app = FastAPI(title="VoxAgent")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    history = []
    print("Client connected.")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            print(f"Received {len(audio_bytes)} bytes of audio.")

            try:
                user_text = transcribe_bytes(audio_bytes)
            except Exception as e:
                print("Transcription failed:")
                traceback.print_exc()
                await websocket.send_text(json.dumps({"type": "error", "message": f"Transcription failed: {e}"}))
                continue

            if not user_text.strip():
                await websocket.send_text(json.dumps({"type": "error", "message": "No speech detected."}))
                continue

            await websocket.send_text(json.dumps({"type": "transcript", "text": user_text}))

            try:
                reply = get_response(user_text, history)
            except Exception as e:
                print("LLM call failed:")
                traceback.print_exc()
                await websocket.send_text(json.dumps({"type": "error", "message": f"LLM call failed: {e}"}))
                continue

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})

            await websocket.send_text(json.dumps({"type": "reply", "text": reply}))

            try:
                audio_reply = await synthesize(reply)
                await websocket.send_bytes(audio_reply)
            except Exception as e:
                print("Speech synthesis failed:")
                traceback.print_exc()
                await websocket.send_text(json.dumps({"type": "error", "message": f"Speech synthesis failed: {e}"}))

    except WebSocketDisconnect:
        print("Client disconnected.")