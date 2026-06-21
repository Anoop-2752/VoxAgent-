# VoxAgent

A real-time, voice-driven AI assistant with agentic tool-calling, inspired by the calm, formal assistant archetype popularized by sci-fi (JARVIS-style persona, original implementation — not affiliated with or using any copyrighted character assets). You talk to it, it talks back, and unlike a simple chatbot wrapper, it can actually take real actions — checking live weather or searching the web — before answering.

Built end to end as a deployment-focused portfolio project: every component runs on free infrastructure, with no paid APIs anywhere in the stack.

## What it does

Hold a button, speak, and VoxAgent transcribes your voice, reasons about what you're asking (deciding on its own whether it needs a tool to answer accurately), and speaks a response back — all over a live WebSocket connection, with sub-few-second round-trip latency. You can interrupt it mid-reply by simply starting to talk again, the way you would in a real conversation.

## Features

- **Real-time voice conversation** — hold-to-talk capture in the browser, no page reloads, no REST round trips
- **Agentic tool use** — a LangGraph ReAct agent decides whether to call a tool or answer directly, instead of guessing from memory
  - Live weather lookup (Open-Meteo, free, no API key)
  - Live web search (DuckDuckGo via `ddgs`, free, no API key)
- **Persona-driven responses** — formal, concise, dryly witty system prompt, paired with a British TTS voice for a distinctive character
- **Barge-in interruption** — speaking while the assistant is replying stops its playback immediately and starts listening to you
- **Conversation memory** — full back-and-forth history maintained per session, sent to the agent on every turn
- **Resilient error handling** — every pipeline stage (transcription, reasoning, speech synthesis) is independently caught and reported, both in server logs and back to the browser, instead of crashing the connection
- **Tuned for accent and place-name accuracy** — uses an `initial_prompt` to bias transcription toward local place names rather than defaulting to whatever sounds phonetically closest in a generic English corpus
- **Latency instrumentation** — per-stage timing logged server-side (transcription / agent reasoning / speech synthesis) to make bottlenecks visible rather than guessed at

## Architecture

```
Browser (mic)
   │  MediaRecorder captures audio on button hold
   ▼
WebSocket  ──────────────────────────────────────────────►  FastAPI server
   │                                                              │
   │                                              1. faster-whisper transcribes audio
   │                                              2. LangGraph agent (Groq LLaMA 3.3 70B)
   │                                                 reasons, optionally calls a tool
   │                                                 (weather / web search), then replies
   │                                              3. edge-tts synthesizes the reply to speech
   ▼                                                              │
Browser plays reply audio  ◄─────────────────────────────────────┘
   (interruptible by holding the button again)
```

## Tech stack

| Layer | Technology |
|---|---|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small.en`, CPU, int8 quantized) |
| Reasoning / agent | [LangGraph](https://langchain-ai.github.io/langgraph/) `create_react_agent` + Groq (`llama-3.3-70b-versatile`) |
| Tools | Open-Meteo (geocoding + weather), DuckDuckGo search (`ddgs`) |
| Text-to-speech | [edge-tts](https://github.com/rany2/edge-tts) (`en-GB-RyanNeural`) |
| Backend | FastAPI, WebSockets, Uvicorn |
| Frontend | Vanilla JS, MediaRecorder API, native WebSocket API — no framework |

## Project structure

```
VoxAgent/
├── pipeline.py        # Stage 1: standalone CLI voice loop (no server, for quick testing)
├── voice_engine.py     # Shared core: transcription, agent reasoning, speech synthesis
├── tools.py             # Agent tools: get_weather, web_search
├── server.py            # FastAPI app + WebSocket endpoint
├── static/
│   └── index.html       # Browser frontend (hold-to-talk UI)
├── requirements.txt
├── .env                  # GROQ_API_KEY (not committed)
└── .gitignore
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Get a free Groq API key from [console.groq.com](https://console.groq.com), then create a `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```

4. Run the server:
   ```
   uvicorn server:app --reload
   ```

5. Open `http://localhost:8000`, allow microphone access, and hold the button to talk.

A standalone CLI version is also available for quick testing without the web UI:
```
python pipeline.py
```

## How it was built

The project was built in four incremental stages, each one a working, testable milestone rather than one large build:

**Stage 1 — Offline pipeline.** A single script proving the core loop works: record audio, transcribe with faster-whisper, send the text to Groq for a response, synthesize speech with edge-tts, and play it back. No web framework, no streaming — just confirming every component talks to every other component correctly.

**Stage 2 — Real-time web app.** The CLI script was rebuilt around a FastAPI WebSocket server and a browser frontend using the MediaRecorder API. Audio now flows in and out over a persistent connection instead of a script running on a local machine, with the recording/transcription/reasoning/synthesis logic refactored into a shared, reusable module (`voice_engine.py`).

**Stage 3 — Agentic tool-calling.** The plain LLM call was replaced with a LangGraph ReAct agent wired to two free tools — live weather and live web search — so the assistant can take real actions instead of only answering from what it already knows, and decides for itself when a tool is actually needed.

**Stage 4 — Interruption handling and tuning.** Added barge-in support so the assistant can be talked over mid-reply, added per-stage latency logging to diagnose slowness, and tuned the transcription model and decoding parameters (model size, beam search, an `initial_prompt` biased toward local place names) to balance response speed against transcription accuracy.

## Known limitations

- Interrupting the assistant stops its *audio playback* immediately, but doesn't cancel an in-flight request on the server — if you interrupt during the "thinking" phase, that response will still arrive and appear in the log a moment later.
- No persistent storage: conversation history resets when the WebSocket reconnects (e.g., on page refresh).
- Speech synthesis happens in full before playback starts, rather than streaming — a streaming TTS implementation would shave additional latency off replies.
- Tool set is currently limited to weather and web search; the agent architecture supports adding more tools easily.

## Possible future directions

- Streaming TTS for lower perceived latency
- True server-side request cancellation on interruption
- Additional tools (calendar, reminders, smart home, etc.)
- Deployment to a public free-tier host (Hugging Face Spaces or Render) for a live demo link