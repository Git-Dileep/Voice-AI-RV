Voice AI Hackathon: Baseline Architecture

Welcome to the baseline repository. Our primary objective is to construct a domain-agnostic voice-agent core prior to the hackathon event. When the problem statement is released, the team will only modify the persona, knowledge base, and tools, ensuring the core infrastructure remains untouched.

Design Principles

Voice-First Design: The demonstration must highlight use cases that are significantly enhanced by or rely entirely on speech (e.g., hands-busy, eyes-busy, low-literacy, or multilingual users).

Low Latency Streaming: Speech-to-text (STT) partials, LLM tokens, and text-to-speech (TTS) chunks must overlap. The target metric is under 1.5 seconds from the end of user speech to the first audio output.

Resilient Infrastructure: Every component must have an offline, local equivalent to mitigate the risk of venue network failures.

Explicit Execution: The LLM manages conversation while tools handle execution. All API, database, and application actions are explicit, validated function calls.

System Transparency: A live dashboard will display transcripts, detected language, tool calls, and per-stage latency to demonstrate system intelligence to the jury.

Failsafe Operations: The system must gracefully ask for clarification upon low confidence and demand spoken confirmation for any high-risk execution actions.

Project Structure and Work Split

To maximize parallel development, work is divided by architectural layer rather than feature.

Person A: Voice Core (/core)

Scope: Audio input/output, provider abstraction, latency optimization, and offline fallbacks.

interfaces.py: Defines the VoiceProvider contract.

vad.py: Manages voice activity detection, barge-in, and noise gating.

stt.py / tts.py: Streaming implementations for primary (ElevenLabs/Gemini) and fallback (Whisper/Piper) providers.

Person B: Agent Brain (/brain)

Scope: Prompt engineering, orchestrator logic, knowledge retrieval, and safety protocols.

orchestrator.py: Manages the LLM, tool calling, and session memory.

tools.yaml: The single source of truth for all tools, formatted for export to provider schemas.

knowledge/: A swappable directory for local Retrieval-Augmented Generation (RAG) using Chroma or FAISS.

Person C: Integrations and Demo (/integrations & /dashboard)

Scope: Action API, live user interface, and overall packaging.

action_api.py: FastAPI backend handling tool execution and state changes.

dashboard/: Web client hosting microphone capture, the live event stream UI, and the simulated environment panel.

Core Technical Contracts

To prevent integration blockers, the following interfaces are strictly defined and must not be altered:

VoiceProvider Interface (core/interfaces.py):
Implementations must provide: start(), on_partial(text), on_final(text, lang, conf), on_tool_call(name, args), speak(text), and interrupt().

Tool Definitions (brain/tools.yaml):
Each entry must specify: name, description, JSON-schema arguments, risk level (safe/confirm), and handler path.

Action API (integrations/action_api.py):
All tool handlers must call POST /api/actions/<tool> and return a standardized JSON response: {ok: bool, result: any, error: string}.

Dashboard Stream:
The WebSocket connection must push JSON data formatted as {ts: timestamp, stage: string, data: object} for every pipeline event.

Getting Started

Install Dependencies:
Ensure your .env file contains the required API credentials (e.g., ElevenLabs, Gemini).

pip install -r requirements.txt


Run the Benchmark:
Execute the day-one benchmark script to evaluate primary versus backup providers against the standardized 20-utterance test set.

python tests/benchmark.py


Launch the Stack:
This script initializes the FastAPI backend, loads the orchestrator, and serves the frontend dashboard.

./run.sh


Pivot Playbook

Upon receiving the hackathon problem statement, the core architecture remains frozen. The team will exclusively update the following four components:

System Prompt and Persona

Knowledge Base Documents (Add relevant domain files to /brain/knowledge)

tools.yaml and Associated Handlers

Target Language Configurations
