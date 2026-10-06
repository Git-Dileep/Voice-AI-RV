# Voice AI Hackathon: Baseline Architecture

[![Build Status](https://img.shields.io/badge/build-passing-success.svg)](#)
[![Latency Target](https://img.shields.io/badge/latency-under_1.5s-blue.svg)](#)
[![Architecture](https://img.shields.io/badge/architecture-event--driven-orange.svg)](#)

> **Objective:** Construct a domain-agnostic, low-latency voice-agent core prior to the hackathon event. Upon release of the problem statement, the team will modify only the persona, knowledge base, and tools. The core infrastructure remains strictly untouched.

## Architecture Overview

Speech enters at the top and reply audio leaves at the bottom. The orchestrator retrieves knowledge, feeds the telemetry dashboard, and executes tools through the integration layer.

```text
[ Mic + VAD ] ---> [ Streaming STT ] ---> [ Orchestrator ] ---> [ Streaming TTS ] ---> [ Speaker ]
                                                |
                                                +---> [ Knowledge Base (RAG) ]
                                                +---> [ Live Dashboard ]
                                                +---> [ Tool Executor ] ---> [ Integrations ]
```

## Core Design Principles

* **Voice-First Paradigm:** Demonstrations must showcase capabilities that are slow or impossible without speech (e.g., hands-busy, eyes-busy, low-literacy users, or driving live software systems).
* **Aggressive Streaming:** Speech-to-text (STT) partials, LLM tokens, and text-to-speech (TTS) chunks must overlap. The baseline target is under 1.5 seconds from the end of user speech to the first audio output.
* **Resilient Infrastructure:** The system is cloud-first but local-twin ready. Every component has an offline fallback (e.g., Whisper, Piper, local LLM) to completely mitigate venue Wi-Fi risks.
* **Explicit Execution:** The LLM manages conversation while tools handle execution. All database and application actions operate as explicit, validated function calls.
* **System Transparency:** A live dashboard exposes system telemetry (transcripts, language identification, tool calls, and per-stage latency) to make the backend intelligence visible to the jury.
* **Failsafe Operations:** Low confidence triggers a clarifying prompt. High-risk actions require spoken confirmation prior to execution.

## Team Matrix & Responsibilities

Work is strictly partitioned by architectural layer to maximize parallel development.

| Owner | Domain | Key Deliverables |
| :--- | :--- | :--- |
| **Person A** | **Voice Core** | Audio I/O, primary/fallback `VoiceProvider` interface, VAD/barge-in logic, language ID, and per-stage latency timers. |
| **Person B** | **Agent Brain** | Orchestrator loop, prompt safety guardrails, session memory, local RAG integration, and the centralized `tools.yaml`. |
| **Person C** | **Integrations** | Action API backend, mock services, live web dashboard, environment panel, and the one-command launch script. |

## Immutable Data Contracts

To prevent integration blockers, the following contracts are frozen from Day 1 and must be strictly adhered to by all team members:

### 1. The Provider Interface (`core/interfaces.py`)
```python
def start(self): ...
def on_partial(self, text: str): ...
def on_final(self, text: str, lang: str, confidence: float): ...
def on_tool_call(self, name: str, args: dict): ...
def speak(self, text: str): ...
def interrupt(self): ...
```

### 2. Tool Definitions (`brain/tools.yaml`)
This file acts as the single source of truth, dynamically exported to every provider's required format.
```yaml
- name: "update_record"
  description: "Updates a database record."
  risk_level: "confirm" # Options: 'safe' or 'confirm'
  handler_path: "/api/actions/update_record"
  parameters: ...
```

### 3. Action API Payload
All tool handlers route through a standardized endpoint.
**Endpoint:** `POST /api/actions/<tool>`
**Expected Response:**
```json
{
  "ok": true,
  "result": { "status": "updated" },
  "error": null
}
```

### 4. Dashboard Event Stream
Pushed via WebSocket for real-time telemetry rendering.
```json
{
  "ts": 1696500000.123,
  "stage": "stt_partial",
  "data": { "text": "turn on the" }
}
```

## Quickstart & Operations

**1. Environment Setup**
Ensure `.env` is populated with the required API credentials (e.g., ElevenLabs, Gemini).
```bash
pip install -r requirements.txt
```

**2. Evaluate Providers**
Run the benchmark script against the standardized 20-utterance test set to compare primary versus local fallback latency and accuracy.
```bash
python tests/benchmark.py
```

**3. Launch the Stack**
This command initializes the FastAPI backend, loads the orchestrator loop, and serves the frontend dashboard.
```bash
./run.sh
```

## Pivot Playbook

When the hackathon problem statement is released, the core architecture remains untouched. Execute these four steps to adapt:

- [ ] **Define the Persona:** Update system prompts and assign the appropriate agent voice.
- [ ] **Swap the Knowledge Base:** Move domain-specific PDFs or text files into `/brain/knowledge`.
- [ ] **Configure Tools:** Update `tools.yaml` and wire the corresponding API endpoints in `integrations/action_api.py`.
- [ ] **Set Language Focus:** Configure STT/TTS language parameters based on the target demographic of the problem statement.
