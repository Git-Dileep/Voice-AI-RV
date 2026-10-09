"""Basic integration tests for the headless Voice AI agent.

These tests validate the core pipeline components (VAD, tool execution,
sentence buffering, brain tool detection) WITHOUT requiring API keys or
a live microphone.

Run:  python tests/test_agent.py
"""

import sys
import os

# Ensure project root is on sys.path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.vad import VoiceActivityDetector
from integrations.tool_executor import execute_tool


# ──────────────────────────────────────────────
# VAD Tests
# ──────────────────────────────────────────────


def test_vad_silence():
    """VAD should return None for continuous silence."""
    vad = VoiceActivityDetector()
    silent = np.zeros(480, dtype=np.int16).tobytes()
    for _ in range(20):
        result = vad.process_chunk(silent)
        assert result is None
    assert not vad.is_speaking
    print("  ✅  VAD silence test passed")


def test_vad_speech_detection():
    """VAD should detect speech onset and return audio when speech ends."""
    vad = VoiceActivityDetector(
        energy_threshold=100,
        min_speech_duration=0.01,
        silence_duration=0.05,
    )

    # Generate loud frames (simulated speech)
    loud = (np.sin(np.arange(480) * 0.1) * 5000).astype(np.int16).tobytes()
    for _ in range(10):
        vad.process_chunk(loud)
    assert vad.is_speaking, "VAD should have detected speech onset"

    # Generate silent frames to end speech
    silent = np.zeros(480, dtype=np.int16).tobytes()
    utterance = None
    for _ in range(50):
        r = vad.process_chunk(silent)
        if r is not None:
            utterance = r
            break

    assert utterance is not None, "VAD should return utterance audio on speech end"
    assert len(utterance) > 0
    print("  ✅  VAD speech detection test passed")


def test_vad_reset():
    """VAD reset should return to idle state."""
    vad = VoiceActivityDetector(energy_threshold=100, min_speech_duration=0.01)
    loud = (np.sin(np.arange(480) * 0.1) * 5000).astype(np.int16).tobytes()
    for _ in range(10):
        vad.process_chunk(loud)
    assert vad.is_speaking

    vad.reset()
    assert not vad.is_speaking
    print("  ✅  VAD reset test passed")


# ──────────────────────────────────────────────
# Tool Executor Tests
# ──────────────────────────────────────────────


def test_tool_executor():
    """Tool executor should handle known and unknown tools correctly."""
    # Known tool — get_weather
    result = execute_tool("get_weather", {"location": "Mumbai"})
    assert result["ok"] is True
    assert result["location"] == "Mumbai"

    # Known tool — update_record
    result = execute_tool("update_record", {"record_id": "001", "value": "test"})
    assert result["ok"] is True

    # Unknown tool
    result = execute_tool("nonexistent_tool", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]

    print("  ✅  Tool executor tests passed")


# ──────────────────────────────────────────────
# Sentence Buffer Tests
# ──────────────────────────────────────────────


def test_sentence_buffer():
    """sentence_buffer should split streaming tokens into sentences."""
    from main import sentence_buffer

    # Simulate a token stream for: "Hello there. How are you? I'm great."
    tokens = [
        "Hello",
        " there",
        ". ",
        "How",
        " are",
        " you",
        "? ",
        "I'm",
        " great",
        ".",
    ]

    sentences = list(sentence_buffer(iter(tokens)))
    assert len(sentences) >= 2, f"Expected ≥2 sentences, got {len(sentences)}: {sentences}"
    assert sentences[0] == "Hello there."
    print(f"  ✅  Sentence buffer test passed — {sentences}")


def test_sentence_buffer_no_trailing_space():
    """sentence_buffer should flush remaining text without trailing boundary."""
    from main import sentence_buffer

    tokens = ["Just", " one", " sentence"]
    sentences = list(sentence_buffer(iter(tokens)))
    assert len(sentences) == 1
    assert sentences[0] == "Just one sentence"
    print("  ✅  Sentence buffer flush test passed")


# ──────────────────────────────────────────────
# Brain Tool Detection Tests
# ──────────────────────────────────────────────


def test_brain_tool_detection():
    """Brain should detect tool calls from keywords."""
    from brain.orchestrator import AgentBrain

    # Create a minimal brain instance (no actual API calls)
    brain = AgentBrain.__new__(AgentBrain)
    brain.tools_config = [
        {"name": "get_weather", "risk_level": "safe"},
        {"name": "update_record", "risk_level": "confirm"},
    ]
    brain.pending_confirmation = None

    # Weather keyword
    call = brain._detect_tool_call("What's the weather like?")
    assert call is not None and call["name"] == "get_weather"

    # Update record keywords
    call = brain._detect_tool_call("Please update the record now")
    assert call is not None and call["name"] == "update_record"

    # No tool match
    call = brain._detect_tool_call("Hello how are you doing today")
    assert call is None

    print("  ✅  Brain tool detection tests passed")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  🧪  Voice AI Agent — Test Suite\n")

    test_vad_silence()
    test_vad_speech_detection()
    test_vad_reset()
    test_tool_executor()
    test_sentence_buffer()
    test_sentence_buffer_no_trailing_space()
    test_brain_tool_detection()

    print("\n  ✅  All tests passed!\n")
