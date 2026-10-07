"""Voice Core Interface Contracts.

This module defines the frozen Day-1 Voice Core contracts and telemetry data
structures for the Voice AI Hackathon baseline architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict


@dataclass
class TelemetryEvent:
    """Standardized telemetry event payload streamed over WebSocket to the dashboard."""

    stage: str
    data: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to a dictionary matching the dashboard schema."""
        return {
            "ts": self.ts,
            "stage": self.stage,
            "data": self.data,
        }


@dataclass
class STTResult:
    """Represents a finalized speech-to-text recognition result."""

    text: str
    lang: str
    confidence: float


@dataclass
class ToolCall:
    """Represents a tool invocation emitted by a native-audio voice provider."""

    name: str
    args: Dict[str, Any]


class VoiceProvider(ABC):
    """Abstract base class defining the frozen Voice Core provider contract.

    This interface abstracts audio input/output, streaming Speech-To-Text (STT),
    streaming Text-To-Speech (TTS), voice activity detection (VAD), and turn-taking.
    Concrete implementations back this interface using cloud services (e.g., ElevenLabs,
    Gemini Live) or local offline pipelines (e.g., faster-whisper, Piper, Silero VAD).
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the voice provider session, audio streaming, and event loops."""
        pass

    @abstractmethod
    def on_partial(self, text: str) -> None:
        """Callback invoked when streaming STT yields interim/partial transcription results.

        Args:
            text: Interim recognized speech text.
        """
        pass

    @abstractmethod
    def on_final(self, text: str, lang: str, confidence: float) -> None:
        """Callback invoked when a speech turn concludes with a finalized transcript.

        Args:
            text: Finalized recognized speech text.
            lang: Detected ISO language code (e.g., 'en', 'hi', 'kn').
            confidence: Recognition confidence score between 0.0 and 1.0.
        """
        pass

    @abstractmethod
    def on_tool_call(self, name: str, args: dict) -> None:
        """Callback invoked when the voice provider emits a native tool call request.

        Args:
            name: The target tool name matching definitions in tools.yaml.
            args: Dictionary of validated tool arguments.
        """
        pass

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesize and stream audio output for the specified text to the speaker.

        Speech should stream sentence-by-sentence to keep time-to-first-audio
        under the 1.5-second baseline target.

        Args:
            text: Response text to synthesize and play aloud.
        """
        pass

    @abstractmethod
    def interrupt(self) -> None:
        """Immediately abort active audio playback upon barge-in or user speech interruption.

        Flushes active audio playback queues and halts current synthesis.
        """
        pass
