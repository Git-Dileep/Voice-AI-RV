"""Voice Activity Detection — energy-based RMS analysis.

Detects speech onset and endpoint in a continuous 16 kHz PCM audio stream.
Returns the complete utterance audio buffer when the speaker pauses.

Replaces the original multi-callback VAD state machine with a simpler
pull-based API suited to the headless conversation loop.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class VoiceActivityDetector:
    """Energy/RMS-based VAD for real-time speech segmentation.

    Usage:
        vad = VoiceActivityDetector()
        for chunk in mic.read_chunks():
            utterance = vad.process_chunk(chunk)
            if utterance is not None:
                # Full spoken utterance ready for transcription
                transcribe(utterance)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        energy_threshold: float = 500.0,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.15,
        pre_speech_padding: float = 0.3,
    ):
        """Initialize the VAD.

        Args:
            sample_rate: Audio sample rate in Hz.
            energy_threshold: RMS amplitude above which a frame is classified
                as speech (0–32767 for 16-bit audio).
            silence_duration: Seconds of continuous silence required to mark
                the end of an utterance.
            min_speech_duration: Seconds of continuous speech required to
                confirm onset (rejects brief noise spikes).
            pre_speech_padding: Seconds of audio buffered before onset to
                capture initial consonants.
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.pre_speech_padding = pre_speech_padding

        self._is_speaking = False
        self._speech_time = 0.0
        self._silence_time = 0.0
        self._speech_chunks: list[bytes] = []
        self._pre_buffer: list[bytes] = []

    @property
    def is_speaking(self) -> bool:
        """True while the user is actively speaking."""
        return self._is_speaking

    def process_chunk(self, chunk: bytes) -> bytes | None:
        """Feed a raw 16-bit PCM audio chunk through the VAD.

        Args:
            chunk: Raw 16-bit PCM audio bytes.

        Returns:
            Complete utterance audio (bytes) when speech ends, or None.
        """
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return None

        chunk_duration = samples.size / self.sample_rate
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        is_voice = rms >= self.energy_threshold

        if not self._is_speaking:
            return self._handle_silence(chunk, is_voice, chunk_duration)
        else:
            return self._handle_speech(chunk, is_voice, chunk_duration)

    # ── Internal State Handlers ──

    def _handle_silence(self, chunk: bytes, is_voice: bool, dt: float) -> None:
        """Process a chunk while in SILENCE state."""
        # Maintain a rolling pre-speech ring buffer
        max_pre = max(1, int(self.pre_speech_padding / dt))
        self._pre_buffer.append(chunk)
        if len(self._pre_buffer) > max_pre:
            self._pre_buffer.pop(0)

        if is_voice:
            self._speech_time += dt
            if self._speech_time >= self.min_speech_duration:
                # Transition → SPEECH
                self._is_speaking = True
                self._silence_time = 0.0
                self._speech_chunks = list(self._pre_buffer)
                self._pre_buffer.clear()
        else:
            self._speech_time = 0.0

        return None

    def _handle_speech(self, chunk: bytes, is_voice: bool, dt: float) -> bytes | None:
        """Process a chunk while in SPEECH state."""
        self._speech_chunks.append(chunk)

        if is_voice:
            self._silence_time = 0.0
        else:
            self._silence_time += dt
            if self._silence_time >= self.silence_duration:
                # Transition → SILENCE — return completed utterance
                utterance = b"".join(self._speech_chunks)
                self.reset()
                return utterance

        return None

    def reset(self):
        """Reset to idle state."""
        self._is_speaking = False
        self._speech_time = 0.0
        self._silence_time = 0.0
        self._speech_chunks.clear()
        self._pre_buffer.clear()
