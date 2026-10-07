"""Voice Activity Detection (VAD) and Barge-In Layer.

This module provides a provider-agnostic, energy/RMS-based Voice Activity Detection
state machine using NumPy. It detects speech start, continued speech, and speech end,
and triggers barge-in interruption when speech occurs while agent audio is playing.
"""

from collections import deque
from enum import Enum
import logging
import threading
from typing import Callable, Deque, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class VADState(str, Enum):
    """VAD state machine states."""

    SILENCE = "silence"
    SPEECH = "speech"


class VAD:
    """Energy/RMS-based Voice Activity Detector and barge-in controller.

    Operates on 16-bit PCM audio chunks (typically 16 kHz mono) and tracks voice
    activity using root-mean-square (RMS) energy.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 600.0,
        silence_duration: float = 0.7,
        min_speech_duration: float = 0.1,
        pre_speech_duration: float = 0.2,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[bytes], None]] = None,
        on_speech_chunk: Optional[Callable[[bytes, float], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        is_agent_speaking: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Initialize the VAD state machine.

        Args:
            sample_rate: Audio sampling frequency in Hz (default: 16000).
            energy_threshold: Linear RMS amplitude threshold (0-32767) above which
                a frame is classified as speech.
            silence_duration: Continuous silence duration in seconds required to
                trigger a speech end transition.
            min_speech_duration: Continuous speech duration in seconds required to
                confirm speech start and reject brief noise spikes/clicks.
            pre_speech_duration: Seconds of audio buffered before speech onset to
                prevent clipping the initial consonants/plosives.
            on_speech_start: Callback invoked when speech onset is detected.
            on_speech_end: Callback invoked when speech concludes; receives the complete
                accumulated utterance audio as 16-bit PCM bytes.
            on_speech_chunk: Optional callback invoked for every active speech chunk;
                receives (chunk_bytes, rms_energy).
            on_barge_in: Callback invoked immediately when speech starts while the
                agent audio is currently playing.
            is_agent_speaking: Callable returning True if agent playback is currently
                active (e.g., AudioOutput.is_playing).
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.pre_speech_duration = pre_speech_duration

        # Event callbacks
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_speech_chunk = on_speech_chunk
        self.on_barge_in = on_barge_in
        self.is_agent_speaking = is_agent_speaking

        # State tracking
        self._state = VADState.SILENCE
        self._current_energy = 0.0
        self._lock = threading.Lock()

        # Timing and frame counters
        self._consecutive_speech_time = 0.0
        self._consecutive_silence_time = 0.0

        # Buffers
        self._pre_speech_buffer: Deque[bytes] = deque()
        self._speech_buffer: list[bytes] = []

    @property
    def state(self) -> VADState:
        """Return the current VAD state (VADState.SILENCE or VADState.SPEECH)."""
        with self._lock:
            return self._state

    @property
    def is_speaking(self) -> bool:
        """Return True if user speech is currently active."""
        with self._lock:
            return self._state == VADState.SPEECH

    @property
    def current_energy(self) -> float:
        """Return the RMS energy of the most recent audio frame."""
        with self._lock:
            return self._current_energy

    def compute_rms(self, samples: np.ndarray) -> float:
        """Compute the Root Mean Square (RMS) energy of 16-bit PCM samples.

        Args:
            samples: 1D NumPy array of audio samples.

        Returns:
            RMS energy as a float between 0.0 and 32767.0.
        """
        if samples.size == 0:
            return 0.0
        # Cast to float64 to prevent 16-bit integer overflow during squaring
        mean_square = np.mean(samples.astype(np.float64) ** 2)
        return float(np.sqrt(mean_square))

    def process_chunk(self, chunk: Union[bytes, np.ndarray]) -> VADState:
        """Process a single 16-bit PCM audio chunk through the VAD state machine.

        This method is non-blocking and safe to use directly as an audio stream callback.

        Args:
            chunk: Raw 16-bit PCM bytes or 1D int16 NumPy array.

        Returns:
            The current VADState after processing this chunk.
        """
        if isinstance(chunk, np.ndarray):
            raw_bytes = chunk.astype(np.int16).tobytes()
            samples = chunk.astype(np.int16)
        else:
            raw_bytes = chunk
            # Drop trailing odd byte if chunk length is not even
            if len(raw_bytes) % 2 != 0:
                raw_bytes = raw_bytes[: len(raw_bytes) - 1]
            if not raw_bytes:
                return self._state
            samples = np.frombuffer(raw_bytes, dtype=np.int16)

        if samples.size == 0:
            return self._state

        # Calculate chunk duration in seconds
        chunk_duration = samples.size / self.sample_rate

        # Calculate RMS energy
        rms = self.compute_rms(samples)

        with self._lock:
            self._current_energy = rms
            is_voice = rms >= self.energy_threshold

            if self._state == VADState.SILENCE:
                self._handle_silence_state(raw_bytes, is_voice, chunk_duration)
            else:
                self._handle_speech_state(raw_bytes, rms, is_voice, chunk_duration)

            return self._state

    def _handle_silence_state(self, raw_bytes: bytes, is_voice: bool, chunk_duration: float) -> None:
        """Handle state transitions while currently in SILENCE."""
        # Update rolling pre-speech ring buffer
        max_pre_speech_chunks = max(1, int(self.pre_speech_duration / chunk_duration))
        self._pre_speech_buffer.append(raw_bytes)
        while len(self._pre_speech_buffer) > max_pre_speech_chunks:
            self._pre_speech_buffer.popleft()

        if is_voice:
            self._consecutive_speech_time += chunk_duration
            if self._consecutive_speech_time >= self.min_speech_duration:
                # Transition: SILENCE -> SPEECH
                self._state = VADState.SPEECH
                self._consecutive_silence_time = 0.0

                # Prepend pre-speech padding so initial consonants are retained
                self._speech_buffer = list(self._pre_speech_buffer)
                self._pre_speech_buffer.clear()

                # Trigger barge-in immediately if agent is currently speaking
                self._check_barge_in()

                # Notify speech start callback
                if self.on_speech_start is not None:
                    try:
                        self.on_speech_start()
                    except Exception as e:
                        logger.error("Error in on_speech_start callback: %s", e)
        else:
            self._consecutive_speech_time = 0.0

    def _handle_speech_state(
        self, raw_bytes: bytes, rms: float, is_voice: bool, chunk_duration: float
    ) -> None:
        """Handle state transitions and buffering while currently in SPEECH."""
        self._speech_buffer.append(raw_bytes)

        if self.on_speech_chunk is not None:
            try:
                self.on_speech_chunk(raw_bytes, rms)
            except Exception as e:
                logger.error("Error in on_speech_chunk callback: %s", e)

        if is_voice:
            # Voice is active; reset silence accumulation
            self._consecutive_silence_time = 0.0
        else:
            self._consecutive_silence_time += chunk_duration
            if self._consecutive_silence_time >= self.silence_duration:
                # Transition: SPEECH -> SILENCE
                self._state = VADState.SILENCE
                self._consecutive_speech_time = 0.0
                self._consecutive_silence_time = 0.0

                complete_utterance = b"".join(self._speech_buffer)
                self._speech_buffer.clear()

                # Notify speech end callback
                if self.on_speech_end is not None:
                    try:
                        self.on_speech_end(complete_utterance)
                    except Exception as e:
                        logger.error("Error in on_speech_end callback: %s", e)

    def _check_barge_in(self) -> None:
        """Evaluate if barge-in condition is met and invoke on_barge_in callback."""
        if self.on_barge_in is None:
            return

        is_speaking = False
        if self.is_agent_speaking is not None:
            try:
                is_speaking = bool(self.is_agent_speaking())
            except Exception as e:
                logger.error("Error evaluating is_agent_speaking: %s", e)
        else:
            # If no agent check callable is supplied, any speech onset triggers barge-in
            is_speaking = True

        if is_speaking:
            try:
                self.on_barge_in()
            except Exception as e:
                logger.error("Error in on_barge_in callback: %s", e)

    def reset(self) -> None:
        """Reset state machine, counters, and buffers to initial idle state."""
        with self._lock:
            self._state = VADState.SILENCE
            self._current_energy = 0.0
            self._consecutive_speech_time = 0.0
            self._consecutive_silence_time = 0.0
            self._pre_speech_buffer.clear()
            self._speech_buffer.clear()
