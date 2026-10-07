"""Streaming Text-to-Speech (TTS) Layer.

This module provides a provider-agnostic streaming TTS layer.
It converts response text into streaming 16-bit PCM audio chunks and provides
instant cancellation/interruption for conversational barge-in.
"""

from abc import ABC, abstractmethod
import logging
import queue
import threading
from typing import Any, Callable, Iterator, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_TTS_LANGUAGES = {"en", "hi", "kn"}
DEFAULT_TTS_LANGUAGE = "en"


class TTSProvider(ABC):
    """Abstract base class for TTS synthesis engines (cloud or local)."""

    @abstractmethod
    def synthesize_stream(
        self,
        text: str,
        lang: str = DEFAULT_TTS_LANGUAGE,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[bytes]:
        """Synthesize text and yield raw 16-bit PCM audio chunks.

        Implementations should periodically check cancel_event (if provided)
        and terminate synthesis immediately when cancellation is requested.

        Args:
            text: Text to synthesize into speech audio.
            lang: Target ISO language code ('en', 'hi', 'kn').
            cancel_event: Optional threading.Event signaling cancellation.

        Yields:
            Raw 16-bit PCM audio chunks (typically 16 kHz mono).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal provider state and abort active synthesis."""
        pass


class MockTTSProvider(TTSProvider):
    """Deterministic mock TTS provider generating synthetic PCM audio for testing.

    NOTE: This generates synthetic audio waveforms for pipeline validation,
    NOT actual synthetic human speech.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        samples_per_word: int = 2400,  # 150ms of audio per word @ 16kHz
        frequency: float = 440.0,
    ) -> None:
        """Initialize MockTTSProvider.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000).
            chunk_size: Samples per yielded audio block (default: 512).
            samples_per_word: Synthetic audio duration per word in samples.
            frequency: Base tone frequency in Hz (default: 440.0).
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.samples_per_word = samples_per_word
        self.frequency = frequency
        self._simulate_failure: Optional[Exception] = None
        self._lock = threading.Lock()

    def set_simulated_failure(self, error: Optional[Exception]) -> None:
        """Configure provider to raise an exception on next synthesis attempt."""
        with self._lock:
            self._simulate_failure = error

    def synthesize_stream(
        self,
        text: str,
        lang: str = DEFAULT_TTS_LANGUAGE,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[bytes]:
        """Generate synthetic audio chunks corresponding to the input text."""
        with self._lock:
            if self._simulate_failure:
                err = self._simulate_failure
                self._simulate_failure = None
                raise err

        words = text.strip().split()
        total_words = max(1, len(words))
        total_samples = total_words * self.samples_per_word

        # Vary pitch slightly by language for determinism testing
        freq = self.frequency
        if lang == "hi":
            freq = 523.25  # C5
        elif lang == "kn":
            freq = 659.25  # E5

        samples_generated = 0
        while samples_generated < total_samples:
            if cancel_event is not None and cancel_event.is_set():
                break

            frames_to_generate = min(self.chunk_size, total_samples - samples_generated)
            t = (samples_generated + np.arange(frames_to_generate)) / self.sample_rate
            # 4000 amplitude sine wave (~2800 RMS)
            samples = (np.sin(2 * np.pi * freq * t) * 4000).astype(np.int16)
            samples_generated += frames_to_generate

            yield samples.tobytes()

    def reset(self) -> None:
        """Reset mock provider state."""
        with self._lock:
            self._simulate_failure = None


class StreamingTTS:
    """Provider-agnostic Streaming TTS Coordinator.

    Manages non-blocking audio generation on a background worker thread,
    delivers audio chunks via callbacks, and supports immediate barge-in interruption.
    """

    def __init__(
        self,
        provider: Optional[TTSProvider] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_playback_start: Optional[Callable[[str], None]] = None,
        on_playback_end: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        default_language: str = DEFAULT_TTS_LANGUAGE,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize StreamingTTS coordinator.

        Args:
            provider: Concrete TTSProvider instance (defaults to MockTTSProvider).
            on_audio_chunk: Callback invoked for each synthesized PCM chunk: on_audio_chunk(chunk).
            on_playback_start: Optional callback invoked when synthesis begins: on_playback_start(text).
            on_playback_end: Optional callback invoked when synthesis completes normally.
            on_error: Optional callback invoked on provider errors: on_error(exc).
            default_language: Default language fallback ('en').
            sample_rate: Audio sampling frequency in Hz (default: 16000).
        """
        self.provider = provider or MockTTSProvider(sample_rate=sample_rate)
        self.on_audio_chunk = on_audio_chunk
        self.on_playback_start = on_playback_start
        self.on_playback_end = on_playback_end
        self.on_error = on_error
        self.default_language = default_language
        self.sample_rate = sample_rate

        self._request_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self._cancel_event = threading.Event()
        self._is_speaking = False
        self._done_event = threading.Event()
        self._done_event.set()  # Initially idle
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.start()

    def start(self) -> None:
        """Start the background synthesis worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="StreamingTTSWorker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker thread and clean up."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        self.interrupt()
        self._request_queue.put(("STOP", None))
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def speak(self, text: str, lang: Optional[str] = None) -> None:
        """Synthesize and stream audio chunks for the specified text.

        Non-blocking: enqueues the request and synthesizes on the background worker thread.
        If speech is already playing, cancels the prior utterance first.

        Args:
            text: Text string to speak.
            lang: Language code ('en', 'hi', 'kn'). Defaults to default_language.
        """
        if not text or not text.strip():
            return

        target_lang = lang if (lang and lang in SUPPORTED_TTS_LANGUAGES) else self.default_language

        with self._lock:
            if self._is_speaking:
                self._cancel_active()
            self._cancel_event.clear()
            self._is_speaking = True
            self._done_event.clear()
            self._request_queue.put(("SPEAK", (text.strip(), target_lang)))

    def interrupt(self) -> None:
        """Immediately abort active synthesis and discard queued audio for barge-in."""
        with self._lock:
            self._cancel_active()

    def _cancel_active(self) -> None:
        """Internal helper to signal cancellation and drain request queue."""
        self._cancel_event.set()
        # Drain queue
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.provider.reset()
        except Exception as e:
            logger.debug("Error resetting TTS provider on cancel: %s", e)
        self._is_speaking = False
        self._done_event.set()

    @property
    def is_speaking(self) -> bool:
        """Return True if TTS is actively synthesizing or streaming audio."""
        with self._lock:
            return self._is_speaking

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        """Wait until active synthesis completes or timeout expires.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if idle/completed, False if timed out.
        """
        return self._done_event.wait(timeout=timeout)

    def _worker_loop(self) -> None:
        """Background thread worker loop handling synthesis requests."""
        while self._running:
            try:
                cmd, payload = self._request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if cmd == "STOP":
                break
            elif cmd == "SPEAK":
                if self._cancel_event.is_set():
                    continue

                text, lang = payload
                if self.on_playback_start is not None:
                    try:
                        self.on_playback_start(text)
                    except Exception as e:
                        logger.error("Error in on_playback_start callback: %s", e)

                interrupted = False
                try:
                    stream = self.provider.synthesize_stream(text, lang, self._cancel_event)
                    for chunk in stream:
                        if self._cancel_event.is_set():
                            interrupted = True
                            break

                        if self.on_audio_chunk is not None:
                            try:
                                self.on_audio_chunk(chunk)
                            except Exception as e:
                                logger.error("Error in on_audio_chunk callback: %s", e)
                except Exception as exc:
                    logger.error("TTS provider error during synthesis: %s", exc)
                    if self.on_error is not None:
                        try:
                            self.on_error(exc)
                        except Exception:
                            pass
                    try:
                        self.provider.reset()
                    except Exception:
                        pass
                finally:
                    with self._lock:
                        self._is_speaking = False
                        self._done_event.set()

                    if not interrupted and not self._cancel_event.is_set():
                        if self.on_playback_end is not None:
                            try:
                                self.on_playback_end()
                            except Exception as e:
                                logger.error("Error in on_playback_end callback: %s", e)

    def __enter__(self) -> "StreamingTTS":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
