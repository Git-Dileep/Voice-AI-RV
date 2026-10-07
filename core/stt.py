"""Streaming Speech-to-Text (STT) Layer.

This module provides a provider-agnostic streaming STT component.
It ingests PCM audio chunks from the audio/VAD pipeline, yields partial transcripts,
and emits finalized transcripts with detected language and confidence.
"""

from abc import ABC, abstractmethod
import logging
import queue
import threading
from typing import Any, Callable, List, Optional, Tuple

from core.interfaces import STTResult

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "hi", "kn"}
DEFAULT_LANGUAGE = "en"


class STTProvider(ABC):
    """Abstract base class for STT engines/providers."""

    @abstractmethod
    def start_utterance(self) -> None:
        """Prepare the STT provider for a new utterance turn."""
        pass

    @abstractmethod
    def feed_audio(self, chunk: bytes) -> None:
        """Feed a raw 16-bit PCM audio chunk to the STT provider.

        Args:
            chunk: Raw 16-bit PCM audio bytes.
        """
        pass

    @abstractmethod
    def get_partial(self) -> Optional[str]:
        """Return the current interim/partial transcript if available.

        Returns:
            Partial transcript string or None.
        """
        pass

    @abstractmethod
    def finalize(self) -> STTResult:
        """Conclude recognition for the current utterance and return the final result.

        Returns:
            Final STTResult containing text, language, and confidence.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal buffers and state on cancellation, barge-in, or error."""
        pass


class MockSTTProvider(STTProvider):
    """Deterministic mock STT provider for unit testing and offline development.

    Accepts injected transcript results and emits progressive partials without
    requiring external speech models or network connections.
    """

    def __init__(
        self,
        default_result: Optional[STTResult] = None,
        default_partials: Optional[List[str]] = None,
    ) -> None:
        """Initialize MockSTTProvider.

        Args:
            default_result: Fallback STTResult when no injected results remain.
            default_partials: List of interim partial transcripts to emit progressively.
        """
        self._default_result = default_result or STTResult(
            text="turn on the living room light",
            lang=DEFAULT_LANGUAGE,
            confidence=0.95,
        )
        self._default_partials = default_partials or ["turn on", "turn on the", "turn on the living room light"]

        self._result_queue: List[Tuple[STTResult, List[str]]] = []
        self._current_result: STTResult = self._default_result
        self._current_partials: List[str] = list(self._default_partials)
        self._partial_index = 0
        self._chunk_count = 0
        self._simulate_failure: Optional[Exception] = None
        self.recorded_audio = bytearray()
        self._lock = threading.Lock()

    def set_simulated_failure(self, error: Optional[Exception]) -> None:
        """Configure the provider to raise an exception on next operation."""
        with self._lock:
            self._simulate_failure = error

    def inject_result(
        self,
        text: str,
        lang: str = DEFAULT_LANGUAGE,
        confidence: float = 0.95,
        partials: Optional[List[str]] = None,
    ) -> None:
        """Enqueue a deterministic result to be produced for the next utterance.

        Args:
            text: Finalized transcript text.
            lang: Language code (e.g. 'en', 'hi', 'kn').
            confidence: Recognition confidence score (0.0 to 1.0).
            partials: Optional sequence of partial transcript strings.
        """
        result = STTResult(text=text, lang=lang, confidence=confidence)
        part_list = partials if partials is not None else [text]
        with self._lock:
            self._result_queue.append((result, part_list))

    def start_utterance(self) -> None:
        """Start a new turn, pulling next injected result if available."""
        with self._lock:
            if self._simulate_failure:
                err = self._simulate_failure
                self._simulate_failure = None
                raise err

            if self._result_queue:
                self._current_result, self._current_partials = self._result_queue.pop(0)
            else:
                self._current_result = self._default_result
                self._current_partials = list(self._default_partials)

            self._partial_index = 0
            self._chunk_count = 0
            self.recorded_audio.clear()

    def feed_audio(self, chunk: bytes) -> None:
        """Record audio chunk and advance partial counter."""
        with self._lock:
            if self._simulate_failure:
                err = self._simulate_failure
                self._simulate_failure = None
                raise err

            self.recorded_audio.extend(chunk)
            self._chunk_count += 1
            if self._current_partials and self._chunk_count % 2 == 0:
                if self._partial_index < len(self._current_partials) - 1:
                    self._partial_index += 1

    def get_partial(self) -> Optional[str]:
        """Return next partial string."""
        with self._lock:
            if not self._current_partials:
                return None
            idx = min(self._partial_index, len(self._current_partials) - 1)
            return self._current_partials[idx]

    def finalize(self) -> STTResult:
        """Conclude utterance and return STTResult."""
        with self._lock:
            if self._simulate_failure:
                err = self._simulate_failure
                self._simulate_failure = None
                raise err
            return self._current_result

    def reset(self) -> None:
        """Reset buffers and state."""
        with self._lock:
            self._partial_index = 0
            self._chunk_count = 0
            self.recorded_audio.clear()


class StreamingSTT:
    """Provider-agnostic Streaming STT Coordinator.

    Processes audio chunks asynchronously on a background worker thread
    to keep real-time audio and VAD callback threads non-blocking.
    """

    def __init__(
        self,
        provider: Optional[STTProvider] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str, str, float], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        default_language: str = DEFAULT_LANGUAGE,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize StreamingSTT coordinator.

        Args:
            provider: Concrete STTProvider instance (defaults to MockSTTProvider).
            on_partial: Callback for interim partial transcripts: on_partial(text).
            on_final: Callback for final transcripts: on_final(text, lang, confidence).
            on_error: Callback invoked on provider exceptions: on_error(exc).
            default_language: Default language fallback (default: 'en').
            sample_rate: Audio sample rate in Hz (default: 16000).
        """
        self.provider = provider or MockSTTProvider()
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.default_language = default_language
        self.sample_rate = sample_rate

        self._queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self._last_partial: Optional[str] = None
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._sync_done_event = threading.Event()
        self._sync_result: Optional[STTResult] = None
        self._lock = threading.Lock()

        self.start()

    def start(self) -> None:
        """Start the background processing worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="StreamingSTTWorker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker thread and clean up."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._queue.put(("STOP", None))
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def start_utterance(self) -> None:
        """Signal the start of a speech turn (compatible with VAD on_speech_start)."""
        self._sync_done_event.clear()
        self._sync_result = None
        self._last_partial = None
        self._queue.put(("START", None))

    def feed_audio(self, chunk: bytes) -> None:
        """Enqueue raw PCM audio chunk (compatible with VAD on_speech_chunk).

        Non-blocking: safe to invoke directly from audio/VAD callbacks.

        Args:
            chunk: 16-bit PCM audio bytes.
        """
        if not chunk:
            return
        self._queue.put(("AUDIO", chunk))

    def end_utterance(self) -> None:
        """Signal conclusion of speech turn (compatible with VAD on_speech_end).

        Enqueues finalization request non-blockingly. The worker thread will
        finalize the transcript and trigger on_final.
        """
        self._queue.put(("FINALIZE", None))

    def end_utterance_sync(self, timeout: float = 2.0) -> Optional[STTResult]:
        """Signal end of utterance and synchronously wait for the final result.

        Useful for unit tests and deterministic scripts.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            STTResult or None if timeout expired.
        """
        self._sync_done_event.clear()
        self._queue.put(("FINALIZE_SYNC", None))
        if self._sync_done_event.wait(timeout=timeout):
            return self._sync_result
        return None

    def reset(self) -> None:
        """Immediately reset STT state and drain pending queued audio (for barge-in)."""
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._last_partial = None
        self._sync_done_event.clear()
        self._sync_result = None
        self._queue.put(("RESET", None))

    def _worker_loop(self) -> None:
        """Background thread worker loop handling commands and audio."""
        while self._running:
            try:
                cmd, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if cmd == "STOP":
                    break
                elif cmd == "START":
                    self._last_partial = None
                    self.provider.start_utterance()
                elif cmd == "AUDIO":
                    self.provider.feed_audio(payload)
                    partial = self.provider.get_partial()
                    if partial and partial != self._last_partial:
                        self._last_partial = partial
                        if self.on_partial is not None:
                            try:
                                self.on_partial(partial)
                            except Exception as e:
                                logger.error("Error in on_partial callback: %s", e)
                elif cmd in ("FINALIZE", "FINALIZE_SYNC"):
                    result = self.provider.finalize()
                    # Validate language
                    lang = result.lang if result.lang in SUPPORTED_LANGUAGES else self.default_language
                    final_result = STTResult(
                        text=result.text,
                        lang=lang,
                        confidence=max(0.0, min(1.0, float(result.confidence))),
                    )
                    self._sync_result = final_result
                    self._sync_done_event.set()

                    if self.on_final is not None:
                        try:
                            self.on_final(final_result.text, final_result.lang, final_result.confidence)
                        except Exception as e:
                            logger.error("Error in on_final callback: %s", e)
                elif cmd == "RESET":
                    self.provider.reset()
            except Exception as exc:
                logger.error("Error in STT worker loop: %s", exc)
                if self.on_error is not None:
                    try:
                        self.on_error(exc)
                    except Exception:
                        pass
                try:
                    self.provider.reset()
                except Exception:
                    pass
                self._sync_done_event.set()

    def __enter__(self) -> "StreamingSTT":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
