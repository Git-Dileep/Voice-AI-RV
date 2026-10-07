"""Voice Core Audio Input/Output Layer.

This module provides non-blocking microphone audio capture and interruptible
speaker audio playback using sounddevice and 16-bit PCM streaming.
It is provider-agnostic and designed for low-latency Voice Core components (VAD, STT, TTS).
"""

from contextlib import AbstractContextManager
import logging
import queue
import threading
from typing import Any, Callable, Generator, Optional, Union
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"
DEFAULT_BLOCK_SIZE = 512  # ~32ms at 16 kHz, suitable for VAD and real-time STT


class AudioInput(AbstractContextManager):
    """Captures microphone audio non-blockingly at 16 kHz mono 16-bit PCM."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        callback: Optional[Callable[[bytes], None]] = None,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize microphone input capture.

        Args:
            sample_rate: Sampling frequency in Hz (default: 16000).
            channels: Number of channels (default: 1 for mono).
            block_size: Number of frames per block (default: 512).
            callback: Optional callback invoked with raw PCM bytes for each block.
            device: Optional sounddevice input device index or name.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device
        self._callback = callback

        self._stream: Optional[sd.InputStream] = None
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._running = False
        self._lock = threading.Lock()

    def set_callback(self, callback: Optional[Callable[[bytes], None]]) -> None:
        """Register or update the callback for incoming audio chunks."""
        with self._lock:
            self._callback = callback

    def _stream_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal callback invoked by the sounddevice hardware thread."""
        if status:
            logger.debug("Audio input status flag: %s", status)

        raw_bytes = indata.tobytes()

        # Enqueue for pull-based consumers
        try:
            self._queue.put_nowait(raw_bytes)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(raw_bytes)
            except (queue.Empty, queue.Full):
                pass

        # Trigger push-based callback if registered
        cb = self._callback
        if cb is not None:
            try:
                cb(raw_bytes)
            except Exception as e:
                logger.error("Error in audio input callback: %s", e)

    def start(self) -> None:
        """Start capturing microphone audio."""
        with self._lock:
            if self._running:
                return

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=DEFAULT_DTYPE,
                blocksize=self.block_size,
                device=self.device,
                callback=self._stream_callback,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        """Stop capturing microphone audio and close stream."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.debug("Error stopping audio input stream: %s", e)
                finally:
                    self._stream = None

    @property
    def is_active(self) -> bool:
        """Return True if microphone stream is actively capturing."""
        return self._running and self._stream is not None and self._stream.active

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Retrieve the next captured audio chunk from the queue.

        Args:
            timeout: Maximum seconds to wait for a chunk.

        Returns:
            Raw PCM bytes or None if timeout expired.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_chunks(self) -> Generator[bytes, None, None]:
        """Yield captured audio chunks continuously while the stream is active."""
        while self._running:
            chunk = self.get_chunk(timeout=0.1)
            if chunk is not None:
                yield chunk

    def __enter__(self) -> "AudioInput":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class AudioOutput(AbstractContextManager):
    """Provides interruptible audio playback with instant barge-in flushing."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        device: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize speaker audio output.

        Args:
            sample_rate: Sampling frequency in Hz (default: 16000).
            channels: Number of channels (default: 1 for mono).
            block_size: Number of frames per block (default: 512).
            device: Optional sounddevice output device index or name.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device

        self._stream: Optional[sd.OutputStream] = None
        self._playback_queue: queue.Queue[bytes] = queue.Queue()
        self._buffer = bytearray()
        self._interrupted = threading.Event()
        self._lock = threading.Lock()
        self._running = False

    def _stream_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal callback invoked by the sounddevice output hardware thread."""
        if status:
            logger.debug("Audio output status flag: %s", status)

        bytes_needed = frames * self.channels * 2  # 2 bytes per int16 sample

        with self._lock:
            # If interrupted or no playback is queued, output silence immediately
            if self._interrupted.is_set():
                outdata.fill(0)
                return

            # Refill internal buffer from playback queue
            while len(self._buffer) < bytes_needed:
                try:
                    chunk = self._playback_queue.get_nowait()
                    self._buffer.extend(chunk)
                except queue.Empty:
                    break

            # Write available audio or pad with silence
            available_bytes = len(self._buffer)
            if available_bytes >= bytes_needed:
                chunk_to_play = self._buffer[:bytes_needed]
                del self._buffer[:bytes_needed]
                outdata[:] = np.frombuffer(chunk_to_play, dtype=np.int16).reshape(outdata.shape)
            elif available_bytes > 0:
                # Align to sample boundary (channels * 2 bytes)
                aligned_bytes = available_bytes - (available_bytes % (self.channels * 2))
                chunk_to_play = self._buffer[:aligned_bytes]
                del self._buffer[:aligned_bytes]
                samples = np.frombuffer(chunk_to_play, dtype=np.int16).reshape((-1, self.channels))
                outdata[: len(samples)] = samples
                outdata[len(samples) :].fill(0)
            else:
                outdata.fill(0)

    def start(self) -> None:
        """Start the speaker audio output stream."""
        with self._lock:
            if self._running:
                return

            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=DEFAULT_DTYPE,
                blocksize=self.block_size,
                device=self.device,
                callback=self._stream_callback,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        """Stop the speaker stream and release audio devices."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self.interrupt()
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.debug("Error stopping audio output stream: %s", e)
                finally:
                    self._stream = None

    def play(self, audio_data: Union[bytes, np.ndarray]) -> None:
        """Enqueue raw PCM audio data for playback.

        Args:
            audio_data: Raw 16-bit PCM audio bytes or numpy array.
        """
        if isinstance(audio_data, np.ndarray):
            raw_bytes = audio_data.astype(np.int16).tobytes()
        else:
            raw_bytes = audio_data

        if not raw_bytes:
            return

        with self._lock:
            # Clear interrupted flag to allow new speech to play
            self._interrupted.clear()
            self._playback_queue.put(raw_bytes)

    def interrupt(self) -> None:
        """Immediately stop and flush pending playback for barge-in.

        Drains the playback queue and internal buffer atomically, causing the speaker
        output callback to produce immediate silence on the next audio tick.
        """
        with self._lock:
            self._interrupted.set()
            # Drain queue
            while not self._playback_queue.empty():
                try:
                    self._playback_queue.get_nowait()
                except queue.Empty:
                    break
            # Clear internal byte buffer
            self._buffer.clear()

    @property
    def is_playing(self) -> bool:
        """Return True if audio is actively playing or queued."""
        with self._lock:
            if self._interrupted.is_set():
                return False
            return not self._playback_queue.empty() or len(self._buffer) > 0

    @property
    def is_active(self) -> bool:
        """Return True if speaker stream is open and active."""
        return self._running and self._stream is not None and self._stream.active

    def __enter__(self) -> "AudioOutput":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class AudioIO(AbstractContextManager):
    """Unified audio manager holding both microphone capture and speaker playback streams."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        input_callback: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        """Initialize combined AudioIO manager.

        Args:
            sample_rate: Sampling frequency in Hz (default: 16000).
            channels: Number of channels (default: 1 for mono).
            block_size: Number of frames per block (default: 512).
            input_callback: Optional callback for incoming microphone chunks.
        """
        self.input = AudioInput(
            sample_rate=sample_rate,
            channels=channels,
            block_size=block_size,
            callback=input_callback,
        )
        self.output = AudioOutput(
            sample_rate=sample_rate,
            channels=channels,
            block_size=block_size,
        )

    def start(self) -> None:
        """Start both microphone input and speaker output streams."""
        self.input.start()
        self.output.start()

    def stop(self) -> None:
        """Stop both input and output streams."""
        self.input.stop()
        self.output.stop()

    def interrupt(self) -> None:
        """Trigger immediate speaker playback interruption (barge-in)."""
        self.output.interrupt()

    def __enter__(self) -> "AudioIO":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
