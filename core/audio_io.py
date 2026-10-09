"""Audio I/O layer — microphone capture and speaker playback.

Uses sounddevice for cross-platform audio I/O with 16-bit PCM at 16 kHz mono.
Replaces the original sounddevice-based AudioInput/AudioOutput/AudioIO classes
with a simpler, headless-optimized API.
"""

import io
import wave
import logging
import queue

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 480  # 30 ms frames at 16 kHz — optimal for VAD


class MicrophoneStream:
    """Non-blocking microphone capture yielding raw 16-bit PCM chunks."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=300)
        self._stream: sd.InputStream | None = None
        self._running = False

    # ── sounddevice hardware callback (runs on audio thread) ──

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("Mic status: %s", status)
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            pass  # Drop frame rather than block audio thread

    # ── Lifecycle ──

    def start(self):
        """Open and start the microphone stream."""
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info(
            "Microphone stream started (%d Hz, %d-sample blocks)",
            self.sample_rate,
            self.block_size,
        )

    def stop(self):
        """Stop and close the microphone stream."""
        if not self._running:
            return
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error stopping mic: %s", e)
            self._stream = None

    # ── Reading ──

    def read(self, timeout: float = 0.1) -> bytes | None:
        """Read the next audio chunk. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        """Discard all buffered audio (call after agent finishes speaking)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ── Utility Functions ──


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit PCM bytes in a WAV container for the Whisper API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def play_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    """Play raw signed-16-bit PCM audio through the default speaker.

    Blocks until playback completes.
    """
    if not pcm_bytes or len(pcm_bytes) < 2:
        return
    try:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        sd.play(samples, sample_rate)
        sd.wait()
    except Exception as e:
        logger.error("Playback error: %s", e)
