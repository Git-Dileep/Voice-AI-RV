"""Speech-to-Text using Faster-Whisper.

Uses a local offline Whisper model for fast transcription without API keys.
"""

import io
import logging
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisperSTT:
    """Transcribes audio locally using faster-whisper."""

    def __init__(self, model_size: str = "base.en", device: str = "auto", compute_type: str = "default"):
        """Initialize the local Faster-Whisper STT engine.

        Args:
            model_size: Model size string (e.g., 'tiny.en', 'base.en').
            device: 'cpu', 'cuda', or 'auto'.
            compute_type: 'int8', 'float16', 'default'.
        """
        logger.info("Loading local Whisper model '%s' (this may take a moment)...", model_size)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Local Whisper model loaded successfully.")

    def transcribe(self, pcm_bytes: bytes) -> str:
        """Transcribe raw PCM audio to text.

        Args:
            pcm_bytes: Raw 16-bit little-endian PCM audio at 16 kHz.

        Returns:
            Transcribed text string, or empty string on failure.
        """
        if not pcm_bytes:
            return ""
            
        try:
            import numpy as np
            # Convert raw 16-bit PCM directly to a normalized float32 numpy array
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Pass array directly to avoid io.BytesIO/av encoding issues
            segments, info = self.model.transcribe(
                audio_array, 
                beam_size=1, 
                vad_filter=True
            )
            
            text = "".join(segment.text for segment in segments).strip()
            logger.debug("Whisper result: %s", text)
            return text
        except Exception as e:
            logger.error("Faster-Whisper transcription failed: %s", e)
            return ""
