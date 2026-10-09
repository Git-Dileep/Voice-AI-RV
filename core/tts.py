"""Text-to-Speech using the ElevenLabs API.

Synthesizes response text into raw 16-bit PCM audio at 16 kHz
for direct playback through sounddevice.
"""

import logging
from typing import Iterator

from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel"
DEFAULT_MODEL = "eleven_turbo_v2_5"


class ElevenLabsTTS:
    """Streaming TTS via the ElevenLabs API — outputs raw 16-bit PCM at 16 kHz.

    Each sentence from the LLM is synthesized independently so the first
    sentence plays while the model is still generating the rest, minimizing
    perceived dead-air latency.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL,
    ):
        """Initialize the ElevenLabs TTS client.

        Args:
            api_key: ElevenLabs API key (ELEVENLABS_API_KEY).
            voice_id: Target voice identifier.
            model_id: TTS model identifier (turbo recommended for latency).
        """
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM audio chunks for the given text.

        Args:
            text: Text to synthesize.

        Yields:
            Raw signed-16-bit little-endian PCM audio chunks at 16 kHz.
        """
        if not text or not text.strip():
            return
        try:
            audio_iter = self.client.text_to_speech.convert_as_stream(
                text=text.strip(),
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="pcm_16000",
            )
            for chunk in audio_iter:
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error("ElevenLabs TTS streaming error: %s", e)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to a single PCM audio buffer.

        Collects all streaming chunks into one contiguous byte buffer.

        Args:
            text: Text to synthesize.

        Returns:
            Raw PCM audio bytes, or empty bytes on error.
        """
        if not text or not text.strip():
            return b""
        try:
            # Try non-streaming convert first (more efficient for single sentences)
            result = self.client.text_to_speech.convert(
                text=text.strip(),
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="pcm_16000",
            )
            # Handle both bytes and iterator return types
            if isinstance(result, bytes):
                return result
            return b"".join(chunk for chunk in result if chunk)
        except Exception as e:
            logger.error("ElevenLabs TTS error: %s", e)
            return b""
