"""Integration tests for the Voice Core pipeline (VAD -> Streaming STT).

Verifies the integration between the VAD state machine and StreamingSTT coordinator
using synthetic PCM audio and MockSTTProvider without hardware or external APIs.
"""

import threading
import unittest
import numpy as np

from core.stt import MockSTTProvider, StreamingSTT
from core.vad import VAD, VADState


class TestVoiceCorePipeline(unittest.TestCase):
    """Verifies synthetic audio flow through VAD and StreamingSTT."""

    def setUp(self) -> None:
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms per chunk
        self.silence_chunk = np.zeros(self.chunk_size, dtype=np.int16).tobytes()

        # Generate 440 Hz sine wave tone (~5600 RMS, well above VAD 600 threshold)
        t = np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size, endpoint=False)
        self.speech_chunk = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes()

    def test_voice_pipeline_normal_utterance(self) -> None:
        """Test complete turn: silence -> speech onset -> partials -> silence -> finalization."""
        # 1. Setup mock STT provider with injected scenario
        provider = MockSTTProvider()
        expected_text = "turn on the fan"
        expected_lang = "en"
        expected_confidence = 0.98
        expected_partials = ["turn", "turn on", "turn on the fan"]

        provider.inject_result(
            text=expected_text,
            lang=expected_lang,
            confidence=expected_confidence,
            partials=expected_partials,
        )

        partials_received = []
        finals_received = []
        final_event = threading.Event()
        speech_started = False
        speech_ended = False

        # 2. Wire StreamingSTT callbacks
        stt = StreamingSTT(
            provider=provider,
            on_partial=lambda text: partials_received.append(text),
            on_final=lambda text, lang, conf: (
                finals_received.append((text, lang, conf)),
                final_event.set(),
            ),
        )

        def on_speech_start() -> None:
            nonlocal speech_started
            speech_started = True
            stt.start_utterance()

        def on_speech_end(utterance_audio: bytes) -> None:
            nonlocal speech_ended
            speech_ended = True
            stt.end_utterance()

        # 3. Wire VAD
        vad = VAD(
            sample_rate=self.sample_rate,
            energy_threshold=600.0,
            silence_duration=0.3,  # 300ms silence threshold for test speed
            min_speech_duration=0.08,  # ~2.5 chunks required to confirm speech
            on_speech_start=on_speech_start,
            on_speech_chunk=lambda chunk, rms: stt.feed_audio(chunk),
            on_speech_end=on_speech_end,
            on_barge_in=stt.reset,
        )

        try:
            # Phase 1: Silence before speech (5 chunks = ~160ms)
            for _ in range(5):
                state = vad.process_chunk(self.silence_chunk)
                self.assertEqual(state, VADState.SILENCE)

            self.assertFalse(speech_started)
            self.assertEqual(len(partials_received), 0)

            # Phase 2: Speech above threshold (8 chunks = ~256ms)
            for _ in range(8):
                vad.process_chunk(self.speech_chunk)

            self.assertTrue(speech_started, "Speech onset was not detected by VAD")
            self.assertEqual(vad.state, VADState.SPEECH, "VAD should be in SPEECH state")

            # Phase 3: Silence long enough to end the utterance (12 chunks = ~384ms > 300ms)
            for _ in range(12):
                vad.process_chunk(self.silence_chunk)

            self.assertTrue(speech_ended, "Speech conclusion was not detected by VAD")
            self.assertEqual(
                vad.state,
                VADState.SILENCE,
                "Pipeline should return to SILENCE after utterance",
            )

            # Wait for background worker to finalize
            received_final = final_event.wait(timeout=2.0)
            self.assertTrue(received_final, "STT did not emit on_final within timeout")

            # Assertions
            self.assertGreater(len(provider.recorded_audio), 0, "STT received no audio chunks")
            self.assertGreaterEqual(
                len(partials_received),
                1,
                "At least one partial transcript must be emitted",
            )
            self.assertEqual(len(finals_received), 1)

            final_text, final_lang, final_conf = finals_received[0]
            self.assertEqual(final_text, expected_text)
            self.assertEqual(final_lang, expected_lang)
            self.assertAlmostEqual(final_conf, expected_confidence, places=2)

        finally:
            stt.stop()

    def test_voice_pipeline_low_confidence(self) -> None:
        """Test pipeline preserves low confidence scores (e.g. 0.3) for soft-fail handling."""
        provider = MockSTTProvider()
        expected_text = "muffled voice"
        expected_lang = "en"
        expected_confidence = 0.3

        provider.inject_result(
            text=expected_text,
            lang=expected_lang,
            confidence=expected_confidence,
            partials=["muffled"],
        )

        finals_received = []
        final_event = threading.Event()

        stt = StreamingSTT(
            provider=provider,
            on_final=lambda text, lang, conf: (
                finals_received.append((text, lang, conf)),
                final_event.set(),
            ),
        )

        vad = VAD(
            sample_rate=self.sample_rate,
            energy_threshold=600.0,
            silence_duration=0.2,
            min_speech_duration=0.06,
            on_speech_start=stt.start_utterance,
            on_speech_chunk=lambda chunk, rms: stt.feed_audio(chunk),
            on_speech_end=lambda audio: stt.end_utterance(),
        )

        try:
            # Silence
            for _ in range(3):
                vad.process_chunk(self.silence_chunk)

            # Speech
            for _ in range(6):
                vad.process_chunk(self.speech_chunk)

            # Trailing silence to end turn
            for _ in range(10):
                vad.process_chunk(self.silence_chunk)

            self.assertTrue(final_event.wait(timeout=2.0))
            self.assertEqual(len(finals_received), 1)

            final_text, final_lang, final_conf = finals_received[0]
            self.assertEqual(final_text, expected_text)
            self.assertEqual(final_lang, expected_lang)
            self.assertAlmostEqual(final_conf, 0.3, places=2)
            self.assertEqual(vad.state, VADState.SILENCE)

        finally:
            stt.stop()


if __name__ == "__main__":
    unittest.main()
