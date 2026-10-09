"""End-to-End Voice Core integration tests.

Verifies the complete Person A local audio loop:
Synthetic PCM audio -> VAD -> Streaming STT -> Mock Brain Response -> Streaming TTS -> Audio Sink.
Tests full turn execution, conversational barge-in interruption, low-confidence preservation,
and multilingual pipeline processing without physical hardware, cloud APIs, or network access.
"""

import threading
import time
import unittest
import numpy as np

from core.stt import MockSTTProvider, StreamingSTT
from core.tts import MockTTSProvider, StreamingTTS
from core.vad import VAD, VADState


class FakeAudioSink:
    """Mock audio playback sink simulating AudioOutput for testing."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.interrupted: bool = False
        self._lock = threading.Lock()

    def play(self, chunk: bytes) -> None:
        """Receive a playback chunk."""
        with self._lock:
            self.chunks.append(chunk)

    def interrupt(self) -> None:
        """Simulate speaker flushing on barge-in."""
        with self._lock:
            self.interrupted = True

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return len(self.chunks) > 0 and not self.interrupted


class TestVoiceCoreEndToEnd(unittest.TestCase):
    """End-to-end integration tests for Person A Voice Core components."""

    def setUp(self) -> None:
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms
        self.silence_chunk = np.zeros(self.chunk_size, dtype=np.int16).tobytes()

        # 440 Hz tone, amplitude 8000 (~5600 RMS, well above VAD threshold)
        t = np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size, endpoint=False)
        self.speech_chunk = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes()

    def test_complete_voice_turn(self) -> None:
        """Test 1: Full turn from synthetic mic input to transcript to synthesized TTS audio output."""
        stt_provider = MockSTTProvider()
        expected_text = "turn on the fan"
        expected_lang = "en"
        expected_confidence = 0.98

        stt_provider.inject_result(
            text=expected_text,
            lang=expected_lang,
            confidence=expected_confidence,
            partials=["turn", "turn on", "turn on the fan"],
        )

        tts_provider = MockTTSProvider(chunk_size=512, samples_per_word=1200)
        received_tts_chunks = []
        tts_done_event = threading.Event()
        stt_final_data = {}

        tts = StreamingTTS(
            provider=tts_provider,
            on_audio_chunk=lambda chunk: received_tts_chunks.append(chunk),
            on_playback_end=lambda: tts_done_event.set(),
        )

        def handle_stt_final(text: str, lang: str, confidence: float) -> None:
            stt_final_data["text"] = text
            stt_final_data["lang"] = lang
            stt_final_data["confidence"] = confidence

            # Mock Brain decision: respond with spoken confirmation
            mock_response = "Sure, turning on the fan."
            tts.speak(mock_response, lang=lang)

        stt = StreamingSTT(
            provider=stt_provider,
            on_final=handle_stt_final,
        )

        vad = VAD(
            sample_rate=self.sample_rate,
            energy_threshold=600.0,
            silence_duration=0.3,
            min_speech_duration=0.08,
            on_speech_start=stt.start_utterance,
            on_speech_chunk=lambda chunk, rms: stt.feed_audio(chunk),
            on_speech_end=lambda audio: stt.end_utterance(),
            on_barge_in=stt.reset,
        )

        try:
            # 1. User is silent before speaking
            for _ in range(5):
                vad.process_chunk(self.silence_chunk)

            # 2. User speaks "turn on the fan"
            for _ in range(8):
                vad.process_chunk(self.speech_chunk)

            self.assertEqual(vad.state, VADState.SPEECH)

            # 3. User stops speaking -> trailing silence concludes speech turn
            for _ in range(12):
                vad.process_chunk(self.silence_chunk)

            self.assertEqual(vad.state, VADState.SILENCE)

            # 4. Wait for TTS synthesis to complete
            self.assertTrue(tts_done_event.wait(timeout=2.0), "TTS playback did not complete within timeout")

            # 5. Verify final transcript data
            self.assertEqual(stt_final_data.get("text"), expected_text)
            self.assertEqual(stt_final_data.get("lang"), expected_lang)
            self.assertAlmostEqual(stt_final_data.get("confidence", 0.0), expected_confidence, places=2)

            # 6. Verify synthesized TTS audio output
            self.assertGreater(len(received_tts_chunks), 0, "TTS should generate audio chunks")
            self.assertFalse(tts.is_speaking, "TTS should be idle after finishing speech")

            # Validate that generated chunks are valid 16-bit PCM
            for chunk in received_tts_chunks:
                self.assertIsInstance(chunk, bytes)
                self.assertEqual(len(chunk) % 2, 0)
        finally:
            stt.stop()
            tts.stop()

    def test_full_turn_with_barge_in(self) -> None:
        """Test 2: User speech onset mid-playback triggers VAD barge-in, halting TTS and audio sink."""
        tts_provider = MockTTSProvider(chunk_size=512, samples_per_word=4800)  # slow synthesis
        sink = FakeAudioSink()
        first_chunk_event = threading.Event()

        def on_tts_chunk(chunk: bytes) -> None:
            sink.play(chunk)
            first_chunk_event.set()

        tts = StreamingTTS(
            provider=tts_provider,
            on_audio_chunk=on_tts_chunk,
        )

        def on_barge_in_trigger() -> None:
            tts.interrupt()
            sink.interrupt()

        vad = VAD(
            sample_rate=self.sample_rate,
            energy_threshold=600.0,
            silence_duration=0.3,
            min_speech_duration=0.06,
            is_agent_speaking=lambda: tts.is_speaking,
            on_barge_in=on_barge_in_trigger,
        )

        try:
            # 1. Start long agent speech
            tts.speak("one two three four five six seven eight nine ten eleven twelve", lang="en")

            # 2. Confirm audio output begins
            self.assertTrue(first_chunk_event.wait(timeout=1.0), "No TTS chunk emitted")
            self.assertTrue(tts.is_speaking, "TTS should be actively speaking")

            # 3. Simulate user interrupting by speaking into microphone
            for _ in range(4):
                vad.process_chunk(self.speech_chunk)

            # 4. Verify barge-in took effect immediately
            self.assertFalse(tts.is_speaking, "TTS should report idle immediately upon barge-in")
            self.assertTrue(sink.interrupted, "Audio output sink should be marked interrupted")

            chunk_count_at_interrupt = len(sink.chunks)
            time.sleep(0.05)
            self.assertEqual(
                len(sink.chunks),
                chunk_count_at_interrupt,
                "No additional chunks should be emitted after barge-in interruption",
            )

            # 5. Verify TTS reusability after interruption
            sink.chunks.clear()
            sink.interrupted = False
            tts_completed_event = threading.Event()
            tts.on_playback_end = lambda: tts_completed_event.set()

            tts.speak("New response after interruption", lang="en")
            self.assertTrue(tts_completed_event.wait(timeout=2.0), "Second utterance timed out")
            self.assertGreater(len(sink.chunks), 0, "Second utterance must produce audio chunks")
            self.assertFalse(tts.is_speaking)
        finally:
            tts.stop()

    def test_low_confidence_turn(self) -> None:
        """Test 3: Low confidence score (0.3) is preserved and passed without crash or truncation."""
        stt_provider = MockSTTProvider()
        expected_text = "turn on the fan"
        expected_lang = "en"
        low_confidence = 0.3

        stt_provider.inject_result(
            text=expected_text,
            lang=expected_lang,
            confidence=low_confidence,
        )

        stt_final_data = {}
        final_received_event = threading.Event()

        def handle_stt_final(text: str, lang: str, confidence: float) -> None:
            stt_final_data["text"] = text
            stt_final_data["lang"] = lang
            stt_final_data["confidence"] = confidence
            final_received_event.set()

        stt = StreamingSTT(
            provider=stt_provider,
            on_final=handle_stt_final,
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
            # Silence -> Speech -> Silence
            for _ in range(3):
                vad.process_chunk(self.silence_chunk)
            for _ in range(6):
                vad.process_chunk(self.speech_chunk)
            for _ in range(10):
                vad.process_chunk(self.silence_chunk)

            self.assertTrue(final_received_event.wait(timeout=2.0))
            self.assertEqual(stt_final_data.get("text"), expected_text)
            self.assertEqual(stt_final_data.get("lang"), expected_lang)
            self.assertAlmostEqual(stt_final_data.get("confidence", 0.0), low_confidence, places=2)
        finally:
            stt.stop()

    def test_multilingual_voice_core(self) -> None:
        """Test 4: Voice Core STT and TTS handle all required hackathon languages (en, hi, kn)."""
        languages = [("en", "hello world"), ("hi", "namaste"), ("kn", "namaskara")]

        for lang, phrase in languages:
            stt_provider = MockSTTProvider()
            stt_provider.inject_result(text=phrase, lang=lang, confidence=0.95)

            tts_provider = MockTTSProvider(chunk_size=512, samples_per_word=800)
            received_chunks = []
            turn_done_event = threading.Event()
            detected_lang = []

            tts = StreamingTTS(
                provider=tts_provider,
                default_language=lang,
                on_audio_chunk=lambda chunk: received_chunks.append(chunk),
                on_playback_end=lambda: turn_done_event.set(),
            )

            def on_final_cb(text: str, detected: str, confidence: float) -> None:
                detected_lang.append(detected)
                tts.speak(f"Reply in {detected}", lang=detected)

            stt = StreamingSTT(
                provider=stt_provider,
                default_language=lang,
                on_final=on_final_cb,
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
                for _ in range(3):
                    vad.process_chunk(self.silence_chunk)
                for _ in range(6):
                    vad.process_chunk(self.speech_chunk)
                for _ in range(10):
                    vad.process_chunk(self.silence_chunk)

                self.assertTrue(
                    turn_done_event.wait(timeout=2.0),
                    f"Turn failed to complete for language {lang}",
                )
                self.assertEqual(detected_lang, [lang])
                self.assertGreater(
                    len(received_chunks),
                    0,
                    f"No TTS audio produced for language {lang}",
                )
                self.assertFalse(tts.is_speaking)
            finally:
                stt.stop()
                tts.stop()


if __name__ == "__main__":
    unittest.main()
