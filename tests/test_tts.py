"""Unit and integration tests for StreamingTTS and MockTTSProvider.

Verifies non-blocking streaming TTS, audio chunk generation, language support,
immediate barge-in cancellation, and provider error recovery without hardware or cloud APIs.
"""

import threading
import time
import unittest

from core.tts import (
    DEFAULT_TTS_LANGUAGE,
    SUPPORTED_TTS_LANGUAGES,
    MockTTSProvider,
    StreamingTTS,
)


class TestStreamingTTS(unittest.TestCase):
    """Test suite for StreamingTTS coordinator and MockTTSProvider."""

    def test_basic_streaming(self) -> None:
        """Verify normal synthesis flow: start event, chunk emission, completion, and idle state."""
        provider = MockTTSProvider(chunk_size=512, samples_per_word=1200)
        chunks = []
        start_events = []
        end_event = threading.Event()

        tts = StreamingTTS(
            provider=provider,
            on_audio_chunk=lambda c: chunks.append(c),
            on_playback_start=lambda t: start_events.append(t),
            on_playback_end=lambda: end_event.set(),
        )

        try:
            tts.speak("hello world", lang="en")
            completed = tts.wait_done(timeout=2.0)
            self.assertTrue(completed, "TTS did not complete within timeout")
            self.assertTrue(end_event.is_set(), "on_playback_end callback was not invoked")

            self.assertGreater(len(chunks), 0, "No audio chunks were received")
            for chunk in chunks:
                self.assertIsInstance(chunk, bytes)
                self.assertEqual(len(chunk) % 2, 0, "Chunk byte length must be even for int16 PCM")
                self.assertGreater(len(chunk), 0)

            self.assertFalse(tts.is_speaking, "TTS should be idle after completion")
            self.assertEqual(start_events, ["hello world"])
        finally:
            tts.stop()

    def test_language_support(self) -> None:
        """Verify speech synthesis completes successfully for en, hi, and kn."""
        provider = MockTTSProvider(chunk_size=512, samples_per_word=800)
        chunks_by_lang = {}

        for lang in ("en", "hi", "kn"):
            chunks = []
            end_event = threading.Event()

            tts = StreamingTTS(
                provider=provider,
                on_audio_chunk=lambda c: chunks.append(c),
                on_playback_end=lambda: end_event.set(),
            )

            try:
                tts.speak(f"testing language {lang}", lang=lang)
                self.assertTrue(tts.wait_done(timeout=2.0), f"TTS timed out for language {lang}")
                self.assertTrue(end_event.is_set(), f"Playback end not invoked for language {lang}")
                self.assertGreater(len(chunks), 0, f"No chunks produced for language {lang}")
                chunks_by_lang[lang] = len(chunks)
            finally:
                tts.stop()

        self.assertEqual(len(chunks_by_lang), 3)

    def test_interrupt_and_reusability(self) -> None:
        """Verify interruption halts generation immediately and coordinator remains reusable."""
        # Use higher sample count per word to guarantee time to interrupt mid-utterance
        provider = MockTTSProvider(chunk_size=512, samples_per_word=4800)
        chunks = []
        first_chunk_event = threading.Event()

        def handle_chunk(c: bytes) -> None:
            chunks.append(c)
            first_chunk_event.set()

        tts = StreamingTTS(
            provider=provider,
            on_audio_chunk=handle_chunk,
        )

        try:
            tts.speak("one two three four five six seven eight nine ten eleven twelve", lang="en")
            # Wait for first chunk to begin streaming
            self.assertTrue(first_chunk_event.wait(timeout=1.0), "No chunks emitted before interrupt")
            self.assertTrue(tts.is_speaking, "TTS should be actively speaking before interrupt")

            # Execute barge-in interruption
            tts.interrupt()
            self.assertFalse(tts.is_speaking, "TTS should report is_speaking=False immediately")

            count_at_interrupt = len(chunks)
            time.sleep(0.05)
            self.assertEqual(
                len(chunks),
                count_at_interrupt,
                "No further chunks should be emitted after interrupt()",
            )

            # Test reusability with a subsequent utterance
            chunks.clear()
            end_event = threading.Event()
            tts.on_playback_end = lambda: end_event.set()

            tts.speak("second utterance after interrupt", lang="en")
            self.assertTrue(tts.wait_done(timeout=2.0))
            self.assertTrue(end_event.is_set())
            self.assertGreater(len(chunks), 0, "Second utterance should successfully produce audio")
            self.assertFalse(tts.is_speaking)
        finally:
            tts.stop()

    def test_provider_failure_recovery(self) -> None:
        """Verify provider exceptions trigger on_error and subsequent utterances succeed."""
        provider = MockTTSProvider(chunk_size=512, samples_per_word=800)
        errors = []
        error_event = threading.Event()
        chunks = []

        tts = StreamingTTS(
            provider=provider,
            on_audio_chunk=lambda c: chunks.append(c),
            on_error=lambda err: (errors.append(err), error_event.set()),
        )

        try:
            provider.set_simulated_failure(RuntimeError("test failure"))
            tts.speak("failing utterance", lang="en")

            self.assertTrue(error_event.wait(timeout=1.0), "Error callback was not invoked")
            self.assertTrue(tts.wait_done(timeout=1.0))
            self.assertFalse(tts.is_speaking, "TTS should return to idle on error")
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertEqual(str(errors[0]), "test failure")

            # Subsequent utterance works normally
            chunks.clear()
            end_event = threading.Event()
            tts.on_playback_end = lambda: end_event.set()

            tts.speak("successful recovery utterance", lang="en")
            self.assertTrue(tts.wait_done(timeout=2.0))
            self.assertTrue(end_event.is_set())
            self.assertGreater(len(chunks), 0, "Recovered utterance must produce audio")
            self.assertFalse(tts.is_speaking)
        finally:
            tts.stop()

    def test_supported_languages_and_fallback(self) -> None:
        """Verify supported language sets and safe fallback for unrecognized languages."""
        self.assertIn("en", SUPPORTED_TTS_LANGUAGES)
        self.assertIn("hi", SUPPORTED_TTS_LANGUAGES)
        self.assertIn("kn", SUPPORTED_TTS_LANGUAGES)

        provider = MockTTSProvider(chunk_size=512, samples_per_word=800)
        chunks = []
        end_event = threading.Event()

        tts = StreamingTTS(
            provider=provider,
            default_language="en",
            on_audio_chunk=lambda c: chunks.append(c),
            on_playback_end=lambda: end_event.set(),
        )

        try:
            # Pass unsupported language 'fr'
            tts.speak("bonjour tout le monde", lang="fr")
            self.assertTrue(tts.wait_done(timeout=2.0), "TTS should not crash on unsupported language")
            self.assertTrue(end_event.is_set())
            self.assertGreater(len(chunks), 0, "Should generate audio using default language")
            self.assertFalse(tts.is_speaking)
        finally:
            tts.stop()


if __name__ == "__main__":
    unittest.main()
