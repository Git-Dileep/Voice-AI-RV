"""Local Hardware Voice Core Demo.

Demonstrates and validates Person A Voice Core using the laptop's
REAL microphone and REAL speakers with local mock STT and TTS providers.

Pipeline:
REAL MICROPHONE -> AudioInput -> VAD -> StreamingSTT -> Mock Brain Response -> StreamingTTS -> AudioOutput -> REAL SPEAKERS
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sounddevice as sd

from core.audio_io import AudioInput, AudioOutput
from core.interfaces import STTResult
from core.stt import MockSTTProvider, StreamingSTT
from core.tts import MockTTSProvider, StreamingTTS
from core.vad import VAD


def stop_audio_output(spk: AudioOutput) -> None:
    """Safely stop and clean up AudioOutput stream without lock deadlocks."""
    try:
        spk.interrupt()
        if spk._stream is not None:
            spk._stream.stop()
            spk._stream.close()
            spk._stream = None
        spk._running = False
    except Exception as e:
        print(f"[WARN] Error during speaker cleanup: {e}", file=sys.stderr)


def main() -> None:
    print("=" * 60)
    print("Voice Core Hardware Demo (Person A)")
    print("=" * 60)

    # 1. Query audio devices
    try:
        input_dev = sd.query_devices(kind="input")
        output_dev = sd.query_devices(kind="output")
        print(f"[DEVICE] Mic: {input_dev['name']} (sample rate: {input_dev['default_samplerate']} Hz)")
        print(f"[DEVICE] Speaker: {output_dev['name']} (sample rate: {output_dev['default_samplerate']} Hz)")
    except Exception as e:
        print(f"[ERROR] Could not query audio devices: {e}", file=sys.stderr)
        return

    # 2. Initialize AudioOutput (Real Speakers)
    audio_out = AudioOutput(sample_rate=16000, channels=1, block_size=512)
    try:
        audio_out.start()
        print("[SPK] Speaker stream opened successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to open speaker output stream: {e}", file=sys.stderr)
        return

    # 3. Initialize TTS with MockTTSProvider
    tts_provider = MockTTSProvider(
        sample_rate=16000,
        chunk_size=512,
        samples_per_word=2400,  # ~150ms of audio per word @ 16kHz
        frequency=440.0,
    )

    def on_tts_start(text: str) -> None:
        print("[TTS] Speaking...")

    def on_tts_end() -> None:
        print("[TTS] Finished")

    tts = StreamingTTS(
        provider=tts_provider,
        on_audio_chunk=audio_out.play,
        on_playback_start=on_tts_start,
        on_playback_end=on_tts_end,
    )

    # 4. Initialize STT with MockSTTProvider
    target_transcript = "hello voice core"
    stt_provider = MockSTTProvider(
        default_result=STTResult(text=target_transcript, lang="en", confidence=0.98),
        default_partials=["hello", "hello voice", target_transcript],
    )

    def on_stt_final(text: str, lang: str, confidence: float) -> None:
        print(f"[STT] Final: {text}")
        response = "Voice Core is working."
        print(f"[CORE] Response: {response}")
        tts.speak(response, lang="en")

    stt = StreamingSTT(
        provider=stt_provider,
        on_final=on_stt_final,
    )

    # 5. Initialize VAD (Voice Activity Detection & Barge-in)
    def on_speech_start() -> None:
        print("\n[VAD] Speech started")
        # Ensure STT is primed with the target transcript
        stt_provider.inject_result(
            text=target_transcript,
            lang="en",
            confidence=0.98,
            partials=["hello", "hello voice", target_transcript],
        )
        stt.start_utterance()

    def on_speech_end(audio_bytes: bytes) -> None:
        print("[VAD] Speech ended")
        stt.end_utterance()

    def on_barge_in() -> None:
        print("\n[VAD] Barge-in detected! Stopping speaker playback.")
        tts.interrupt()
        audio_out.interrupt()
        stt.reset()

    vad = VAD(
        sample_rate=16000,
        energy_threshold=float(os.environ.get("VAD_THRESHOLD", "500.0")),
        silence_duration=0.7,
        min_speech_duration=0.1,
        is_agent_speaking=lambda: audio_out.is_playing or tts.is_speaking,
        on_speech_start=on_speech_start,
        on_speech_chunk=lambda chunk, rms: stt.feed_audio(chunk),
        on_speech_end=on_speech_end,
        on_barge_in=on_barge_in,
    )

    # 6. Initialize AudioInput (Real Microphone)
    audio_in = AudioInput(
        sample_rate=16000,
        channels=1,
        block_size=512,
        callback=vad.process_chunk,
    )

    try:
        audio_in.start()
        print("[MIC] Microphone stream opened successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to open microphone input stream: {e}", file=sys.stderr)
        tts.stop()
        stt.stop()
        stop_audio_output(audio_out)
        return

    print("-" * 60)
    print("[MIC] Listening...")
    print("Speak into your microphone. Say anything to trigger a turn.")
    print("While audio is playing, speak again to test barge-in interruption.")
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    # Check for optional runtime duration (for non-interactive validation runs)
    duration = None
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass
    if duration is None and "DEMO_DURATION" in os.environ:
        try:
            duration = float(os.environ["DEMO_DURATION"])
        except ValueError:
            pass

    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)
            if duration is not None and (time.time() - start_time) >= duration:
                print(f"\n[DEMO] Completed scheduled run of {duration:.1f}s.")
                break
    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted by user (Ctrl+C).")
    finally:
        print("\nShutting down Voice Core components...")
        try:
            audio_in.stop()
            print("[MIC] Microphone stream closed.")
        except Exception as e:
            print(f"[WARN] Error stopping mic: {e}")

        try:
            stt.stop()
            print("[STT] STT worker stopped.")
        except Exception as e:
            print(f"[WARN] Error stopping STT: {e}")

        try:
            tts.stop()
            print("[TTS] TTS worker stopped.")
        except Exception as e:
            print(f"[WARN] Error stopping TTS: {e}")

        stop_audio_output(audio_out)
        print("[SPK] Speaker stream closed.")
        print("[DEMO] Voice Core cleanly shut down.")


if __name__ == "__main__":
    main()
