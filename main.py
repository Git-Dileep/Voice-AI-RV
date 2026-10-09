"""Voice AI Agent — Headless Terminal Application.

Continuous conversation loop:  Listen → Transcribe → Think → Speak

Usage:
    python main.py

All API keys are loaded from the .env file in the project root.
"""

import os
import sys
import time
import logging

from dotenv import load_dotenv

from core.audio_io import MicrophoneStream, pcm_to_wav, play_pcm
from core.vad import VoiceActivityDetector
from core.stt import FasterWhisperSTT
from core.tts import ElevenLabsTTS
from brain.orchestrator import AgentBrain

# ──────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BANNER = r"""
╔════════════════════════════════════════════════════════╗
║                                                        ║
║        🎙️   Voice AI Agent — Terminal Edition   🎙️     ║
║                                                        ║
║   Speak naturally.  I'm listening continuously.        ║
║   Press Ctrl+C to quit.                                ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
"""


# ──────────────────────────────────────────────
# Sentence Buffering (LLM tokens → TTS chunks)
# ──────────────────────────────────────────────

SENTENCE_BOUNDARIES = [". ", "! ", "? ", ".\n", "!\n", "?\n"]


def sentence_buffer(token_stream):
    """Buffer streaming LLM tokens into sentence-sized chunks for TTS.

    Yields each complete sentence as soon as a boundary is detected,
    so the first sentence can be spoken while the LLM is still generating.
    """
    buf = ""
    for token in token_stream:
        buf += token

        while True:
            earliest = -1
            split_len = 0
            for boundary in SENTENCE_BOUNDARIES:
                idx = buf.find(boundary)
                if idx != -1 and (earliest == -1 or idx < earliest):
                    earliest = idx
                    split_len = len(boundary)

            if earliest == -1:
                break

            sentence = buf[: earliest + 1].strip()
            buf = buf[earliest + split_len :]
            if sentence:
                yield sentence

    # Flush any remaining text (last sentence without trailing space)
    remaining = buf.strip()
    if remaining:
        yield remaining


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────


def main():
    # ── 1. Validate API keys ──
    agnes_key = os.getenv("AGNES_API_KEY", "")
    agnes_url = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
    agnes_model = os.getenv("AGNES_MODEL", "agnes-v1")
    stt_key = os.getenv("STT_API_KEY", "")
    stt_base_url = os.getenv("STT_BASE_URL", "https://api.openai.com/v1")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    missing = []
    if not agnes_key or agnes_key.startswith("your_"):
        missing.append("AGNES_API_KEY")
    if not elevenlabs_key or elevenlabs_key.startswith("your_"):
        missing.append("ELEVENLABS_API_KEY")

    if missing:
        print(f"\n  ❌  Missing API keys in .env:  {', '.join(missing)}")
        print("     Edit the .env file and add your real keys, then restart.\n")
        sys.exit(1)

    # ── 2. Initialize components ──
    print(BANNER)

    mic = MicrophoneStream()
    mic.start()
    print("\n  ⏳  Calibrating microphone for 2 seconds... Please remain quiet.")
    mic.drain()
    
    import numpy as np
    t_end = time.time() + 2.0
    rms_vals = []
    while time.time() < t_end:
        chunk = mic.read(0.1)
        if chunk:
            samples = np.frombuffer(chunk, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
            rms_vals.append(rms)
            
    bg_noise = sum(rms_vals) / len(rms_vals) if rms_vals else 0.0
    thresh = bg_noise + 250.0
    print(f"  ✅  Calibration complete! Background noise: {bg_noise:.0f}. VAD threshold: {thresh:.0f}")

    vad = VoiceActivityDetector(energy_threshold=thresh)
    stt = FasterWhisperSTT()
    tts = ElevenLabsTTS(api_key=elevenlabs_key, voice_id=voice_id)
    brain = AgentBrain(api_key=agnes_key, base_url=agnes_url, model=agnes_model)

    logger.info("Components initialized.")
    print("\n  🟢  Microphone active.  Start speaking!\n")

    _listening_printed = False

    try:
        while True:
            # ── 3. Listen (continuous VAD) ──
            chunk = mic.read(timeout=0.1)
            if chunk is None:
                continue

            speech_audio = vad.process_chunk(chunk)

            # Show a live indicator while the user is speaking
            if vad.is_speaking and not _listening_printed:
                print("  🔴  Listening...", end="\r", flush=True)
                _listening_printed = True

            if speech_audio is None:
                continue

            # Speech ended — clear indicator
            _listening_printed = False
            print("                       ", end="\r")

            # ── 4. Transcribe (Whisper STT) ──
            t_stt = time.perf_counter()
            user_text = stt.transcribe(speech_audio)
            stt_latency = time.perf_counter() - t_stt

            if not user_text:
                logger.warning("Empty transcription — background noise? Try again.")
                continue

            print(f"  🗣️   You:   {user_text}   (STT {stt_latency:.1f}s)")

            # ── 5. Think + Speak (streaming LLM → sentence TTS) ──
            t_llm = time.perf_counter()
            token_gen = brain.chat_stream(user_text)

            print("  🤖  Agent:  ", end="", flush=True)
            first_sentence = True

            for sentence in sentence_buffer(token_gen):
                if first_sentence:
                    ttfb = time.perf_counter() - t_llm
                    first_sentence = False

                # Print sentence to terminal
                print(sentence, end=" ", flush=True)

                # Synthesize and play this sentence while LLM continues
                audio = tts.synthesize(sentence)
                play_pcm(audio)

            if not first_sentence:
                print(f"  (TTFB {ttfb:.2f}s)")
            else:
                print()

            # ── 6. Reset for next turn ──
            mic.drain()
            vad.reset()

    except KeyboardInterrupt:
        print("\n\n  👋  Goodbye!\n")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
