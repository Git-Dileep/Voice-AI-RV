"""Voice Core Performance and Latency Benchmark.

Measures actual elapsed wall-clock execution time (using time.perf_counter)
across all Person A Voice Core components:
- VAD speech detection
- STT first-partial latency
- STT finalization latency
- TTS first-audio latency
- TTS completion latency
- Barge-in software interruption latency
- End-to-end mock turn to first TTS audio

NOTE: This benchmark measures local software, threading, queue, and provider-mock overhead.
It does NOT represent real microphone, speaker, or cloud network API latency.
"""

import os
import platform
import statistics
import sys
import threading
import time
from typing import Dict, List

# Ensure project root is on sys.path for direct script invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core.stt import MockSTTProvider, StreamingSTT
from core.tts import MockTTSProvider, StreamingTTS
from core.vad import VAD, VADState

ITERATIONS = 20
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms


def generate_audio_chunks():
    """Generate reusable synthetic audio chunks."""
    silence = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
    t = np.linspace(0, CHUNK_SIZE / SAMPLE_RATE, CHUNK_SIZE, endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes()
    return silence, speech


def bench_vad_speech_detection(silence_chunk: bytes, speech_chunk: bytes) -> float:
    """Measure software time required for VAD to detect speech start from incoming chunks."""
    start_event = threading.Event()
    vad = VAD(
        sample_rate=SAMPLE_RATE,
        on_speech_start=lambda: start_event.set(),
    )

    t0 = time.perf_counter()
    # Feed chunks until VAD confirms speech start
    while not start_event.is_set():
        vad.process_chunk(speech_chunk)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed


def bench_stt_first_partial(speech_chunk: bytes) -> float:
    """Measure time from speech ingestion until first interim partial STT callback."""
    provider = MockSTTProvider()
    provider.inject_result(
        text="turn on the fan",
        partials=["turn", "turn on", "turn on the fan"],
    )
    partial_event = threading.Event()
    stt = StreamingSTT(
        provider=provider,
        on_partial=lambda text: partial_event.set(),
    )

    try:
        stt.start_utterance()
        t0 = time.perf_counter()
        # Feed chunks to trigger first partial
        while not partial_event.is_set():
            stt.feed_audio(speech_chunk)
            time.sleep(0.001)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed
    finally:
        stt.stop()


def bench_stt_finalization(speech_chunk: bytes) -> float:
    """Measure time from end_utterance() call until final STT transcript callback."""
    provider = MockSTTProvider()
    provider.inject_result(text="turn on the fan", confidence=0.98)
    final_event = threading.Event()
    stt = StreamingSTT(
        provider=provider,
        on_final=lambda t, l, c: final_event.set(),
    )

    try:
        stt.start_utterance()
        stt.feed_audio(speech_chunk)
        stt.feed_audio(speech_chunk)

        t0 = time.perf_counter()
        stt.end_utterance()
        final_event.wait(timeout=2.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed
    finally:
        stt.stop()


def bench_tts_first_audio() -> float:
    """Measure time from calling speak() until first audio chunk is delivered via callback."""
    provider = MockTTSProvider(chunk_size=CHUNK_SIZE, samples_per_word=1200)
    first_chunk_event = threading.Event()
    tts = StreamingTTS(
        provider=provider,
        on_audio_chunk=lambda c: first_chunk_event.set(),
    )

    try:
        t0 = time.perf_counter()
        tts.speak("hello world", lang="en")
        first_chunk_event.wait(timeout=2.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed
    finally:
        tts.stop()


def bench_tts_completion() -> float:
    """Measure time from speak() until full utterance synthesis is complete and idle."""
    provider = MockTTSProvider(chunk_size=CHUNK_SIZE, samples_per_word=600)
    end_event = threading.Event()
    tts = StreamingTTS(
        provider=provider,
        on_playback_end=lambda: end_event.set(),
    )

    try:
        t0 = time.perf_counter()
        tts.speak("turn on the living room lights", lang="en")
        end_event.wait(timeout=2.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed
    finally:
        tts.stop()


def bench_barge_in_interruption(speech_chunk: bytes) -> float:
    """Measure elapsed time from barge-in trigger until TTS becomes idle."""
    provider = MockTTSProvider(chunk_size=CHUNK_SIZE, samples_per_word=4800)
    first_chunk_event = threading.Event()
    tts = StreamingTTS(
        provider=provider,
        on_audio_chunk=lambda c: first_chunk_event.set(),
    )

    barge_in_fired = threading.Event()
    t_interrupt = 0.0

    def on_barge():
        nonlocal t_interrupt
        t_interrupt = time.perf_counter()
        tts.interrupt()
        barge_in_fired.set()

    vad = VAD(
        sample_rate=SAMPLE_RATE,
        min_speech_duration=0.06,
        is_agent_speaking=lambda: tts.is_speaking,
        on_barge_in=on_barge,
    )

    try:
        tts.speak("this is a long sentence meant to be interrupted by user speech", lang="en")
        first_chunk_event.wait(timeout=1.0)

        # Feed speech to trigger barge-in
        while not barge_in_fired.is_set():
            vad.process_chunk(speech_chunk)

        # Wait until TTS is confirmed idle
        tts.wait_done(timeout=1.0)
        t_idle = time.perf_counter()
        elapsed = (t_idle - t_interrupt) * 1000.0
        return elapsed
    finally:
        tts.stop()


def bench_end_to_end_turn(silence_chunk: bytes, speech_chunk: bytes) -> float:
    """Measure elapsed time for complete turn: VAD -> STT -> Mock Brain -> First TTS audio."""
    stt_provider = MockSTTProvider()
    stt_provider.inject_result(text="turn on the fan", confidence=0.98)

    tts_provider = MockTTSProvider(chunk_size=CHUNK_SIZE, samples_per_word=1200)
    first_audio_event = threading.Event()

    tts = StreamingTTS(
        provider=tts_provider,
        on_audio_chunk=lambda c: first_audio_event.set(),
    )

    def on_stt_final(text, lang, confidence):
        # Mock brain logic: dispatch speech
        tts.speak("Turning on the fan.", lang=lang)

    stt = StreamingSTT(
        provider=stt_provider,
        on_final=on_stt_final,
    )

    vad = VAD(
        sample_rate=SAMPLE_RATE,
        silence_duration=0.15,
        min_speech_duration=0.06,
        on_speech_start=stt.start_utterance,
        on_speech_chunk=lambda c, r: stt.feed_audio(c),
        on_speech_end=lambda a: stt.end_utterance(),
    )

    try:
        t0 = time.perf_counter()
        # Feed speech
        for _ in range(4):
            vad.process_chunk(speech_chunk)
        # Feed silence to trigger speech end
        for _ in range(8):
            vad.process_chunk(silence_chunk)

        first_audio_event.wait(timeout=2.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return elapsed
    finally:
        stt.stop()
        tts.stop()


def compute_stats(data: List[float]) -> Dict[str, float]:
    """Compute min, mean, median, p95, and max statistics."""
    sorted_data = sorted(data)
    p95_idx = int(0.95 * len(sorted_data))
    p95_val = sorted_data[min(p95_idx, len(sorted_data) - 1)]
    return {
        "min": min(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "p95": p95_val,
        "max": max(data),
    }


def format_stats(stats: Dict[str, float]) -> str:
    """Format statistics dictionary in milliseconds."""
    return (
        f"min: {stats['min']:.2f} ms | mean: {stats['mean']:.2f} ms | "
        f"median: {stats['median']:.2f} ms | p95: {stats['p95']:.2f} ms | "
        f"max: {stats['max']:.2f} ms"
    )


def run_benchmark():
    """Execute complete 20-iteration benchmark suite."""
    print("=" * 70)
    print("Voice Core Performance Benchmark")
    print("=" * 70)
    print(f"Iterations: {ITERATIONS}")
    print(f"Python Version: {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print("Benchmark Mode: Synthetic / Mock Local Pipeline (Software & Threading Overhead Only)")
    print("-" * 70)
    print("Running iterations, please wait...")

    silence_chunk, speech_chunk = generate_audio_chunks()

    vad_times = []
    stt_first_partial_times = []
    stt_final_times = []
    tts_first_audio_times = []
    tts_completion_times = []
    barge_in_times = []
    e2e_times = []

    for _ in range(ITERATIONS):
        vad_times.append(bench_vad_speech_detection(silence_chunk, speech_chunk))
        stt_first_partial_times.append(bench_stt_first_partial(speech_chunk))
        stt_final_times.append(bench_stt_finalization(speech_chunk))
        tts_first_audio_times.append(bench_tts_first_audio())
        tts_completion_times.append(bench_tts_completion())
        barge_in_times.append(bench_barge_in_interruption(speech_chunk))
        e2e_times.append(bench_end_to_end_turn(silence_chunk, speech_chunk))

    stats_vad = compute_stats(vad_times)
    stats_stt_partial = compute_stats(stt_first_partial_times)
    stats_stt_final = compute_stats(stt_final_times)
    stats_tts_first = compute_stats(tts_first_audio_times)
    stats_tts_comp = compute_stats(tts_completion_times)
    stats_barge = compute_stats(barge_in_times)
    stats_e2e = compute_stats(e2e_times)

    print("\nPERFORMANCE SUMMARY")
    print("---------------------------------")
    print(f"VAD speech detection:\n  {format_stats(stats_vad)}\n")
    print(f"STT first partial:\n  {format_stats(stats_stt_partial)}\n")
    print(f"STT final:\n  {format_stats(stats_stt_final)}\n")
    print(f"TTS first audio:\n  {format_stats(stats_tts_first)}\n")
    print(f"TTS completion:\n  {format_stats(stats_tts_comp)}\n")
    print(f"Barge-in interruption:\n  {format_stats(stats_barge)}\n")
    print(f"End-to-end to first TTS audio:\n  {format_stats(stats_e2e)}\n")

    # Interpretation
    all_stages = {
        "VAD speech detection": stats_vad["median"],
        "STT first partial": stats_stt_partial["median"],
        "STT final": stats_stt_final["median"],
        "TTS first audio": stats_tts_first["median"],
        "Barge-in interruption": stats_barge["median"],
    }
    fastest_stage = min(all_stages.items(), key=lambda x: x[1])
    slowest_stage = max(all_stages.items(), key=lambda x: x[1])
    barge_in_sub_10ms = stats_barge["max"] < 10.0

    print("=" * 70)
    print("BENCHMARK INTERPRETATION")
    print("=" * 70)
    print(f"1. Fastest Stage: {fastest_stage[0]} (median: {fastest_stage[1]:.2f} ms)")
    print(f"2. Slowest Pipeline Stage: {slowest_stage[0]} (median: {slowest_stage[1]:.2f} ms)")
    print(
        f"3. Software Interruption Latency < 10 ms: "
        f"{'YES' if barge_in_sub_10ms else 'NO'} "
        f"(max observed: {stats_barge['max']:.2f} ms, mean: {stats_barge['mean']:.2f} ms)"
    )
    print("4. Benchmark Limitations:")
    print("   - Measures ONLY in-process Python queue/thread coordination and mock math overhead.")
    print("   - Does NOT include cloud API network round-trip time (ElevenLabs / Gemini Live).")
    print("   - Does NOT include physical microphone driver buffer latency or OS soundcard audio DAC latency.")
    print("   - Serves as the zero-network lower-bound baseline for Person A's software architecture.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
