import queue
import sys
import threading
import time
import numpy as np
from datetime import datetime
from faster_whisper import WhisperModel
import sounddevice as sd
import pyaudiowpatch as pyaudio
from scipy.signal import resample as scipy_resample

print_lock = threading.Lock()

def erase_line():
    sys.stdout.write("\033[A\033[2K")
    sys.stdout.flush()

SAMPLE_RATE = 16000
BLOCK = 512
SILENCE_THRESHOLD = 0.01
SILENCE_SEC = 0.8
MIN_SPEECH_SEC = 0.4

print("모델 로딩 중 (small)...")
model = WhisperModel("small", device="cpu", compute_type="int8")
model_lock = threading.Lock()
print("준비 완료\n")


def ts():
    return datetime.now().strftime("%H:%M:%S")


def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def do_transcribe(audio: np.ndarray) -> str:
    with model_lock:
        segments, _ = model.transcribe(
            audio,
            language="ko",
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.0,
            no_repeat_ngram_size=3,
            vad_filter=False,
        )
        return "".join(s.text for s in segments).strip()


def vad_worker(audio_q: queue.Queue, label: str, src_sr: int):
    """VAD + 전사 처리 스레드"""
    buf = []
    silence_count = 0
    speaking = False
    speech_start = None

    silence_blocks = int(src_sr * SILENCE_SEC / BLOCK)
    min_blocks = int(src_sr * MIN_SPEECH_SEC / BLOCK)
    prefix = "🎤 나  " if label == "mic" else "🔊 상대"

    while True:
        chunk = audio_q.get()
        if chunk is None:
            break

        level = rms(chunk)

        if level > SILENCE_THRESHOLD:
            if not speaking:
                speaking = True
                speech_start = ts()
                with print_lock:
                    print(f"[{speech_start}] {prefix} | 말 감지...")
            silence_count = 0
            buf.append(chunk)
        elif speaking:
            buf.append(chunk)
            silence_count += 1

            if silence_count >= silence_blocks:
                if len(buf) >= min_blocks:
                    audio = np.concatenate(buf)

                    if src_sr != SAMPLE_RATE:
                        target_len = int(len(audio) * SAMPLE_RATE / src_sr)
                        audio = scipy_resample(audio, target_len).astype(np.float32)

                    text = do_transcribe(audio)
                    with print_lock:
                        erase_line()
                        if text:
                            print(f"[{speech_start} ~ {ts()}] {prefix} | {text}\n")
                else:
                    with print_lock:
                        erase_line()

                buf = []
                silence_count = 0
                speaking = False
                speech_start = None


def find_microphone():
    exclude = {"loopback", "stereo mix", "스테레오 믹스", "what u hear"}
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            if not any(k in d["name"].lower() for k in exclude):
                print(f"마이크: [{i}] {d['name']}")
                return i
    raise RuntimeError("마이크를 찾을 수 없습니다.")


def start_mic_stream(audio_q: queue.Queue):
    mic_id = find_microphone()

    def cb(indata, frames, time, status):
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        device=mic_id, samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=BLOCK, callback=cb,
    )
    stream.start()
    return stream


def find_loopback(p: pyaudio.PyAudio):
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        raise RuntimeError("WASAPI를 찾을 수 없습니다. Windows 환경인지 확인하세요.")

    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    for lb in p.get_loopback_device_info_generator():
        if default_out["name"] in lb["name"]:
            print(f"스피커: [{lb['index']}] {lb['name']}")
            return lb
    raise RuntimeError("스피커 loopback 장치를 찾을 수 없습니다.")


def start_speaker_stream(audio_q: queue.Queue):
    p = pyaudio.PyAudio()
    loopback = find_loopback(p)
    sr = int(loopback["defaultSampleRate"])
    channels = loopback["maxInputChannels"]

    def cb(in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        audio_q.put(audio)
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sr,
        input=True,
        input_device_index=loopback["index"],
        frames_per_buffer=BLOCK,
        stream_callback=cb,
    )
    stream.start_stream()
    return p, stream, sr


if __name__ == "__main__":
    mic_q = queue.Queue()
    spk_q = queue.Queue()

    # 스피커 loopback
    p, spk_stream, spk_sr = start_speaker_stream(spk_q)
    spk_thread = threading.Thread(
        target=vad_worker, args=(spk_q, "speaker", spk_sr), daemon=True
    )
    spk_thread.start()

    # 마이크
    mic_stream = start_mic_stream(mic_q)
    mic_thread = threading.Thread(
        target=vad_worker, args=(mic_q, "mic", SAMPLE_RATE), daemon=True
    )
    mic_thread.start()

    print("\n말하세요... (Ctrl+C 종료)\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        mic_q.put(None)
        spk_q.put(None)
        spk_stream.stop_stream()
        spk_stream.close()
        p.terminate()
        print("\n종료합니다.")
