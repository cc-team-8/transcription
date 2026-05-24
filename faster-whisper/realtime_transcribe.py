import sounddevice as sd
import numpy as np
import queue
import threading
import sys
from datetime import datetime
from faster_whisper import WhisperModel

# ── 설정 ──────────────────────────────────────────
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
SILENCE_THRESHOLD = 0.01   # RMS 임계값 (낮출수록 더 민감)
SILENCE_SECONDS = 0.8      # 이만큼 조용하면 발화 종료
MIN_SPEECH_SECONDS = 0.4   # 이보다 짧으면 노이즈로 무시
MODEL_SIZE = "small"       # tiny(빠름) / small / medium / large-v3(정확)
MAX_SPEECH_SECONDS = 25    # 이 이상 쌓이면 중간에 잘라 전사 (Whisper 최적 범위)
# ─────────────────────────────────────────────────

SILENCE_BLOCKS = int(SAMPLE_RATE * SILENCE_SECONDS / BLOCK_SIZE)
MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * MIN_SPEECH_SECONDS)
MAX_SPEECH_SAMPLES = int(SAMPLE_RATE * MAX_SPEECH_SECONDS)
LEVEL_BAR_WIDTH = 20

print_lock = threading.Lock()


def ts(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def level_bar(rms_val: float) -> str:
    filled = min(int(rms_val / SILENCE_THRESHOLD * LEVEL_BAR_WIDTH), LEVEL_BAR_WIDTH)
    bar = "█" * filled + "░" * (LEVEL_BAR_WIDTH - filled)
    marker = "●" if rms_val > SILENCE_THRESHOLD else "○"
    return f"{marker} [{bar}]"


def clear_line():
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


def list_microphones():
    exclude = {"loopback", "stereo mix", "스테레오 믹스", "what u hear", "wave out"}
    mics = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        if any(kw in d["name"].lower() for kw in exclude):
            continue
        mics.append((i, d["name"]))
    return mics


def select_microphone():
    mics = list_microphones()
    if not mics:
        raise RuntimeError("사용 가능한 마이크가 없습니다.")
    if len(mics) == 1:
        print(f"마이크 자동 선택: {mics[0][1]}")
        return mics[0][0]
    print("\n사용 가능한 마이크:")
    for idx, (dev_id, name) in enumerate(mics):
        print(f"  {idx}: {name}")
    while True:
        raw = input("번호 선택 (Enter → 0): ").strip()
        choice = int(raw) if raw.isdigit() else 0
        if 0 <= choice < len(mics):
            return mics[choice][0]


def load_model():
    print(f"모델 로딩 중 ({MODEL_SIZE})...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("준비 완료. 말하세요. (Ctrl+C 종료)\n")
    return model


def transcribe(model, audio: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio,
        language="ko",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters={"threshold": 0.45, "min_silence_duration_ms": 200},
        condition_on_previous_text=False,  # 반복 토큰 생성 방지 (핵심)
        compression_ratio_threshold=2.0,   # 반복 많은 세그먼트 버림
        log_prob_threshold=-1.0,
        no_repeat_ngram_size=3,
    )
    return "".join(s.text for s in segments).strip()


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def processor(model, audio_q: queue.Queue):
    speech_buf: list[np.ndarray] = []
    silence_count = 0
    speaking = False
    speech_start: datetime | None = None

    while True:
        chunk: np.ndarray = audio_q.get()
        level = rms(chunk)
        now = datetime.now()

        if level > SILENCE_THRESHOLD:
            if not speaking:
                speaking = True
                speech_start = now
                with print_lock:
                    clear_line()
                    print(f"[{ts(now)}] 🎤 듣는 중...")
            silence_count = 0
            speech_buf.append(chunk)

            # 최대 길이 초과 시 중간 전사 후 버퍼 리셋
            total_samples = sum(len(c) for c in speech_buf)
            if total_samples >= MAX_SPEECH_SAMPLES:
                speech_end = now
                audio = np.concatenate(speech_buf)
                speech_buf = []

                with print_lock:
                    print("⏳ 처리 중... (길어서 중간 전사)")

                text = transcribe(model, audio)

                with print_lock:
                    clear_line()
                    if text:
                        print(f"[{ts(speech_start)} ~ {ts(speech_end)}] {text}")
                    speech_start = now
                    print(f"[{ts(now)}] 🎤 계속 듣는 중...")
        else:
            if not speaking:
                with print_lock:
                    sys.stdout.write(f"\r{level_bar(level)}")
                    sys.stdout.flush()

            if speaking:
                speech_buf.append(chunk)
                silence_count += 1

                if silence_count >= SILENCE_BLOCKS:
                    speech_end = now
                    audio = np.concatenate(speech_buf)
                    speech_buf = []
                    silence_count = 0
                    speaking = False

                    if len(audio) < MIN_SPEECH_SAMPLES:
                        continue

                    with print_lock:
                        print("⏳ 처리 중...")

                    text = transcribe(model, audio)

                    with print_lock:
                        clear_line()
                        if text:
                            print(f"[{ts(speech_start)} ~ {ts(speech_end)}] {text}\n")
                        else:
                            print("(인식 결과 없음)\n")


def main():
    mic_id = select_microphone()
    model = load_model()

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time, status):
        audio_q.put(indata[:, 0].copy())

    t = threading.Thread(target=processor, args=(model, audio_q), daemon=True)
    t.start()

    with sd.InputStream(
        device=mic_id,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        try:
            while True:
                sd.sleep(200)
        except KeyboardInterrupt:
            print("\n\n종료합니다.")


if __name__ == "__main__":
    main()
