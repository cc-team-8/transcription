import os
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import httpx
import soundfile as sf
import sounddevice as sd
from datetime import datetime, timezone, timedelta
from client.normalizer import normalize_audio
from client.vad import VADEngine
from client.asr import ASREngine
from client.pipeline import AudioPipeline

SESSION_ID = "meet-001"
PARTICIPANT_ID = "p01"
HUB_URL = "http://localhost:8000"
RECORD_SECONDS = 10
SR = 16000
WAV_PATH = "recordings/recorded.wav"
KST = timezone(timedelta(hours=9))


def record_audio() -> tuple[np.ndarray, datetime]:
    os.makedirs("recordings", exist_ok=True)
    input(f"{RECORD_SECONDS}초 녹음합니다. 준비되면 Enter를 누르세요...")
    print("녹음 중...")
    start_time = datetime.now(KST)
    audio = sd.rec(RECORD_SECONDS * SR, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    sf.write(WAV_PATH, audio, SR)
    print(f"녹음 저장: {WAV_PATH}")
    return audio, start_time


def load_audio(path: str) -> tuple[np.ndarray, datetime]:
    audio, sr = sf.read(path)
    return normalize_audio(audio, src_sr=sr), datetime.now(KST)


vad = VADEngine()
asr = ASREngine(model_size="small")
pipeline = AudioPipeline(vad=vad, asr=asr, session_id=SESSION_ID, participant_id=PARTICIPANT_ID)

try:
    audio, record_start = record_audio()
except Exception as e:
    print(f"마이크 오류: {e}")
    print(f"{WAV_PATH} 파일을 사용합니다.")
    audio, record_start = load_audio(WAV_PATH)

print("전사 중...")
segments = pipeline.process_chunk(audio, time_offset=0.0)

for seg in segments:
    httpx.post(f"{HUB_URL}/segments", json=seg.to_dict())
    t_start = (record_start + timedelta(seconds=seg.start_time)).strftime("%H:%M:%S")
    t_end   = (record_start + timedelta(seconds=seg.end_time)).strftime("%H:%M:%S")
    print(f"[{t_start} ~ {t_end}] {seg.text}")

print(f"완료: {len(segments)}개 세그먼트 전송")
