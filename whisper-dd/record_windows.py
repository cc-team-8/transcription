"""Windows PowerShell에서 실행: python record_windows.py"""
import os
import numpy as np
import sounddevice as sd
import soundfile as sf

RECORD_SECONDS = 10
SR = 16000
OUT_PATH = os.path.join(os.path.dirname(__file__), "recordings", "recorded.wav")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
input(f"{RECORD_SECONDS}초 녹음합니다. 준비되면 Enter를 누르세요...")
print("녹음 중... (말하세요)")
audio = sd.rec(RECORD_SECONDS * SR, samplerate=SR, channels=1, dtype="float32")
sd.wait()
sf.write(OUT_PATH, audio.flatten(), SR)
print(f"저장 완료: {OUT_PATH}")
print("이제 WSL2에서 python3 run_client.py 를 실행하세요.")
